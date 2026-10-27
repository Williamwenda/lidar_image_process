"""
Reading the Ouster scans, the sensor metadata and the static transforms from the ROS 2 bag.
"""
import json

import numpy as np
from scipy.spatial.transform import Rotation

# sensor_msgs/PointField datatype -> numpy dtype
PF_DTYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def decode_cloud(msg, H, W):
    """Organized PointCloud2 -> xyz (H, W, 3) float64 (NaN = no return) and t (H, W) [s since stamp]."""
    fields = {f.name: f for f in msg.fields}
    names = ["x", "y", "z", "t"]
    dtype = np.dtype({
        "names": names,
        "formats": [PF_DTYPES[fields[n].datatype] for n in names],
        "offsets": [fields[n].offset for n in names],
        "itemsize": msg.point_step,
    })
    pts = np.frombuffer(np.asarray(msg.data, dtype=np.uint8).tobytes(), dtype=dtype)
    assert pts.size == H * W, f"cloud has {pts.size} points, expected {H}x{W}"
    pts = pts.reshape(H, W)
    xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=-1).astype(np.float64)
    t = pts["t"].astype(np.float64) * 1e-9
    return xyz, t


def decode_mono16(msg):
    assert msg.encoding in ("mono16", "16UC1"), msg.encoding
    dtype = ">u2" if msg.is_bigendian else "<u2"
    img = np.frombuffer(np.asarray(msg.data, dtype=np.uint8).tobytes(), dtype=dtype)
    return img.reshape(msg.height, msg.step // 2)[:, : msg.width].astype(np.uint16)


def read_metadata(reader, topic="/ouster/metadata"):
    conns = [c for c in reader.connections if c.topic == topic]
    for conn, _, raw in reader.messages(connections=conns):
        return json.loads(reader.deserialize(raw, conn.msgtype).data)
    raise RuntimeError(f"no message on {topic}")


def read_static_transform(reader, target_frame, source_frame, topic="/tf_static"):
    """4x4 T_target_source from /tf_static, so that p_target = T_target_source @ p_source.
    Chains as many static transforms as needed (breadth-first search over the TF tree)."""
    # neighbours[frame] = list of (other_frame, T_frame_other)
    neighbours = {}
    conns = [c for c in reader.connections if c.topic == topic]
    for conn, _, raw in reader.messages(connections=conns):
        for tf in reader.deserialize(raw, conn.msgtype).transforms:
            q = tf.transform.rotation
            tr = tf.transform.translation
            T_parent_child = np.eye(4)
            T_parent_child[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T_parent_child[:3, 3] = [tr.x, tr.y, tr.z]
            parent, child = tf.header.frame_id, tf.child_frame_id
            neighbours.setdefault(parent, []).append((child, T_parent_child))
            neighbours.setdefault(child, []).append((parent, np.linalg.inv(T_parent_child)))

    T_target = {target_frame: np.eye(4)}  # T_target_frame for every frame reached so far
    queue = [target_frame]
    while queue:
        frame = queue.pop(0)
        for other, T_frame_other in neighbours.get(frame, []):
            if other not in T_target:
                T_target[other] = T_target[frame] @ T_frame_other
                queue.append(other)
    if source_frame not in T_target:
        raise RuntimeError(f"no static transform chain between {target_frame} and {source_frame}")
    return T_target[source_frame]


def compute_col_times(t, valid, model):
    """Firing time [s since stamp] of each measurement column, (W,).
    Averaged over the valid pixels of the column (invalid pixels have t = 0); columns without
    any return are interpolated from their neighbours."""
    meas = model.meas_col_of_pixel[valid]
    sums = np.bincount(meas, weights=t[valid], minlength=model.W)
    counts = np.bincount(meas, minlength=model.W)
    has_return = counts > 0
    cols = np.arange(model.W)
    if not has_return.any():
        return cols / model.W * model.scan_period
    col_times = np.zeros(model.W)
    col_times[has_return] = sums[has_return] / counts[has_return]
    return np.interp(cols, cols[has_return], col_times[has_return])


def iter_scans(reader, model, points_topic, signal_topic, reflec_topic, start_ns=None, stop_ns=None):
    """Yield one dict per scan that has a point cloud, a signal image and a reflectivity image with
    the same header stamp:
        stamp_ns  : int, header stamp (start of the scan)
        xyz       : (H, W, 3) float64 raw (motion-distorted) points in os_lidar, NaN = no return
        valid     : (H, W) bool
        range     : (H, W) float32 range of the raw point [m], 0 = no return
        col_times : (W,) float64 firing time of each measurement column [s since stamp]
        signal    : (H, W) uint16
        reflec    : (H, W) uint16
    """
    topics = [points_topic, signal_topic, reflec_topic]
    pending = {topic: {} for topic in topics}  # topic -> {stamp: msg}
    conns = [c for c in reader.connections if c.topic in topics]
    for conn, _, raw in reader.messages(connections=conns, start=start_ns, stop=stop_ns):
        msg = reader.deserialize(raw, conn.msgtype)
        stamp = stamp_ns(msg)
        pending[conn.topic][stamp] = msg
        if not all(stamp in pending[topic] for topic in topics):
            continue

        cloud_msg = pending[points_topic].pop(stamp)
        signal_msg = pending[signal_topic].pop(stamp)
        reflec_msg = pending[reflec_topic].pop(stamp)
        # messages arrive in time order: drop unmatched leftovers from older scans
        for topic in topics:
            for old in [s for s in pending[topic] if s < stamp]:
                del pending[topic][old]

        xyz, t = decode_cloud(cloud_msg, model.H, model.W)
        valid = np.isfinite(xyz).all(axis=-1)
        rng = np.zeros((model.H, model.W), dtype=np.float32)
        rng[valid] = np.linalg.norm(xyz[valid], axis=-1)
        yield dict(
            stamp_ns=stamp,
            xyz=xyz,
            valid=valid,
            range=rng,
            col_times=compute_col_times(t, valid, model),
            signal=decode_mono16(signal_msg),
            reflec=decode_mono16(reflec_msg),
        )
