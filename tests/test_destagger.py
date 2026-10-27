import numpy as np

raw = np.array([
    [ 0,  1,  2,  3,  4,  5],
    [10, 11, 12, 13, 14, 15],
    [20, 21, 22, 23, 24, 25],
    [30, 31, 32, 33, 34, 35]
])

pixel_shift_by_row = [2, 1, -1, -2]

destaggered = np.zeros_like(raw)

for roll in range(4):
    destaggered[roll, :] = np.roll(
        raw[roll, :], pixel_shift_by_row[roll]
    )

print("Raw:")
print(raw)

print("\nDestaggered:")
print(destaggered)