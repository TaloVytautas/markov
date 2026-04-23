import numpy as np

data = np.load("trajectory.npy")

[training_data, testing_data] = np.array_split(data, 2)

print(testing_data)