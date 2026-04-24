import numpy as np

data = np.load("trajectory.npy")

[training_data, testing_data] = np.array_split(data, 2)

model = [[0]*100]*100

model = np.array(model)

print(model)
pi = 0