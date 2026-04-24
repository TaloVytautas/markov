import numpy as np

data = np.load("trajectory.npy")

[training_data, testing_data] = np.array_split(data, 2)

model = [[0]*100]*100

model = np.array(model)

pi = 0

for i, item in enumerate(training_data, 1):
    model[training_data[i-1]-1][item-1] += 1

print(model)