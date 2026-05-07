import numpy as np
import matplotlib.pyplot as plt

data = np.load("trajectory.npy")

[training_data, testing_data] = np.array_split(data, 2)

model = np.zeros((100,100))

for current, next in zip(training_data[:-1], training_data[1:]):
    model[current-1][next-1] += 1

row_sum = np.sum(model, 0)

for row in model/row_sum:
    print(row)
