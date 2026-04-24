import numpy as np

data = np.load("trajectory.npy")

[training_data, testing_data] = np.array_split(data, 2)

model = np.zeros((100,100))

pi = 1

for current, next in zip(training_data[:-1], training_data[1:]):
    model[current-1][next-1] += 1

for row in model:
    sum = row.sum()
    if sum != 0:
        row /= sum

for current, next in zip(testing_data[:-1], testing_data[1:]):
    pi *= model[current-1][next-1]

print(pi)