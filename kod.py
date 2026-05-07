import numpy as np
import matplotlib.pyplot as plt

data = np.load("trajectory.npy")

[training_data, testing_data] = np.array_split(data, 2)

model = np.zeros((100,100))

for current, next in zip(training_data[:-1], training_data[1:]):
    model[current-1][next-1] += 1

row_sum = np.sum(model, 1, keepdims=True)

def normalise(matrix, sums, b):
    matrix += b
    sums += b*100
    with np.errstate(divide="ignore"):
        return np.where(sums == 0, 0, matrix/sums)

def testing(model, validation):
    pi = 1
    pi_log10 = 0

    for current, next in zip(validation[:-1], validation[1:]):
        pi *= model[current-1][next-1]
        pi_log10 += np.log(model[current-1][next-1])/np.log(10)

    return pi, pi_log10

print(testing(normalise(model, row_sum, 0), testing_data))