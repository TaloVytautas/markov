import numpy as np

data = np.load("trajectory.npy")

[training_data, testing_data] = np.array_split(data, 2)

model = np.zeros((100,100))

pi = 1
pi_log = 0

for current, next in zip(training_data[:-1], training_data[1:]):
    model[current-1][next-1] += 1

for row in model:
    divisor = row.sum()
    if divisor != 0:
        row /= divisor

for current, next in zip(testing_data[:-1], testing_data[1:]):
    pi *= model[current-1][next-1]
    pi_log += np.log(model[current-1][next-1])

print(pi)
print(pi_log/np.log(10))