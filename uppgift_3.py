import numpy as np
import random

data = np.load("trajectory.npy")

[training_data, testing_data] = np.array_split(data, 2)

model = np.zeros((100,100))

for current, next in zip(training_data[:-1], training_data[1:]):
    model[current-1][next-1] += 1

best_pi = 1
best_pi_log = -1000000000000000000000000000000000000000000000000
best_b = 0

for i in range(10000):
    testing_model = []
    b = random.uniform(0.0, 10.0)
    pi = 1
    pi_log = 0
    for row in model:
        divisor = row.sum() + b*100
        testing_row = row + b
        testing_row /= divisor
        testing_model.append(testing_row)
    
    for current, next in zip(testing_data[:-1], testing_data[1:]):
        pi *= testing_model[current-1][next-1]
        pi_log += np.log(testing_model[current-1][next-1])
    
    if best_pi_log < pi_log:
        best_pi_log = pi_log
        best_pi = pi
        best_b = b

print(best_pi)
print(best_pi_log/np.log(10))
print(best_b)