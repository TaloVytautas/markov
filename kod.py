import numpy as np
import matplotlib.pyplot as plt

data = np.load("trajectory.npy")

[training_data, testing_data] = np.array_split(data, 2)

model = np.zeros((100,100))

for current, next in zip(training_data[:-1], training_data[1:]):
    model[current-1][next-1] += 1

row_sum = np.sum(model, 1, keepdims=True)

def normalise(matrix, sums, b):
    testing_matrix = matrix + b
    testing_sums = sums + b*100
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(testing_sums == 0, 0, testing_matrix/testing_sums)

def testing(model, validation):
    val_curr = validation[:-1] - 1
    val_next = validation[1:] - 1

    probs = model[val_curr, val_next]

    pi_log10 = np.sum(np.log10(probs))
    pi = 10**pi_log10

    return pi, pi_log10

def span(start, end, amount, matrix, sums, validation):
    b_values = np.linspace(start, end, amount)
    pi_values = []

    for b in b_values:
        pi, pi_log10 = testing(normalise(matrix, sums, b), validation)
        pi_values.append((pi, pi_log10))
    
    return list(zip(b_values, pi_values))

#Uppgift 1

pi, pi_log10 = testing(normalise(model, row_sum, 0), testing_data)

print("Pi:", pi)
print("log10(Pi):", pi_log10)

#Uppgift 2

pi, pi_log10 = testing(normalise(model, row_sum, 0.2), testing_data)

print("Pi:", pi)
print("log10(Pi):", pi_log10)

#Uppgift 3

values = span(0, 1, 10000, model, row_sum, testing_data)

greatest_pi_log10_tuple = max(values, key=lambda x: x[1][1])

b_values = [item[0] for item in values]
pi_log10_values = [item[1][1] for item in values]

plt.plot(b_values, pi_log10_values)
plt.show()