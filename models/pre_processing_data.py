import csv
import numpy as np

# Load data
data = []
with open('../data/hmnist_28_28_RGB.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        data.append(row)
data = np.array(data[1:], dtype=np.float32)

# Separate features
X = data[:, :-1]  # features
y = data[:, -1]   # labels

# Normalizing pixels to binary values
X /= 255.0

# Reshaping x to a 3d array
X = X.reshape(-1, 28, 28, 3)

# Encode labels
num_classes = len(np.unique(y))
y = np.eye(num_classes)[y.astype(np.int32)]
