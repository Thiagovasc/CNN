import csv
import numpy as np
import pandas as pd

lesion_types: object = {
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'akiec': 'Actinic Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanomic Neves',
    'vasc': 'Vascular Lesion'
}
metadata = pd.read_csv('../data/HAM10000_metadata.csv')
metadata['dx'] = metadata['dx'].map(lesion_types.get)


# Load data
data: np.array = []

with open('../data/hmnist_28_28_RGB.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        data.append(row)
data = np.array(data[1:], dtype=np.float32)

# Separate features
x = data[:, :-1]  # features
y = data[:, -1]   # labels

# Normalizing pixels to binary values
x /= 255.0

# Reshaping x to a 3d array
x = x.reshape(-1, 28, 28, 3)

# Encode labels
num_classes = len(np.unique(y))
y = np.eye(num_classes)[y.astype(np.int32)]
