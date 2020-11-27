# In this program, we use K-Means to cluster images of handwritten digits.

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

# loading the data.

digits = datasets.load_digits()
# print(digits.data)
# print(digits.target)

# Creating the model. Note that since there are 10 digits from 0 to 9, our model is supposed to detect 10 clusters.

model = KMeans(n_clusters = 10, random_state = 42)
model.fit(digits.data)

# Now we visualize centroids.

fig = plt.figure(figsize = (8, 3))
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i)
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap = plt.cm.binary)

plt.show()

# As we can see, at indices 0 to 9, respectively, lie the numbers 0, 9, 2, 1, 6, 8, 4, 5, 7, and 3.

# Now let's make a sample handwriting and predict it.
# Also, we need to map clusters to labels.

new_sample_1 = np.array([0.,  0.,  0.,  15.,  10.,  5.,  0.5,  0.,  0.,  0.,  0.5, 10., 15.,  9.,  0.,  0.5,  0.,  0.,
  3., 15., 16.,  6.,  0.,  0.,  0.,  6.5, 15., 15., 14.,  2.,  0.5,  0.,  0.,  0.2,  1., 16.2,
 16.,  3.5,  0.5,  0.,  0.7,  0.,  1., 16., 16.,  6.7,  0.,  0.,  0.5,  0.,  1., 16.3, 16.,  6.9,
  0.,  0.1,  0.,  0.,  0., 11.1, 16., 10.2,  0.,  0.]).reshape(1, -1)

new_sample_2 = digits.data[1].reshape(1, -1)
samples = [new_sample_1, new_sample_2]

new_labels = []

for sample in samples:
    label = model.predict(sample)
    prediction = None
    
    if label == 0:
        prediction = 0
        new_labels.append(0)
    elif label == 1:
        prediction = 9
        new_labels.append(9)
    elif label == 2:
        prediction = 2
        new_labels.append(2)
    elif label == 3:
        prediction = 1
        new_labels.append(1)
    elif label == 4:
        prediction = 6
        new_labels.append(6)
    elif label == 5:
        prediction = 8
        new_labels.append(8)
    elif label == 6:
        prediction = 4
        new_labels.append(4)
    elif label == 7:
        prediction = 5
        new_labels.append(5)
    elif label == 8:
        prediction = 7
        new_labels.append(7)
    elif label == 9:
        prediction = 3
        new_labels.append(3)
    
    print(prediction, '\n')

print('\nThanks for reviewing')

# Thanks for reviewing
