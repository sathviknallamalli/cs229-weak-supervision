import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re

weak_labels = []

fileObject = open("data/perfectweak.txt", "r")
data = fileObject.read()
for prob_label in data.split('\n'):
    if prob_label == '':
        break
    prob_label = prob_label[1:-1]
    prob_label = ' '.join(prob_label.split())
    prob = float(prob_label.split()[1])
    weak_labels.append(prob)

book_reviews = pd.read_csv("data/csvs/bookVectors.csv", skiprows=lambda i: i % 100 != 0)
ood_reviews = pd.read_csv("data/csvs/oodVectors-200-new.csv", skiprows=lambda i: i % 10 != 0)

book_labels = pd.read_csv("data/book_reviews_labels.csv", skiprows=lambda i: i % 100 != 0)
ood_labels = pd.read_csv("data/OOD_reviews.csv", skiprows=lambda i: i % 10 != 0)

embeddings = []
for i in range(len(book_reviews)):
    l = [float(j) for j in book_reviews['vectors'][i].split(':')[:-1]]
    embeddings.append(l)

ood_embeddings = []
for i in range(len(ood_reviews)):
    l = [float(j) for j in ood_reviews['vectors'][i].split(':')[:-1]]
    ood_embeddings.append(l)

pca = PCA(n_components=2, random_state=1)
Y = pca.fit_transform(embeddings)
ood_Y = pca.transform(ood_embeddings)

x_coords = Y[:, 0]
y_coords = Y[:, 1]

x_coords_pos = []
y_coords_pos = []

x_coords_neg = []
y_coords_neg = []

for i in range(len(x_coords)):
    if weak_labels[i] > 0.5:
        x_coords_pos.append(x_coords[i])
        y_coords_pos.append(y_coords[i])
    else:
        x_coords_neg.append(x_coords[i])
        y_coords_neg.append(y_coords[i])

plt.scatter(x_coords_pos, y_coords_pos, color="blue", s=5, label="Positive (Weak Label)")
plt.scatter(x_coords_neg, y_coords_neg, color="red", s=5, label="Negative (Weak Label)")

plt.legend()
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()


# BY CATEGORY
'''
ood_reviews = pd.read_csv("data/csvs/oodVectors-200-new.csv", nrows=50000, skiprows=lambda i: i % 5 != 0)
ood_labels = pd.read_csv("data/OOD_reviews.csv", nrows=50000, skiprows=lambda i: i % 5 != 0)

ood_embeddings = []
for i in range(len(ood_reviews.index)):
    l = [float(j) for j in ood_reviews['vectors'][i].split(':')[:-1]]
    ood_embeddings.append(l)

pca = PCA(n_components=2, random_state=1)
Y = pca.fit_transform(ood_embeddings)

x_coords = Y[:, 0]
y_coords = Y[:, 1]

x_arrays = {}
y_arrays = {}

for i in range(len(x_coords)):
    if ood_labels['category'][i] not in x_arrays:
        x_arrays[ood_labels['category'][i]] = [x_coords[i]]
        y_arrays[ood_labels['category'][i]] = [y_coords[i]]
    else:
        x_arrays[ood_labels['category'][i]].append(x_coords[i])
        y_arrays[ood_labels['category'][i]].append(y_coords[i])

colors = ['blue', 'red', 'orange', 'green', 'purple', 'yellow', 'brown', 'pink', 'indigo', 'navy',
          'black', 'gray', 'maroon', 'lawngreen', 'deepskyblue', 'gold', 'teal']
counter = 0
for key in x_arrays:
    if len(x_arrays[key]) > 500:
        plt.scatter(x_arrays[key], y_arrays[key], color=colors[counter], s=8, label=key)
        counter += 1

plt.legend()
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()
'''
