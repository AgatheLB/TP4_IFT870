# %%
# leba3207

from collections import defaultdict

import sklearn
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

# %%
#Chargement d'un ensemble de données de faces de personnages connus
from sklearn.datasets import fetch_lfw_people

# %%
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

# %%
faces = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

# %%
#format des images et nombres de clusters
print("Format des images: {}".format(faces.images.shape))
print("Nombre de classes: {}".format(len(faces.target_names)))

# %%
#nombre de données par cluster
number_target_per_face = np.bincount(faces.target)
for i, (nb, nom) in enumerate(zip(number_target_per_face, faces.target_names)):
    print("{0:25} {1:3}".format(nom, nb), end='   ')
    if (i + 1) % 3 == 0:
        print()

# %%
#Affichage des 10 premières faces
fig, axes = plt.subplots(2, 5, figsize=(10, 6),
                         subplot_kw={'xticks': (), 'yticks': ()})
for nom, image, ax in zip(faces.target, faces.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(faces.target_names[nom])

# %%
"""
# 1. Balancement des données
"""

# %%

data = pd.DataFrame(faces.data)

# %%

filtered_faces = dict()
filtered_data = pd.DataFrame()
for i, (tn, nb) in enumerate(zip(faces.target_names, number_target_per_face)):
    # retrieve first forty indexes of target associated
    positions = np.where(faces.target == i)[0][:40]

    filtered_faces[tn] = []
    for p in positions:
        filtered_faces[tn].append(faces.images[p])
        filtered_data.loc[:, p] = data.iloc[p]

filtered_data = filtered_data.T

# %%
"""
# 2. Réduction de la dimensionalité des données
"""

# %%

pca = PCA(n_components=100, whiten=True, random_state=0)
reduced_data = pca.fit_transform(filtered_data)

# %%
"""
# 3. Analyse avec K-Means
## a. méthode Elbow
"""

# %%
mean_distances = []
for k in range(40, 85, 5):
    model = KMeans(n_clusters=k)
    model.fit(reduced_data)
    dist_to_best_centroid = np.min(cdist(reduced_data, model.cluster_centers_, 'euclidean'), axis=1)
    mean_distances.append(sum(dist_to_best_centroid) / reduced_data.shape[0])

plt.plot(range(40, 85, 5), mean_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Mean distance')
plt.title('Elbow Method with K-Means model')
plt.show()