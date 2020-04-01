# %%
# leba3207

import sklearn
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import DBSCAN
from itertools import product

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
filtered_target = []
for i, (tn, nb) in enumerate(zip(faces.target_names, number_target_per_face)):
    # retrieve first forty indexes of target associated
    positions = np.where(faces.target == i)[0][:40]

    filtered_faces[tn] = []
    for p in positions:
        filtered_faces[tn].append(faces.images[p])
        filtered_data.loc[:, p] = data.iloc[p]
        filtered_target.append(faces.target[p])


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
## a. Méthode Elbow
"""

# %%
K = range(40, 85, 5)
mean_distances = []
for k in K:
    model = KMeans(n_clusters=k)
    model.fit(reduced_data)
    dist_to_best_centroid = np.min(cdist(reduced_data, model.cluster_centers_, 'euclidean'), axis=1)
    mean_distances.append(sum(dist_to_best_centroid) / reduced_data.shape[0])

plt.plot(K, mean_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Mean distance')
plt.title('Elbow Method with K-Means model')
plt.show()

# %%
"""
## b. Approche de validation croisée
"""

# %%

mean_scores = []
for k in K:
    scorer = make_scorer(adjusted_rand_score)
    model = KMeans(n_clusters=k)
    cv_results = cross_validate(model, reduced_data, filtered_target, cv=10, scoring=scorer, n_jobs=-1)
    mean_scores.append(np.mean(cv_results['test_score']))

plt.plot(K, mean_scores, 'bx-')
plt.xlabel('k')
plt.ylabel('Mean score')
plt.title('Cross validation with K-Means model')
plt.show()

# %%
"""
# 4. Analyse avec DBSCAN
## a. Utilisation du coefficient de silhouette
"""

# %%

pairwise_dist = cdist(reduced_data, reduced_data, 'euclidean')
silh_scores = np.zeros(110)
best_params = {}
for i, (e, s) in enumerate(product(range(5, 16), range(1, 11))):
    pred = DBSCAN(eps=e, min_samples=s).fit_predict(reduced_data)
    if len(np.unique(pred)) != 1:
        silh_scores[i] = silhouette_score(pairwise_dist, pred)

        if np.max(silh_scores) == silh_scores[i]:
            best_params['eps'] = e
            best_params['min_sample'] = s

print(f'Les paramètres présentant les meilleurs silhouette score sont: eps {best_params.get("eps")} et min_sample '
      f'{best_params.get("min_sample")}')

# %%
"""
## b. 
"""
