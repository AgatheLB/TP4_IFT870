# %%
# leba3207

from collections import defaultdict

import sklearn
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
reduced_faces = dict()
for i, (tn, nb) in enumerate(zip(faces.target_names, number_target_per_face)):
    # if nb > 40:
    # retrieve first forty indexes of target associated
    positions = np.where(faces.target == i)[0][:40]

    reduced_faces[tn] = []
    for p in positions:
        reduced_faces[tn].append(faces.images[p])

# %%
"""
# 2. Réduction de la dimensionalité des données
"""

# %%
