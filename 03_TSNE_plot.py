from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import h5py
import plotly.express as px

dataPath = './data_processed/Backhand_allSensors.hdf5'
fin = h5py.File(dataPath, 'r')
savePath = './tnse/noBody_Backhand.png'

feature_matrices = np.array(fin['example_matrices'])
feature_target = np.array(fin['example_expert_label_indexes'])
feature_matrices = feature_matrices[:, :, 0:58]
# feature_matrices = np.concatenate((feature_matrices[:, :, 0:22], feature_matrices[:, :, 58:121]), axis=2)
print(feature_matrices.shape)

feature_matrices = feature_matrices.reshape((len(feature_matrices), -1))
df = pd.DataFrame(feature_matrices)
df.isnull().sum()
print(df.isnull().sum())

tsne = TSNE(random_state=42, perplexity=80, n_iter=300, n_components=2).fit_transform(feature_matrices)

plt.scatter(tsne[:, 0], tsne[:, 1], c=feature_target, cmap='viridis')

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig(savePath)