############
# MIT License
#
# Copyright (c) 2023 Minwoo Seong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


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
