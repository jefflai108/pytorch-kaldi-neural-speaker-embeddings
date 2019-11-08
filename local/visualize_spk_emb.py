from sklearn.manifold import TSNE
from kaldi_io import read_vec_flt_scp
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# example usage
# python scripts/visualize_spk_emb.py spk_embs_2/vctk_spk_resnet_mfcc_3-8_200_32_mean_lde_sqr_asoftmax_m2.scp 108 output.png
# reference: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

tsne = TSNE(n_components=2, verbose=1)
X, y = [], []
index = 0
for key,vec in read_vec_flt_scp(sys.argv[1]):
    X.append(vec)
    y.append(key)
    #y.append(index)
    index += 1
X, y = np.array(X), np.array(y)

X_emb = tsne.fit_transform(X) # tsne transformed

# For reproducability of the results
np.random.seed(42)
N = int(sys.argv[2])
rndperm = np.random.permutation(X_emb.shape[0])
X_emb, y = X_emb[rndperm[:N]], y[rndperm[:N]]

feat_cols = [ 'pixel'+str(i) for i in range(X_emb.shape[1])  ]
df = pd.DataFrame(X_emb,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
df['tsne-1'] = X_emb[:,0]
df['tsne-2'] = X_emb[:,1]
#df['tsne-3'] = X_emb[:,2]

## 2D plot
plt.figure(figsize=(16,10))
sns_plt = sns.scatterplot(
    x="tsne-1", y="tsne-2",
    hue="y",
    palette=sns.color_palette("hls", N),
    data=df,
    legend=False, # “brief”, “full”
    alpha=0.5
)
sns_plt.figure.savefig(sys.argv[3])

## 3D plot
#ax = plt.figure(figsize=(16,10)).gca(projection='3d')
#ax.scatter(
#    xs=df["tsne-1"],
#    ys=df["tsne-2"],
#    zs=df["tsne-3"],
#    c=df["y"],
#    cmap='tab10'
#)
#ax.set_xlabel('tsne-one')
#ax.set_ylabel('tsne-two')
#ax.set_zlabel('tsne-three')
#ax.figure.savefig(sys.argv[3])
