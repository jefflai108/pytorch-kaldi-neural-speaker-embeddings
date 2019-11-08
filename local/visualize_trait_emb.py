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
# python local/visualize_trait_emb.py age/accent/gender exp/vctk_lde/resnet_mfcc_3-8_200_32_mean_lde_sqr_asoftmax_m2/lde.scp 43873 output.png
# reference: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
# speaker-info.txt
# ID  AGE  GENDER  ACCENTS  REGION
# 225  23  F    English    Southern  England
# 226  22  M    English    Surrey
# 227  38  M    English    Cumbria
# 228  22  F    English    Southern  England
# 229  23  F    English    Southern  England
# 230  22  F    English    Stockton-on-tees
# 231  23  F    English    Southern  England
# 232  23  M    English    Southern  England

#speaker_info = '/export/c01/jlai/nii/spk_enc/Erica_VCTK_processed/vctk-speaker-info.txt'
speaker_info = '/data/sls/scratch/clai24/data/Erica_VCTK_processed/vctk-speaker-info.txt'

with open(speaker_info, 'r') as f:
    context = f.readlines()
context = [x.strip() for x in context][1:]
spk2trait = {}
for i in context:
    spk = i.split()[0]
    if spk != 's5': # add prefix 'p'
        spk = 'p' + spk
    if sys.argv[1] == 'age':
        trait = int(i.split()[1])
    elif sys.argv[1] == 'gender':
        trait = i.split()[2]
    elif sys.argv[1] == 'accent':
        trait = i.split()[3]
    spk2trait[spk] = trait
print('speaker to trait is %s' % spk2trait)

tsne = TSNE(n_components=2, verbose=1)
X, y = [], []
index = 0
for key,vec in read_vec_flt_scp(sys.argv[2]):
    X.append(vec)
    spk = key.split('-')[0]
    y.append(spk2trait[spk])
    #print(vec.shape)
    #y.append(index)
    index += 1
X, y = np.array(X), np.array(y)
print(len(y))
print(np.unique(y))
X_emb = tsne.fit_transform(X) # tsne transformed

# For reproducability of the results
np.random.seed(42)
N = int(sys.argv[3])
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
    palette=sns.color_palette("hls", len(np.unique(y))),
    data=df,
    legend='brief', # “brief”, “full”
    alpha=0.5
)
sns_plt.figure.savefig(sys.argv[4])

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
