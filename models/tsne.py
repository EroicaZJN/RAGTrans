from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

"""
读取从模型中保存的tensor文件,由于需要TSNE降维,需要全部转移至cpu上
"""
with open('/home/zjn/mmhg_v4/data/SMP/label111.pickle' , 'rb') as ftest:
# with open('../' + config.test, 'rb') as ftest:
    test_label = pickle.load(ftest)
    # _, _, test_label = pickle.load(ftest)
    test_label=test_label.cpu()
with open('/home/zjn/mmhg_v4/data/SMP/output111.pickle', 'rb') as f:
# with open('./casflow-casflow-asterisk-dense-5.pkl', 'rb') as f:
    # latents = pickle.load(f)
    latents = pickle.load(f)
    latents=latents.cpu()
with open('/home/zjn/mmhg_v4/data/SMP/img_gcn_output111.pickle', 'rb') as f:
# with open('./casflow-casflow-asterisk-dense-5.pkl', 'rb') as f:
    # latents = pickle.load(f)
    latentsimg = pickle.load(f)
    latentsimg=latentsimg.cpu()
# print('hi')




"""
3维降维
"""
latents_reshaped = latents.reshape(464, 1 * 768)
pca = PCA(n_components=50)
latents_reduced = pca.fit_transform(latents_reshaped)
tsne = TSNE(n_components=2, perplexity=80, verbose=1)
hi1 = tsne.fit_transform(latents_reduced)

"""
2维降维,perplexity越大聚合越明显5~50
"""
# hi1 = TSNE(n_components=2, perplexity=80, verbose=1).fit_transform(latents)
# hi2 = TSNE(n_components=2, perplexity=50, verbose=1).fit_transform(latentsimg)
"""
保存降维后二维数据
"""
with open('/home/zjn/mmhg_v4/data/SMP/output111hi80.pkl', 'wb') as f:
    pickle.dump(hi1, f)
# with open('/home/zjn/mmhg_v4/data/SMP/img_gcn_output111hi50.pkl', 'wb') as f:
#     pickle.dump(hi2, f)
with open('/home/zjn/mmhg_v4/data/SMP/text_embedding111hi50.pkl', 'rb') as f:
    hi1 = pickle.load(f)
with open('/home/zjn/mmhg_v4/data/SMP/img_embedding111hi50.pkl', 'rb') as f:
    hi2 = pickle.load(f)
# hi=hi[:512]

fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111,
#                      # projection='2d'
#                      )
# sca = plt.scatter(hi[:, 0], hi[:, 1],c=test_label,alpha=.5)
# c=np.log(test_label)
"""
c为label作颜色渐变,需要与x,y有相同数量
"""
c=np.log(test_label.numpy()).flatten()[:2000]
# c=np.random.choice(c, 2000)
# c=np.full((139200, 3), [255, 0, 0])

x1=hi1[:, 0]
x1=x1[:2000]
y1=hi1[:, 1]
y1=y1[:2000]


x2=hi2[:, 0]
x2=x2[:2000]
y2=hi2[:, 1]
y2=y2[:2000]
# sca = plt.scatter(x, y, c, alpha=.5)
plt.scatter(x1, y1, c=c, marker='o', s=50, cmap='Oranges')
plt.scatter(x2, y2, c=c, marker='o', s=10, cmap='Blues')
plt.axis('off') # 关闭横纵坐标
# plt.colorbar(sca)
# plt.savefig("output10.pdf") # 保存为pdf
plt.show()