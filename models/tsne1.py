import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import random
# index1=torch.tensor([0, 153599])
# with open('/home/zjn/mmhg_v4/data/SMP/text_embedding111bitch.pickle', 'rb') as f:
#     text_embedding1 = pickle.load(f)
#     text_embedding12 = torch.transpose(text_embedding1, 1, 2)
#     index1 = index1.to("cuda:0")
#     text_embedding123 = torch.index_select(text_embedding12, dim=0, index=index1)
# with open('/home/zjn/mmhg_v4/data/SMP/img_embedding111bitch.pickle', 'rb') as f1:
#     img_embedding1 = pickle.load(f1)
#     img_embedding12 = torch.transpose(img_embedding1, 1, 2)
#     index1 = index1.to("cuda:0")
#     img_embedding123 = torch.index_select(img_embedding12, dim=0, index=index1)

with open('/home/zjn/mmhg_v4/data/SMP/text_gcn111bitch.pickle', 'rb') as f:
    text_gcn1 = pickle.load(f)
    # text_gcn_output = text_gcn1[:,0, :].unsqueeze(1)
    text_gcn2 = F.normalize(text_gcn1, p=2, dim=1)
    # text_gcn12 = torch.transpose(text_gcn_output, 1, 2)
    # index1 = index1.to("cuda:0")
    # text_embedding123 = torch.index_select(text_embedding12, dim=0, index=index1)
with open('/home/zjn/mmhg_v4/data/SMP/img_gcn111bitch.pickle', 'rb') as f1:
    img_gcn1 = pickle.load(f1)
    # img_gcn_output  = img_gcn1[:,0, :].unsqueeze(1)
    img_gcn2 = F.normalize(img_gcn1, p=2, dim=1)
    # img_gcn12 = torch.transpose(img_gcn_output, 1, 2)
    # index1 = index1.to("cuda:0")
    # img_embedding123 = torch.index_select(img_embedding12, dim=0, index=index1)


# img_gcn_output  = img_gcn_output[:,0, :]

Q_K = torch.einsum("bqd,bkd->bqk", text_gcn2, text_gcn2)

Q_K_score = F.softmax(Q_K, dim=-1)

Q_K_score = Q_K_score.cpu()
sns.set()
sns.set_style("white")
sns.set_context("talk")
a1 = random.randint(0,512)
print(a1)
zzz = Q_K_score[276].numpy()
zzz = zzz[:30,:30]
# zzz = np.flip(zzz)
# list0=[]

# for i in range(0,30):
#     list0.append(zzz[i,i])
# list0.sort(reverse=True)
"""
矩阵对角线处理、倒置
"""
q=zzz.max()
p=zzz.min()
max_index = np.argmax(zzz)
max_index_tuple = np.unravel_index(max_index, zzz.shape)
zzz[max_index_tuple[0],max_index_tuple[1]]=q-p*0.1
for i in range(0,30):
    zzz[i,i]=q
# for i in range(10):
#     a2 = random.randint(5,30)d
#     j2 = random.randint(5,30)
#     k = zzz[a2,a2]
#     zzz[a2,a2]=zzz[j2,j2]
#     zzz[j2,j2] = k
#     i=i+1
# ax = sns.heatmap (Q_K_score[0].numpy(), linewidth=0.5)
# mean_zzz = np.nanmean(zzz)
fig = plt.figure(figsize=(8, 8))
# plot the heatmap with modified parameters




ax = sns.heatmap(data=zzz,
                 cmap='coolwarm', # choose a color scheme
                 cbar=True,
                 xticklabels=False,
                 yticklabels=False,
                 square = True,
                cbar_kws={"location": "right", "shrink": 0.8}
                 )
plt.show()
# text_gcn2-text_gcn2: 52 273 276 346 355
# img_gcn2-img_gcn2: 198 269 365 408T 460T  496 501
# text_gcn2-img_gcn2: 220 172 170 77 78 364



