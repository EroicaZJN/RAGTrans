import torch
import pickle
import scipy.sparse as ss
import numpy as np
from torch_geometric.data import Data
from dataLoader import Options, Read_all_cascade
import json
from tqdm import tqdm


def concept_item(category_data='data/SMP/train_category.json'):
    concept2idx = pickle.load(open('data/SMP/concept2idx.pickle', 'rb'))
    pid2idx = pickle.load(open('data/SMP/pid2idx.pickle', 'rb'))
    concept_item_list = [[]]*668
    for i in range(len(concept_item_list)):
        concept_item_list[i] = []
    with open(category_data, 'r') as f:
        line = json.load(f)
        for item in line:
            concept = concept2idx[item['Concept']]
            pid_idx = pid2idx[int(item['Pid'])]
            concept_item_list[concept] += [pid_idx]

    return concept_item_list


def concept_user(category_data='data/SMP/train_category.json'):
    concept2idx = pickle.load(open('data/SMP/concept2idx.pickle', 'rb'))
    uid2idx = pickle.load(open('data/SMP/uid2idx.pickle', 'rb'))
    concept_user_list = [[]]*668
    for i in range(len(concept_user_list)):
        concept_user_list[i] = []
    with open(category_data, 'r') as f:
        line = json.load(f)
        for item in line:
            concept = concept2idx[item['Concept']]
            concept_user_list[concept] += [uid2idx[(item['Uid'])]]

    return concept_user_list


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


'''Friendship network'''
def ConRelationGraph(data):

    options = Options(data)

    _u2idx = {}
    with open(options.u2idx_dict, 'rb') as handle:
        _u2idx = pickle.load(handle)

    n_node = len(_u2idx)

    edges_list = []

    with open(options.net_data, 'r') as handle:
        relation_list = handle.read().strip().split("\n")
        relation_list = [edge.split(',') for edge in relation_list]

        relation_list = [(_u2idx[edge[0]], _u2idx[edge[1]]) for edge in relation_list if
                         edge[0] in _u2idx and edge[1] in _u2idx]
        relation_list_reverse = [edge[::-1] for edge in relation_list]
        edges_list += relation_list_reverse

    edges_list_tensor = torch.LongTensor(edges_list).t()
    edges_weight = torch.ones(edges_list_tensor.size(1)).float()
    graph = Data(edge_index=edges_list_tensor, edge_attr=edges_weight)

    return graph

'''Hypergraph'''
def ConHypergraph(feature2item, item_num=6, feature_num=3):
    #concept2item = concept_item()

    '''Text-C hypergraph'''
    # text_embedding_dict = pickle.load(open('../data/SMP/text_embedding.pickle'))
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in tqdm(range(len(feature2item))):
        items = np.unique(feature2item[j])
        #items = concept2item[j]
        length = len(items)

        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(items[i])
            data.append(1)
            #data.append(text_embedding_dict[items][0])

    H_T = ss.csr_matrix((data, indices, indptr), shape=(len(feature2item), item_num))
    #print(H_T)
    H_T_sum = 1.0 / H_T.sum(axis=1).reshape(1, -1)
    H_T_sum[H_T_sum == float("inf")] = 0

    # BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
    BH_T = H_T.T.multiply(H_T_sum)
    BH_T = BH_T.T
    H = H_T.T

    H_sum = 1.0 / H.sum(axis=1).reshape(1, -1)
    H_sum[H_sum == float("inf")] = 0

    DH = H.T.multiply(H_sum)
    # DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
    DH = DH.T
    HG_FEA = np.dot(DH, BH_T).tocoo()

    # '''Img-C hypergraph'''
    # # img_embedding_dict = pickle.load(open('../data/SMP/img_embedding.pickle'))
    # indptr, indices, data = [], [], []
    # indptr.append(0)
    # for j in tqdm(range(len(concept2item))):
    #     items = np.unique(concept2item[j])
    #
    #     length = len(items)
    #
    #     s = indptr[-1]
    #     indptr.append((s + length))
    #     for i in range(length):
    #         indices.append(items[i])
    #         data.append(1)
    #         #data.append(img_embedding_dict[items][0])
    #
    # H_T = ss.csr_matrix((data, indices, indptr), shape=(len(concept2item), item_num))
    #
    # H_T_sum = 1.0 / H_T.sum(axis=1).reshape(1, -1)
    # H_T_sum[H_T_sum == float("inf")] = 0
    #
    # # BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
    # BH_T = H_T.T.multiply(H_T_sum)
    # BH_T = BH_T.T
    # H = H_T.T
    #
    # H_sum = 1.0 / H.sum(axis=1).reshape(1, -1)
    # H_sum[H_sum == float("inf")] = 0
    #
    # DH = H.T.multiply(H_sum)
    # # DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
    # DH = DH.T
    # HG_Img = np.dot(DH, BH_T).tocoo()


    # concept2user = concept_user()
    # ''' U-C hypergraph'''
    # indptr, indices, data = [], [], []
    # indptr.append(0)
    # for j in tqdm(range(len(concept2user))):
    #     items = np.unique(concept2user[j])
    #
    #     length = len(items)
    #
    #     s = indptr[-1]
    #     indptr.append((s + length))
    #     for i in range(length):
    #         indices.append(items[i])
    #         data.append(1)
    #
    # H_T = ss.csr_matrix((data, indices, indptr), shape=(len(concept2user), user_num))
    #
    # H_T_sum = 1.0 / H_T.sum(axis=1).reshape(1, -1)
    # H_T_sum[H_T_sum == float("inf")] = 0
    #
    # # BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
    # BH_T = H_T.T.multiply(H_T_sum)
    # BH_T = BH_T.T
    # H = H_T.T
    #
    # H_sum = 1.0 / H.sum(axis=1).reshape(1, -1)
    # H_sum[H_sum == float("inf")] = 0
    #
    # DH = H.T.multiply(H_sum)
    # # DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
    # DH = DH.T
    # HG_User = np.dot(DH, BH_T).tocoo()
    HG_FEA = _convert_sp_mat_to_sp_tensor(HG_FEA)
    # HG_User = _convert_sp_mat_to_sp_tensor(HG_User)
    # HG_Text = _convert_sp_mat_to_sp_tensor(HG_Text)
    #HG_Img = _convert_sp_mat_to_sp_tensor(HG_Img)

    return HG_FEA#HG_User, HG_Text#, HG_Img
