import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import pickle
#from torch.utils.data import Dataset
from torch_geometric.data import Dataset
from torch_geometric.data import Data

class Options(object):
    def __init__(self, data_name='WEIXIN2'):
        self.data = 'data/' + data_name + '/all_data.txt'
        self.category = 'data/' + data_name + '/train_category.json'

        self.net_data = 'data/' + data_name + '/edges.txt'

        # train file path.
        self.train_data = 'data/' + data_name + '/train_data.pickle'
        # valid file path.
        self.valid_data = 'data/' + data_name + '/val_data.pickle'
        # test file path.
        self.test_data = 'data/' + data_name + '/test_data.pickle'





def _readFromFile(filename):
    graph_data = []
    #text_embedding_list = pickle.load(open('data/SMP/text_embedding_list.pickle', 'rb'))
    all_data = pickle.load(open(filename, 'rb'))
    ####process the raw data
    for line in all_data:
        pid = torch.LongTensor([int(line[0])])
        uid = torch.LongTensor([int(line[-1])])
        popularity = torch.FloatTensor([float(line[1])])
        related_items = torch.LongTensor(np.array(line[2]))
        hg_idx = torch.LongTensor(np.array(line[3]))
        hg_data = Data(edge_index=hg_idx, y=popularity, related_items=related_items, pid=pid, uid=uid)
        graph_data.append(hg_data)

    return graph_data

def Read_data(data_name, with_EOS=True):

    options = Options(data_name)

    train = _readFromFile(options.train_data)
    valid = _readFromFile(options.valid_data)
    test = _readFromFile(options.test_data)

    return train, valid, test

def Read_target():

    test = _readFromFile('data/SMP/target_data.pickle')

    return test
    
def buildIndex(data):
    user_set = set()
    u2idx = {}
    idx2u = []

    lineid = 0
    for line in open(data):
        lineid += 1
        if len(line.strip()) == 0:
            continue
        chunks = line.strip().split(',')
        for chunk in chunks:
            try:
                if len(chunk.split()) == 2:
                    user, timestamp = chunk.split()
                elif len(chunk.split()) == 3:
                    root, user, timestamp = chunk.split()
                    user_set.add(root)
            except:
                print(line)
                print(chunk)
                print(lineid)
            user_set.add(user)
    pos = 0
    u2idx['<blank>'] = pos
    idx2u.append('<blank>')
    pos += 1
    u2idx['</s>'] = pos
    idx2u.append('</s>')
    pos += 1

    for user in user_set:
        u2idx[user] = pos
        idx2u.append(user)
        pos += 1
    user_size = len(user_set) + 2
    print("user_size : %d" % (user_size))
    return user_size, u2idx, idx2u

def Split_data(data_name, train_rate=0.8, valid_rate=0.1, load_dict=False):

    options = Options(data_name)

    if not load_dict:
        user_size, u2idx, idx2u = buildIndex(options.data)
        with open(options.u2idx_dict, 'wb') as handle:
            pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(options.idx2u_dict, 'wb') as handle:
            pickle.dump(idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(options.u2idx_dict, 'rb') as handle:
            u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            idx2u = pickle.load(handle)

    user_size = len(u2idx)

    t_cascades = []
    timestamps = []
    ####process the raw data
    for line in open(options.data):
        if len(line.strip()) == 0:
            continue
        timestamplist = []
        userlist = []

        chunks = line.strip().strip(',').split(',')
        for chunk in chunks:
            try:
                # Twitter,Douban
                if len(chunk.split()) == 2:
                    user, timestamp = chunk.split()
                # Android,Christianity
                elif len(chunk.split()) == 3:
                    root, user, timestamp = chunk.split()

                    userlist.append((u2idx[root]))
                    timestamplist.append(float(timestamp))
            except:
                print(chunk)

            userlist.append((u2idx[user]))
            timestamplist.append(float(timestamp))

        t_cascades.append(userlist)
        timestamps.append(timestamplist)

    '''ordered by timestamps'''
    order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x: x[1])]
    timestamps = sorted(timestamps)
    t_cascades[:] = [t_cascades[i] for i in order]
    cas_idx = [i for i in range(len(t_cascades))]

    '''data split'''
    train_idx_ = int(train_rate * len(t_cascades))
    train = t_cascades[0:train_idx_]
    train_t = timestamps[0:train_idx_]
    train_idx = cas_idx[0:train_idx_]

    valid_idx_ = int((train_rate + valid_rate) * len(t_cascades))
    valid = t_cascades[train_idx_:valid_idx_]
    valid_t = timestamps[train_idx_:valid_idx_]
    valid_idx = cas_idx[train_idx_:valid_idx_]

    test = t_cascades[valid_idx_:]
    test_t = timestamps[valid_idx_:]
    test_idx = cas_idx[valid_idx_:]

    '''empty folder'''
    with open(options.train_data, 'w') as file:
        file.truncate(0)

    '''write training set '''
    with open(options.train_data, 'w') as f:
        for i in range(len(train)):
            data = list(zip(train[i], train_t[i]))
            data_lst0 = [','.join(map(str, x)) for x in data]
            # data = str(train_idx[i]) + ' ' + ' '.join(data_lst0)
            data = ' '.join(data_lst0)
            f.writelines(data + "\n")

    with open(options.valid_data, 'w') as file:
        file.truncate(0)

    '''write validation set '''
    with open(options.valid_data, 'w') as f:
        for i in range(len(valid)):
            data = list(zip(valid[i], valid_t[i]))
            data_lst0 = [','.join(map(str, x)) for x in data]
            # data = str(valid_idx[i]) + ' ' + ' '.join(data_lst0)
            data = ' '.join(data_lst0)
            f.writelines(data + "\n")

    with open(options.test_data, 'w') as file:
        file.truncate(0)

    '''write testing set '''
    with open(options.test_data, 'w') as f:
        for i in range(len(test)):
            data = list(zip(test[i], test_t[i]))
            data_lst0 = [','.join(map(str, x)) for x in data]
            # data = str(test_idx[i]) + ' ' + ' '.join(data_lst0)
            data = ' '.join(data_lst0)
            f.writelines(data + "\n")

    total_len = sum(len(i) - 1 for i in t_cascades)
    train_size = len(train_t)
    valid_size = len(valid_t)
    test_size = len(test_t)
    print("training size:%d\n   valid size:%d\n  testing size:%d" % (train_size, valid_size, test_size))
    print("total size:%d " % (len(t_cascades)))
    print("average length:%f" % (total_len / len(t_cascades)))
    print('maximum length:%f' % (max(len(cas) for cas in t_cascades)))
    print('minimum length:%f' % (min(len(cas) for cas in t_cascades)))
    print("user size:%d" % (user_size - 2))

    # return user_size, t_cascades, timestamps


def Read_all_cascade(data_name, with_EOS=False):

    options = Options(data_name)

    '''user size'''
    with open(options.u2idx_dict, 'rb') as handle:
        u2idx = pickle.load(handle)
    with open(options.idx2u_dict, 'rb') as handle:
        idx2u = pickle.load(handle)
    user_size = len(u2idx)

    '''load train data, validation data and test data'''
    train, train_t  = _readFromFile(options.train_data, with_EOS)
    valid, valid_t = _readFromFile(options.valid_data, with_EOS)
    test, test_t = _readFromFile(options.test_data, with_EOS)

    all_cascade = train + valid + test

    source_user = []
    for line in all_cascade:
        source_user.append(line[0])

    source_user = list(set(source_user))

    return all_cascade, source_user


class datasets(Dataset):
    def __init__(self, data):

        self.item_num = len(data)
        self.data = data

    def __len__(self):
        return self.item_num

    def __getitem__(self, index):
        # pid = self.data[index][0]
        # uid = self.data[index][1]
        # popularity = self.data[index][-1]
        # img_cluser = self.data[index][3]
        # related_items = np.array([int(x) for x in self.data[index][-1]])

        return self.data[index]

    def len(self):
        return self.item_num

    def get(self, index):
        return self.data[index]


