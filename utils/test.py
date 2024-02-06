import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from graphConstruct import ConHypergraph
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import HypergraphConv

class MyData(Data):
    def __int__(self, x, edge_index, edge_attr, y, pos, **kwargs):
        super(MyData, self).__int__(x, edge_index, edge_attr, y, pos, **kwargs)

        return super().__init__()


    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.x_s.size(0)

        return super().__inc__(key, value, *args, **kwargs)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index]

    def len(self):
        return self.len

    def get(self, index):
        return self.data[index]

#
# test_list1 = [[1,6,8],
#              [2,5,3],
#              [1,7,9,7]]
# test_list2 = [[8],
#              [2,6,3],
#              [4,5]]
#
# HG1 = ConHypergraph(test_list1, item_num=10)
# HG1_indices = HG1
# row_indices_HG1 = HG1.coalesce().indices()[0]
# col_indices_HG1 = HG1.coalesce().indices()[1]
# values_HG1 = HG1.coalesce().values()#.reshape(-1, 1)
# HG2 = ConHypergraph(test_list2, item_num=10)

#print(type(HG))
# edge_index1 = torch.stack([row_indices_HG1, col_indices_HG1])
batch_size = 2
# data1 = MyData(x=values_HG1, edge_index=edge_index1)
# data2 = MyData(x=HG2.coalesce().values(), edge_index=torch.stack([HG2.coalesce().indices()[0], HG2.coalesce().indices()[1]]))
# test_data = MyDataset([data1, data2])

hyperedge_index1 = torch.tensor([
    [0, 1, 2, 1, 2, 3],
    [0, 0, 0, 1, 1, 1],
])
hyperedge_index2 = torch.tensor([
    [0, 1, 2, 1, 2, 3, 2, 3, 4],
    [0, 0, 0, 1, 1, 1, 2, 2, 2],
])
x1 = torch.randn(4, 12)
x2 = torch.randn(4, 12)
hyperedge_dataset = MyDataset([Data(edge_index=hyperedge_index1, x=x1, pid=17), Data(edge_index=hyperedge_index2, x=x2, pid=2)])
test_loader = DataLoader(hyperedge_dataset, batch_size=batch_size)

for step, batch in enumerate(test_loader):
    pid = batch.pid
    num_graphs = batch.num_graphs
    num_nodes = batch[0].num_nodes  # 获取当前图的节点数量
    num_edges = batch[0].num_edges  # 获取当前图的边数量
    num_node_features = batch[0].num_node_features  # 获取当前图的节点特征数量
    num_edge_features = batch[0].num_edge_features
    num_nodes1 = batch[1].num_nodes  # 获取当前图的节点数量
    num_edges1 = batch[1].num_edges  # 获取当前图的边数量
    num_node_features1 = batch[1].num_node_features  # 获取当前图的节点特征数量
    num_edge_features1 = batch[1].num_edge_features
    edge_index = batch.edge_index
    x = batch.x
    # batch_shape = torch.Size([batch_size*10, batch_size*10])
    # batch_sparse_tensor = torch.sparse.FloatTensor(edge_index, x, torch.Size(batch_shape))
    conv = GCNConv(40, 16)
    x = torch.unsqueeze(x, 0)
    x = conv(x, edge_index)

    print(step)
    print(batch[0])
    print(batch)

