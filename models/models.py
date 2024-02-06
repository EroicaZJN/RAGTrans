import math
import pickle

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import HypergraphConv

from torch_geometric.nn import GATConv
from Optim import ScheduledOptim

from models.TransformerBlock import TransformerBlock


#from Hypergraph_Transformer import Hypergraph_TransformerConv
'''To GPU'''
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

'''To CPU'''
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

'''Mask previous activated users'''
def get_previous_user_mask(seq, user_size):
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    return masked_seq.cuda()

''''Load embedding from pickle file'''
def load_embedding(format, data_name):
    if format == 'text':
        path = 'data/'+data_name+'/text_embedding_list.pickle'
        text_embedding_list = pickle.load(open(path, 'rb'))
        return text_embedding_list
    elif format == 'img':
        path = 'data/'+data_name+'/img_embedding_list.pickle'
        img_embedding_list = pickle.load(open(path, 'rb'))
        return img_embedding_list

'''Learn friendship network'''
class GraphATT(nn.Module):
    def __init__(self, ninp, nout, layers):
        super(GraphATT, self).__init__()

        self.ninp = ninp
        self.nout = nout
        self.gnn_list = [GATConv(ninp, nout).cuda() for _ in range(layers)]
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.nout)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, graph, embedding, layer):
        graph_edge_index = graph.edge_index
        graph_output = self.gnn_list[layer](embedding, graph_edge_index)
        return graph_output

class TimeEncode(nn.Module):
    def __init__(self, time_dim, factor=5):
        super(TimeEncode, self).__init__()

        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic

class fusion(nn.Module):
    def __init__(self, input_size, dropout=0.2):
        super(fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out


class MMHG(nn.Module):
    def __init__(self, args, dropout=0.2):
        super(MMHG, self).__init__()

        # parameters
        self.emb_size = args.embSize
        self.pos_dim = args.posSize
        #self.n_node = args.n_node
        self.layers = args.layer
        self.dropout = nn.Dropout(dropout)
        self.data_name = args.data_name
        self.att_head = 2
        self.batch_size = args.batch_size
        #self.n_channel = len(hypergraphs)#+1  ## Hypergraph 2, Social graph

        # graph and hypergraph
        # self.adjacency = adjacency     #social graph
        # self.H_User = hypergraphs[0] #categoryu-user hypergraph
        # self.H_Text = hypergraphs[1]   #category-text hypergraph
        #self.H_Img = hypergraphs[2]     #category-img hypergraph

        ###### user and position embedding
        #self.user_embedding = nn.Embedding(self.n_node, self.emb_size, padding_idx=0).cuda()
        #self.pid_embedding = nn.Embedding(305613, self.emb_size, padding_idx=0).cuda()
        self.text_embedding = torch.from_numpy(np.array(load_embedding('text', self.data_name))).cuda().to(torch.float32)
        self.img_embedding = torch.from_numpy(np.array(load_embedding('img', self.data_name))).cuda().to(torch.float32)

        #self.hgtrmConv = Hypergraph_TransformerConv(128, 32)
        self.hgConv1 = HypergraphConv(64, 64, use_attention=False)
        self.hgConv2 = HypergraphConv(64, 64, use_attention=False)
        self.pos_embedding = nn.Embedding(300, self.pos_dim)

        #
        # ### channel self-gating parameters
        # self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        # self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])
        #
        # ### channel self-supervised parameters
        # self.ssl_weights = nn.ParameterList([nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        # self.ssl_bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])
        #
        # ### attention parameters
        # self.att = nn.Parameter(torch.zeros(1, self.emb_size))
        # self.att_m = nn.Parameter(torch.zeros(self.emb_size, self.emb_size))

        # graph model
        #self.GraphConv = GraphATT(self.emb_size, self.emb_size, self.layers)

        #sequence model
        # self.past_rnn = nn.GRU(input_size=self.emb_size + self.pos_dim, hidden_size=self.emb_size, batch_first= True)
        # self.future_rnn = nn.GRU(input_size=self.emb_size + self.pos_dim, hidden_size=self.emb_size, batch_first=True, bidirectional=True)

        # multi-head attention
        self.img_att = TransformerBlock(input_size=self.emb_size, n_heads=4,
                                               attn_dropout=dropout)
        self.img_to_text_att = TransformerBlock(input_size=self.emb_size, n_heads=4,
                                               attn_dropout=dropout)
        # self.future_multi_att = TransformerBlock(input_size=self.emb_size + self.pos_dim, n_heads=4,
        #                                          is_future=True, attn_dropout=dropout)

        #continue-time embedding
        # self.timeEncode = TimeEncode(time_dim=self.pos_dim)
        #
        if self.data_name == 'WEIXIN2':
            #self.node_embedding = nn.Embedding(64704, self.emb_size)
            self.linear1 = nn.Linear(256, self.emb_size)
            self.linear2 = nn.Linear(256, self.emb_size)
            self.user_embedding = nn.Embedding(20000, self.emb_size // 2)
        elif self.data_name == 'SMP':
            #self.node_embedding = nn.Embedding(305613, self.emb_size)
            self.linear1 = nn.Linear(384, self.emb_size)
            self.linear2 = nn.Linear(2048, self.emb_size)
            self.user_embedding = nn.Embedding(38312, self.emb_size//2)
        elif self.data_name == 'SIPD2020CHALLENGE':
            self.linear1 = nn.Linear(384, self.emb_size)
            self.linear2 = nn.Linear(2048, self.emb_size)
            self.user_embedding = nn.Embedding(25000, self.emb_size // 2)
        self.linear3 = nn.Linear(self.emb_size*3, self.emb_size+self.emb_size//2)
        self.linear4 = nn.Linear(self.emb_size+self.emb_size//2, 1)
        self.linear5 = nn.Linear(self.emb_size*2, self.emb_size)

        ### fusion layer
        self.fus = fusion(32, dropout)
        self.dense = torch.nn.Sequential(
            nn.Linear(self.emb_size+self.pos_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # # nn.ReLU(),
            # nn.Linear(128, 1),
        )
        self.reset_parameters()

        #### optimizer and loss function
        self.optimizerAdam = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.optimizer = ScheduledOptim(self.optimizerAdam, self.emb_size, args.n_warmup_steps)
        #self.loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=0)
        self.loss_function = nn.MSELoss()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))

    def self_supervised_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.ssl_weights[channel]) + self.ssl_bias[channel]))

    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score

    def hierarchical_ssl(self, em, adj):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        user_embeddings = em
        edge_embeddings = torch.sparse.mm(adj, em)

        # Local MIM
        pos = score(user_embeddings, edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        local_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))

        # Global MIM
        graph = torch.mean(edge_embeddings, 0)
        pos = score(edge_embeddings, graph)
        neg1 = score(row_column_shuffle(edge_embeddings), graph)
        global_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)))
        return global_loss + local_loss

    def seq2seq_ssl(self, inp_subseq_encodings: torch.Tensor, label_subseq_encodings: torch.Tensor, mask) -> torch.Tensor:

        inp_subseq_encodings = F.normalize(inp_subseq_encodings, p=2, dim=1)
        label_subseq_encodings = F.normalize(label_subseq_encodings, p=2, dim=1)

        sqrt_hidden_size = np.sqrt(self.emb_size)
        product = torch.mul(inp_subseq_encodings, label_subseq_encodings)  # [B, K, D]
        normalized_dot_product = torch.sum(product, dim=-1) / sqrt_hidden_size  # [B, K]
        numerator = torch.exp(normalized_dot_product)  # [B, K]

        inp_subseq_encodings_trans = inp_subseq_encodings.transpose(0, 1)  # [K, B, D]
        inp_subseq_encodings_trans_expanded = inp_subseq_encodings_trans.unsqueeze(1)  # [K, 1, B, D]
        label_subseq_encodings_trans = label_subseq_encodings.transpose(0, 1).transpose(1, 2)  # [K, D, B]
        dot_products = torch.matmul(inp_subseq_encodings_trans_expanded, label_subseq_encodings_trans)  # [K, K, B, B]
        dot_products = torch.exp(dot_products / sqrt_hidden_size)
        dot_products = dot_products.sum(-1)  # [K, K, B]
        temp = dot_products.sum(1)  # [K, B]
        denominator = temp.transpose(0, 1)  # [B, K]

        seq2seq_loss_k = -torch.log2(numerator / denominator)

        seq2seq_loss_k = seq2seq_loss_k.view(-1)
        seq2seq_loss_k = seq2seq_loss_k.masked_fill(mask.view(-1), 0)

        seq2seq_loss = torch.sum(seq2seq_loss_k)

        assert not torch.isnan(seq2seq_loss).any()
        return seq2seq_loss

    '''social structure and hypergraph structure embeddding'''
    def structure_embed(self, pid, related_items, HG_FEAS):

        # u_emb_c1 = self.self_gating(self.user_embedding.weight, 0)
        # u_emb_c2 = self.self_gating(self.user_embedding.weight, 1)
        # simple_user_emb = self.self_gating(self.user_embedding.weight, 2)
        #
        # all_emb_c1 = [u_emb_c1]
        # all_emb_c2 = [u_emb_c2]
        # all_emb_simple = [simple_user_emb]
        text_embedding = self.linear1(self.text_embedding[[related_items]])#batch_size*300*embedding_size
        img_embedding = self.linear2(self.img_embedding[[related_items]])
        #user_embedding = self.self_gating(self.user_embedding.weight, 0)
        #user_embedding = self.user_embedding(uid)

        text_img_embedding = torch.concat([text_embedding, img_embedding], 2)
        embedding_all = [text_img_embedding]

        # text_embedding_all = [text_embedding]
        # img_embedding_all = [img_embedding]
        # user_embedding_all = [user_embedding]
        for k in range(self.layers):
            # Channel Text
            #t_emb = torch.sparse.mm(HG_FEAS, text_embedding)#300*300 * 300*128
            all_norms = []
            for i, x in enumerate(HG_FEAS):
                t_i_emb = torch.sparse.mm(x, text_img_embedding[i])
                norm_embeddings1 = F.normalize(t_i_emb, p=2, dim=1)
                all_norms.append(norm_embeddings1)
            all_norms = torch.tensor([item.cpu().detach().numpy() for item in all_norms]).cuda()
            embedding_all += [trans_to_cuda(torch.Tensor(all_norms))]

            # Channel Img
            # i_emb = torch.sparse.mm(self.H_Text, img_embedding)#668*64
            # norm_embeddings2 = F.normalize(i_emb, p=2, dim=1)
            # img_embedding_all += [norm_embeddings2]
            #
            # # Channel User
            # u_emb = torch.sparse.mm(self.H_User, user_embedding)#668*64
            # norm_embeddings3 = F.normalize(u_emb, p=2, dim=1)
            # user_embedding_all += [norm_embeddings3]

            # user convolution
            # simple_user_emb = self.GraphConv(self.adjacency, self.user_embedding.weight, k)
            # norm_embeddings3 = F.normalize(simple_user_emb, p=2, dim=1)
            # all_emb_simple += [norm_embeddings3]

        embedding_all = torch.stack(embedding_all, dim=2)
        embedding_all = torch.sum(embedding_all, dim=2)
        # img_embedding = torch.stack(img_embedding_all, dim=1)
        # img_embedding = torch.sum(img_embedding, dim=1)
        #
        # user_embedding = torch.stack(user_embedding_all, dim=1)
        # user_embedding = torch.sum(user_embedding, dim=1)

        return embedding_all
        # aggregating channel-specific embeddings
        # high_embs, attention_score = self.channel_attention(u_emb_c1, u_emb_c2)
        # agg_Uemb = high_embs + simple_user_emb / 2
        #
        # return agg_Uemb, high_embs, simple_user_emb

    def prepare_key(self, k):
        head_size = self.emb_size//self.att_head
        key = trans_to_cuda(torch.zeros(self.att_head, k.size(0), head_size))
        for i in range(self.att_head):
            key[i] += k[:, i*head_size:(i+1)*head_size]
        return key

    def propagate(self, q, k, v):
        v_0 = v
        k = self.prepare_key(k)
        v = v.resize(v.size(0), self.att_head, self.emb_size//self.att_head)
        v = torch.transpose(v, 0, 1)
        v = torch.transpose(v, 1, 2)
        vk = torch.matmul(v, k)
        q = q.resize(q.size(0), self.att_head, self.emb_size//self.att_head)
        q = torch.transpose(q, 0, 1)
        q = torch.transpose(q, 1, 2)
        vkq = torch.matmul(vk, q)
        vkq = torch.transpose(vkq, 1, 2)
        qvkq = torch.matmul(q, vkq)
        kqvkq = torch.matmul(k, qvkq)
        kqvkq = torch.transpose(kqvkq, 0, 1)
        kqvkq = kqvkq.resize_as(v_0)
        output = kqvkq + v_0
        return output


    def forward(self, input, hg_idx, related_items, label, uid):
        # hyper_edge_feature = trans_to_cuda(torch.zeros(int(hg_idx[1].max())+1, self.emb_size))
        #
        # idx = 0
        # for i in range((hg_idx.size(1))):
        #     if hg_idx[1][i] == idx:
        #         hyper_edge_feature[idx] += self.node_embedding(related_items[hg_idx[0][i]])
        #     elif hg_idx[1][i] == idx + 1:
        #         idx += 1
        #         hyper_edge_feature[idx] += self.node_embedding(related_items[hg_idx[0][i]])
        #     else:
        #         idx += 1

        user_embedding = self.user_embedding(uid)
        #print(user_embedding)
        text_embedding = self.text_embedding[related_items]
        text_embedding = self.linear1(text_embedding)
        #text_embedding = self.linear1(self.text_embedding[related_items])
        #text_embedding = text_embedding.resize(-1, 300, self.emb_size)
        text_embedding = torch.reshape(text_embedding, (-1, 300, self.emb_size))
        img_embedding = self.linear2(self.img_embedding[related_items])
        #img_embedding = img_embedding.resize(-1, 300, self.emb_size)
        img_embedding = torch.reshape(img_embedding, (-1, 300, self.emb_size))

        img_self_att = self.img_att(img_embedding, img_embedding, img_embedding)
        k_in = img_embedding + text_embedding
        v_in = img_embedding + text_embedding
        img_to_text_att = self.img_to_text_att(text_embedding, k_in, v_in)

        #img_self_att = img_self_att.resize(-1, self.emb_size)
        img_self_att = torch.reshape(img_self_att, (-1, self.emb_size))
        #img_to_text_att = img_to_text_att.resize(-1, self.emb_size)
        img_to_text_att = torch.reshape(img_to_text_att, (-1, self.emb_size))

        img_gcn_output = self.hgConv1(img_self_att, hg_idx)
        text_gcn_output = self.hgConv2(img_to_text_att, hg_idx)

        img_gcn_output = img_gcn_output[::300, :]
        text_gcn_output = text_gcn_output[::300, :]
        text_gcn_output_t = torch.transpose(text_gcn_output, 0, 1)
        text_text = torch.matmul(text_gcn_output_t, text_gcn_output)
        img = torch.matmul(img_gcn_output, text_text)

        text_user = torch.cat([text_gcn_output, user_embedding], 1)
        img_user = torch.cat([img, user_embedding], 1)

        text_0 = text_embedding[:, 0, :]
        img_0 = img_embedding[:, 0, :]
        text_0_user = torch.cat([text_0, user_embedding], 1)
        img_0_user = torch.cat([img_0, user_embedding], 1)

        text_img_user = torch.cat([text_user, img_user], 1)
        text_img_user = self.linear3(text_img_user)#b*d
        text_img_user = text_img_user.unsqueeze(1)#b*1*d

        text_0_user = text_0_user.unsqueeze(1)
        img_0_user = img_0_user.unsqueeze(1)
        text_img_0_user = torch.cat([text_0_user, img_0_user], 1)#b*2*d

        text_user_t = text_user.unsqueeze(1)
        img_user_t = img_user.unsqueeze(1)
        text_img_user_t = torch.cat([text_user_t, img_user_t], 1)#b*2*d
        text_img_user_t = torch.transpose(text_img_user_t, 1, 2)#b*d*2

        d_d = torch.matmul(text_img_user_t, text_img_0_user)#b*d*d

        output = torch.matmul(text_img_user, d_d)#b*1*d
        output = self.linear4(torch.squeeze(output))

        return output





        ###position embedding
        # batch_t = trans_to_cuda(torch.arange(input.size(1)).expand(input.size()))
        # inp_time_embed = self.dropout(self.timeEncode(input_time))
        # lab_time_embed = self.dropout(self.timeEncode(label_time))
        # pos_embed = self.dropout(self.pos_embedding(batch_t))
        #
        # mask = (input == 0)
        # mask_label = (label == 0)
        #
        # '''structure embeddding'''
        # agg_Uemb, HG_Uemb, S_Uemb = self.structure_embed()
        #
        # '''past cascade embeddding'''
        # cas_seq_emb = F.embedding(input, S_Uemb)
        # cas_seq_emb = torch.cat([cas_seq_emb, pos_embed], dim=-1)
        # user_cas, _ = self.past_rnn(cas_seq_emb)
        # user_cas = torch.cat([user_cas, inp_time_embed], dim=-1)
        # cas_att_output = self.past_multi_att(user_cas, user_cas, user_cas, mask=mask.cuda())
        # cas_output = self.linear(cas_att_output)
        # HG_seq_emb = F.embedding(input, HG_Uemb)
        # output = self.fus(cas_output, HG_seq_emb)
        #
        # '''future cascade embeddding'''
        # future_embs = F.embedding(label, S_Uemb)
        # future_embs = torch.cat([future_embs, pos_embed], dim=-1)
        # future_u_rnn, _  = self.future_rnn(future_embs)
        # future_u_rnn1, future_u_rnn2 = torch.chunk(future_u_rnn, 2, dim=-1)
        # future_cas = future_u_rnn1 + future_u_rnn2
        # future_cas = torch.cat([future_cas, lab_time_embed], dim=-1)
        # future_att_output = self.future_multi_att(future_cas, future_cas, future_cas, mask=mask.cuda())
        # future_cas_output = self.linear(future_att_output)
        # HG_seq_la = F.embedding(label, HG_Uemb)
        # future_output = self.fus(future_cas_output, HG_seq_la)
        #
        # '''SSL loss'''
        # graph_ssl_loss = self.hierarchical_ssl(self.self_supervised_gating(agg_Uemb, 0), self.H_Source)
        # graph_ssl_loss += self.hierarchical_ssl(self.self_supervised_gating(agg_Uemb, 1), self.H_Item)
        # seq_ssl_loss = self.seq2seq_ssl(output, future_output, mask_label)
        #
        # '''Prediction'''
        # pre_y = torch.matmul(output, torch.transpose(agg_Uemb, 1, 0))
        # mask = get_previous_user_mask(input, self.n_node)
        #
        # return (pre_y + mask).view(-1, pre_y.size(-1)).cuda(), graph_ssl_loss, seq_ssl_loss

    def model_prediction(self, input, input_time):

        batch_t = trans_to_cuda(torch.arange(input.size(1)).expand(input.size()))
        pos_embed = self.dropout(self.pos_embedding(batch_t))
        inp_time_embed = self.dropout(self.timeEncode(input_time))
        mask = (input == 0)

        '''structure embeddding'''
        agg_Uemb, HG_Uemb, S_Uemb = self.structure_embed()

        '''cascade embeddding'''
        cas_seq_emb = F.embedding(input, S_Uemb)
        cas_seq_emb = torch.cat([cas_seq_emb, pos_embed], dim=-1)
        user_cas, _ = self.past_rnn(cas_seq_emb)
        user_cas = torch.cat([user_cas, inp_time_embed], dim=-1)
        cas_output = self.past_multi_att(user_cas, user_cas, user_cas, mask=mask.cuda())
        cas_output = self.linear(cas_output)
        HG_seq_emb = F.embedding(input, HG_Uemb)
        output = self.fus(cas_output, HG_seq_emb)
        pre_y = torch.matmul(output, torch.transpose(agg_Uemb, 1, 0))
        mask = get_previous_user_mask(input, self.n_node)

        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda()


