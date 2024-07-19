
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import DenseGCNConv, ChebConv, BatchNorm, PairNorm, GraphNorm,GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

class GCN1(nn.Module):
    def __init__(self, hidden_feats, out_feats):
        super(GCN1, self).__init__()
        self.num_classes = out_feats
        self.convs1 = nn.ModuleList()
        self.bns1 = nn.ModuleList()
        for i in range(len(hidden_feats) - 1):
            self.convs1.append(GraphConv(hidden_feats[i], hidden_feats[i + 1]))
            self.bns1.append(GraphNorm(hidden_feats[i + 1]))
        # Adding the last layer
        self.convs1.append(GraphConv(hidden_feats[-1], hidden_feats[0]))
        self.bns1.append(GraphNorm(hidden_feats[0]))
        self.linear1 = nn.Linear(hidden_feats[0], out_feats)
        
    def forward(self, edge_index, feat):
        feat_shape = feat.size()
        batch = torch.zeros(feat_shape[1], dtype=torch.long).to('cuda:0')

        feat1 = feat.clone()
        for conv, bn in zip(self.convs1[0:3], self.bns1[0:3]):
            feat1 = conv(feat1,edge_index)
            feat1 = bn(feat1)
            feat1 = F.leaky_relu(feat1, negative_slope=0.1)

        feat1 = self.convs1[-1](feat1,edge_index)
        feat1 = feat + self.bns1[-1](feat1)
        feat1 = F.relu(feat1)
        feat1 = pyg_nn.global_mean_pool(feat1, batch)
        feat1 = torch.squeeze(feat1)
        out1 = self.linear1(feat1)
        e1 = F.relu(out1)   

        return e1

class ChebGCN1(nn.Module):
    def __init__(self, hidden_feats, out_feats):
        super(ChebGCN1, self).__init__()
        self.num_classes = out_feats
        self.convs1 = nn.ModuleList()
        self.bns1 = nn.ModuleList()
        for i in range(len(hidden_feats) - 1):
            self.convs1.append(ChebConv(hidden_feats[i], hidden_feats[i + 1],4))
            self.bns1.append(GraphNorm(hidden_feats[i + 1]))
        # Adding the last layer
        self.convs1.append(ChebConv(hidden_feats[-1], hidden_feats[0],4))
        self.bns1.append(GraphNorm(hidden_feats[0]))
        self.linear1 = nn.Linear(hidden_feats[0], out_feats)

    def forward(self, edge_index, feat):
        feat_shape = feat.size()
        batch = torch.zeros(feat_shape[1], dtype=torch.long).to('cuda:0')

        feat1 = feat.clone()
        for conv, bn in zip(self.convs1[0:3], self.bns1[0:3]):
            feat1 = conv(feat1,edge_index)
            feat1 = bn(feat1)
            feat1 = F.leaky_relu(feat1, negative_slope=0.1)

        feat1 = self.convs1[-1](feat1,edge_index)
        feat1 = feat + self.bns1[-1](feat1)
        feat1 = F.relu(feat1)
        feat1 = pyg_nn.global_mean_pool(feat1, batch)
        feat1 = torch.squeeze(feat1)
        out1 = self.linear1(feat1)
        out1 = F.softplus(out1)
        return out1



class ChebGCN2Multi_softmax(nn.Module):
    def __init__(self, hidden_feats, hidden_feats2, out_feats):
        super(ChebGCN2Multi_softmax, self).__init__()
        self.num_classes = out_feats

        self.convs1 = nn.ModuleList()
        self.bns1 = nn.ModuleList()

        for i in range(len(hidden_feats) - 1):
            self.convs1.append(ChebConv(hidden_feats[i], hidden_feats[i + 1], 4))
            self.bns1.append(GraphNorm(hidden_feats[i + 1]))

        # Adding the last layer
        self.convs1.append(ChebConv(hidden_feats[-1], hidden_feats[0], 4))
        self.bns1.append(GraphNorm(hidden_feats[0]))
        self.linear1 = nn.Linear(hidden_feats[0], out_feats)

        self.convs2 = nn.ModuleList()
        self.bns2 = nn.ModuleList()

        for i in range(len(hidden_feats2) - 1):
            self.convs2.append(ChebConv(hidden_feats2[i], hidden_feats2[i + 1], 4))
            self.bns2.append(GraphNorm(hidden_feats2[i + 1]))

        # Adding the last layer
        self.convs2.append(ChebConv(hidden_feats2[-1], hidden_feats2[0], 4))
        self.bns2.append(GraphNorm(hidden_feats2[0]))
        self.linear2 = nn.Linear(hidden_feats2[0], out_feats)

    def forward(self, edge_index, feat, feat_):
        feat_shape = feat.size()
        batch = torch.zeros(feat_shape[1], dtype=torch.long).to('cuda:0')

        feat1 = feat.clone()
        for conv, bn in zip(self.convs1[0:3], self.bns1[0:3]):
            feat1 = conv(feat1,edge_index)
            feat1 = bn(feat1)
            feat1 = F.leaky_relu(feat1, negative_slope=0.1)

        feat1 = self.convs1[-1](feat1,edge_index)
        feat1 = feat + self.bns1[-1](feat1)
        feat1 = F.relu(feat1)

        # 通过平均池化每个节点的表示得到图表示
        feat1 = pyg_nn.global_mean_pool(feat1, batch)
        feat1 = torch.squeeze(feat1)
        feat1 = F.relu(feat1)
        out1 = self.linear1(feat1)
        out1 = F.softmax(out1)
 
        feat2 = feat_.clone()
        for conv, bn in zip(self.convs2[0:3], self.bns2[0:3]):
            feat2 = conv(feat2,edge_index)
            feat2 = bn(feat2)
            feat2 = F.leaky_relu(feat2, negative_slope=0.1)

        feat2 = self.convs2[-1](feat2,edge_index)
        feat2 = feat_ + self.bns2[-1](feat2)
        feat2 = F.relu(feat2)
        # 通过平均池化每个节点的表示得到图表示
        feat2 = pyg_nn.global_mean_pool(feat2, batch)
        feat2 = torch.squeeze(feat2)
        feat2 = F.relu(feat2)
        out2 = self.linear2(feat2)
        out2 = F.softmax(out2)
  
        
        return out1,out2


class ChebGCN2Multi_relu(nn.Module):
    def __init__(self, hidden_feats, hidden_feats2, out_feats):
        super(ChebGCN2Multi_relu, self).__init__()
        self.num_classes = out_feats

        self.convs1 = nn.ModuleList()
        self.bns1 = nn.ModuleList()

        for i in range(len(hidden_feats) - 1):
            self.convs1.append(ChebConv(hidden_feats[i], hidden_feats[i + 1], 4))
            self.bns1.append(GraphNorm(hidden_feats[i + 1]))

        # Adding the last layer
        self.convs1.append(ChebConv(hidden_feats[-1], hidden_feats[0], 4))
        self.bns1.append(GraphNorm(hidden_feats[0]))
        self.linear1 = nn.Linear(hidden_feats[0], out_feats)

        self.convs2 = nn.ModuleList()
        self.bns2 = nn.ModuleList()

        for i in range(len(hidden_feats2) - 1):
            self.convs2.append(ChebConv(hidden_feats2[i], hidden_feats2[i + 1], 4))
            self.bns2.append(GraphNorm(hidden_feats2[i + 1]))

        # Adding the last layer
        self.convs2.append(ChebConv(hidden_feats2[-1], hidden_feats2[0], 4))
        self.bns2.append(GraphNorm(hidden_feats2[0]))
        self.linear2 = nn.Linear(hidden_feats2[0], out_feats)

    def forward(self, edge_index, feat, feat_):
        feat_shape = feat.size()
        batch = torch.zeros(feat_shape[1], dtype=torch.long).to('cuda:0')

        feat1 = feat.clone()
        for conv, bn in zip(self.convs1[0:3], self.bns1[0:3]):
            feat1 = conv(feat1, edge_index)
            feat1 = bn(feat1)
            feat1 = F.leaky_relu(feat1, negative_slope=0.1)

        feat1 = self.convs1[-1](feat1, edge_index)
        feat1 = feat + self.bns1[-1](feat1)
        feat1 = F.relu(feat1)

        # 通过平均池化每个节点的表示得到图表示
        feat1 = pyg_nn.global_mean_pool(feat1, batch)
        feat1 = torch.squeeze(feat1)
        feat1 = F.relu(feat1)
        out1 = self.linear1(feat1)
        out1 = F.relu(out1)

        feat2 = feat_.clone()
        for conv, bn in zip(self.convs2[0:3], self.bns2[0:3]):
            feat2 = conv(feat2, edge_index)
            feat2 = bn(feat2)
            feat2 = F.leaky_relu(feat2, negative_slope=0.1)

        feat2 = self.convs2[-1](feat2, edge_index)
        feat2 = feat_ + self.bns2[-1](feat2)
        feat2 = F.relu(feat2)
        # 通过平均池化每个节点的表示得到图表示
        feat2 = pyg_nn.global_mean_pool(feat2, batch)
        feat2 = torch.squeeze(feat2)
        feat2 = F.relu(feat2)
        out2 = self.linear2(feat2)
        out2 = F.relu(out2)

        return out1, out2


class ChebGCN2Multi_fusion(nn.Module):
    def __init__(self, hidden_feats, hidden_feats2, out_feats):
        super(ChebGCN2Multi_fusion, self).__init__()
        self.num_classes = out_feats

        self.convs1 = nn.ModuleList()
        self.bns1 = nn.ModuleList()

        for i in range(len(hidden_feats) - 1):
            self.convs1.append(ChebConv(hidden_feats[i], hidden_feats[i + 1], 4))
            self.bns1.append(GraphNorm(hidden_feats[i + 1]))

        # Adding the last layer
        self.convs1.append(ChebConv(hidden_feats[-1], hidden_feats[0], 4))
        self.bns1.append(GraphNorm(hidden_feats[0]))
        self.linear1 = nn.Linear(hidden_feats[0], out_feats)

        self.convs2 = nn.ModuleList()
        self.bns2 = nn.ModuleList()

        for i in range(len(hidden_feats2) - 1):
            self.convs2.append(ChebConv(hidden_feats2[i], hidden_feats2[i + 1], 4))
            self.bns2.append(GraphNorm(hidden_feats2[i + 1]))

        # Adding the last layer
        self.convs2.append(ChebConv(hidden_feats2[-1], hidden_feats2[0], 4))
        self.bns2.append(GraphNorm(hidden_feats2[0]))
        self.linear2 = nn.Linear(hidden_feats2[0], out_feats)

    def forward(self, edge_index, feat, feat_):
        feat_shape = feat.size()
        batch = torch.zeros(feat_shape[1], dtype=torch.long).to('cuda:0')

        feat1 = feat.clone()
        for conv, bn in zip(self.convs1[0:3], self.bns1[0:3]):
            feat1 = conv(feat1, edge_index)
            feat1 = bn(feat1)
            feat1 = F.leaky_relu(feat1, negative_slope=0.1)

        feat1 = self.convs1[-1](feat1, edge_index)
        feat1 = feat + self.bns1[-1](feat1)
        feat1 = F.relu(feat1)

        # 通过平均池化每个节点的表示得到图表示
        feat1 = pyg_nn.global_mean_pool(feat1, batch)
        feat1 = torch.squeeze(feat1)
        feat1 = F.relu(feat1)
        out1 = self.linear1(feat1)
        out11 = F.softmax(out1)
        out12 = F.relu(out1)
        out1 = out11*out12

        feat2 = feat_.clone()
        for conv, bn in zip(self.convs2[0:3], self.bns2[0:3]):
            feat2 = conv(feat2, edge_index)
            feat2 = bn(feat2)
            feat2 = F.leaky_relu(feat2, negative_slope=0.1)

        feat2 = self.convs2[-1](feat2, edge_index)
        feat2 = feat_ + self.bns2[-1](feat2)
        feat2 = F.relu(feat2)
        # 通过平均池化每个节点的表示得到图表示
        feat2 = pyg_nn.global_mean_pool(feat2, batch)
        feat2 = torch.squeeze(feat2)
        feat2 = F.relu(feat2)
        out2 = self.linear2(feat2)
        out21 = F.softmax(out2)
        out22 = F.relu(out2)
        out2 = out21 * out22
        return out1, out2




class ChebGCN3Multi_softmax(nn.Module):
    def __init__(self, hidden_feats, hidden_feats2, hidden_feats3, out_feats):
        super(ChebGCN3Multi_softmax, self).__init__()
        self.num_classes = out_feats

        self.convs1 = nn.ModuleList()
        self.bns1 = nn.ModuleList()

        for i in range(len(hidden_feats) - 1):
            self.convs1.append(ChebConv(hidden_feats[i], hidden_feats[i + 1], 4))
            self.bns1.append(GraphNorm(hidden_feats[i + 1]))

        # Adding the last layer
        self.convs1.append(ChebConv(hidden_feats[-1], hidden_feats[0], 4))
        self.bns1.append(GraphNorm(hidden_feats[0]))
        self.linear1 = nn.Linear(hidden_feats[0], out_feats)

        self.convs2 = nn.ModuleList()
        self.bns2 = nn.ModuleList()

        for i in range(len(hidden_feats2) - 1):
            self.convs2.append(ChebConv(hidden_feats2[i], hidden_feats2[i + 1], 4))
            self.bns2.append(GraphNorm(hidden_feats2[i + 1]))

        # Adding the last layer
        self.convs2.append(ChebConv(hidden_feats2[-1], hidden_feats2[0], 4))
        self.bns2.append(GraphNorm(hidden_feats2[0]))
        self.linear2 = nn.Linear(hidden_feats2[0], out_feats)

        self.convs3 = nn.ModuleList()
        self.bns3 = nn.ModuleList()

        for i in range(len(hidden_feats3) - 1):
            self.convs3.append(ChebConv(hidden_feats3[i], hidden_feats3[i + 1], 4))
            self.bns3.append(GraphNorm(hidden_feats3[i + 1]))

        # Adding the last layer
        self.convs3.append(ChebConv(hidden_feats3[-1], hidden_feats3[0], 4))
        self.bns3.append(GraphNorm(hidden_feats3[0]))

        self.linear3 = nn.Linear(hidden_feats3[0], out_feats)


    def forward(self, edge_index, feat, feat_1, feat_2):

        feat_shape = feat.size()
        batch = torch.zeros(feat_shape[1], dtype=torch.long).to('cuda:0')

        feat1 = feat.clone()
        for conv, bn in zip(self.convs1[0:3], self.bns1[0:3]):
            feat1 = conv(feat1,edge_index)
            feat1 = bn(feat1)
            feat1 = F.leaky_relu(feat1, negative_slope=0.1)

        feat1 = self.convs1[-1](feat1,edge_index)
        feat1 = feat + self.bns1[-1](feat1)
        feat1 = F.relu(feat1)
        feat1 = pyg_nn.global_mean_pool(feat1, batch)
        feat1 = torch.squeeze(feat1)
        feat1 = F.relu(feat1)
        out1 = self.linear1(feat1)
        out1 = F.softmax(out1)
  
        


        feat2 = feat_1.clone()
        for conv, bn in zip(self.convs2[0:3], self.bns2[0:3]):
            feat2 = conv(feat2,edge_index)
            feat2 = bn(feat2)
            feat2 = F.leaky_relu(feat2, negative_slope=0.1)

        feat2 = self.convs2[-1](feat2,edge_index)
        feat2 = feat_1 + self.bns2[-1](feat2)
        feat2 = F.relu(feat2)

        feat2 = pyg_nn.global_mean_pool(feat2, batch)
        feat2 = torch.squeeze(feat2)
        feat2 = F.relu(feat2)
        out2 = self.linear2(feat2)
        out2 = F.softmax(out2)


        feat3 = feat_2.clone()
        for conv, bn in zip(self.convs3[0:3], self.bns3[0:3]):
            feat3 = conv(feat3,edge_index)
            feat3 = bn(feat3)
            feat3 = F.leaky_relu(feat3, negative_slope=0.1)
        feat3 = self.convs3[-1](feat3,edge_index)
        feat3 = feat_2 + self.bns3[-1](feat3)
        feat3 = F.relu(feat3)
        feat3 = pyg_nn.global_mean_pool(feat3, batch)
        feat3 = torch.squeeze(feat3)
        feat3 = F.relu(feat3)
        out3 = self.linear3(feat3)
        out3 = F.softmax(out3)
   
    
        return out1,out2,out3


class ChebGCN3Multi_relu(nn.Module):
    def __init__(self, hidden_feats, hidden_feats2, hidden_feats3, out_feats):
        super(ChebGCN3Multi_relu, self).__init__()
        self.num_classes = out_feats

        self.convs1 = nn.ModuleList()
        self.bns1 = nn.ModuleList()

        for i in range(len(hidden_feats) - 1):
            self.convs1.append(ChebConv(hidden_feats[i], hidden_feats[i + 1], 4))
            self.bns1.append(GraphNorm(hidden_feats[i + 1]))

        # Adding the last layer
        self.convs1.append(ChebConv(hidden_feats[-1], hidden_feats[0], 4))
        self.bns1.append(GraphNorm(hidden_feats[0]))
        self.linear1 = nn.Linear(hidden_feats[0], out_feats)

        self.convs2 = nn.ModuleList()
        self.bns2 = nn.ModuleList()

        for i in range(len(hidden_feats2) - 1):
            self.convs2.append(ChebConv(hidden_feats2[i], hidden_feats2[i + 1], 4))
            self.bns2.append(GraphNorm(hidden_feats2[i + 1]))

        # Adding the last layer
        self.convs2.append(ChebConv(hidden_feats2[-1], hidden_feats2[0], 4))
        self.bns2.append(GraphNorm(hidden_feats2[0]))
        self.linear2 = nn.Linear(hidden_feats2[0], out_feats)

        self.convs3 = nn.ModuleList()
        self.bns3 = nn.ModuleList()

        for i in range(len(hidden_feats3) - 1):
            self.convs3.append(ChebConv(hidden_feats3[i], hidden_feats3[i + 1], 4))
            self.bns3.append(GraphNorm(hidden_feats3[i + 1]))

        # Adding the last layer
        self.convs3.append(ChebConv(hidden_feats3[-1], hidden_feats3[0], 4))
        self.bns3.append(GraphNorm(hidden_feats3[0]))

        self.linear3 = nn.Linear(hidden_feats3[0], out_feats)

    def forward(self, edge_index, feat, feat_1, feat_2):

        feat_shape = feat.size()
        batch = torch.zeros(feat_shape[1], dtype=torch.long).to('cuda:0')

        feat1 = feat.clone()
        for conv, bn in zip(self.convs1[0:3], self.bns1[0:3]):
            feat1 = conv(feat1, edge_index)
            feat1 = bn(feat1)
            feat1 = F.leaky_relu(feat1, negative_slope=0.1)

        feat1 = self.convs1[-1](feat1, edge_index)
        feat1 = feat + self.bns1[-1](feat1)
        feat1 = F.relu(feat1)
        feat1 = pyg_nn.global_mean_pool(feat1, batch)
        feat1 = torch.squeeze(feat1)
        feat1 = F.relu(feat1)
        out1 = self.linear1(feat1)
        out1 = F.relu(out1)

        feat2 = feat_1.clone()
        for conv, bn in zip(self.convs2[0:3], self.bns2[0:3]):
            feat2 = conv(feat2, edge_index)
            feat2 = bn(feat2)
            feat2 = F.leaky_relu(feat2, negative_slope=0.1)

        feat2 = self.convs2[-1](feat2, edge_index)
        feat2 = feat_1 + self.bns2[-1](feat2)
        feat2 = F.relu(feat2)

        feat2 = pyg_nn.global_mean_pool(feat2, batch)
        feat2 = torch.squeeze(feat2)
        feat2 = F.relu(feat2)
        out2 = self.linear2(feat2)
        out2 = F.relu(out2)

        feat3 = feat_2.clone()
        for conv, bn in zip(self.convs3[0:3], self.bns3[0:3]):
            feat3 = conv(feat3, edge_index)
            feat3 = bn(feat3)
            feat3 = F.leaky_relu(feat3, negative_slope=0.1)
        feat3 = self.convs3[-1](feat3, edge_index)
        feat3 = feat_2 + self.bns3[-1](feat3)
        feat3 = F.relu(feat3)
        feat3 = pyg_nn.global_mean_pool(feat3, batch)
        feat3 = torch.squeeze(feat3)
        feat3 = F.relu(feat3)
        out3 = self.linear3(feat3)
        out3 = F.relu(out3)

        return out1, out2, out3


class ChebGCN3Multi_fusion(nn.Module):
    def __init__(self, hidden_feats, hidden_feats2, hidden_feats3, out_feats):
        super(ChebGCN3Multi_fusion, self).__init__()
        self.num_classes = out_feats

        self.convs1 = nn.ModuleList()
        self.bns1 = nn.ModuleList()

        for i in range(len(hidden_feats) - 1):
            self.convs1.append(ChebConv(hidden_feats[i], hidden_feats[i + 1], 4))
            self.bns1.append(GraphNorm(hidden_feats[i + 1]))

        # Adding the last layer
        self.convs1.append(ChebConv(hidden_feats[-1], hidden_feats[0], 4))
        self.bns1.append(GraphNorm(hidden_feats[0]))
        self.linear1 = nn.Linear(hidden_feats[0], out_feats)

        self.convs2 = nn.ModuleList()
        self.bns2 = nn.ModuleList()

        for i in range(len(hidden_feats2) - 1):
            self.convs2.append(ChebConv(hidden_feats2[i], hidden_feats2[i + 1], 4))
            self.bns2.append(GraphNorm(hidden_feats2[i + 1]))

        # Adding the last layer
        self.convs2.append(ChebConv(hidden_feats2[-1], hidden_feats2[0], 4))
        self.bns2.append(GraphNorm(hidden_feats2[0]))
        self.linear2 = nn.Linear(hidden_feats2[0], out_feats)

        self.convs3 = nn.ModuleList()
        self.bns3 = nn.ModuleList()

        for i in range(len(hidden_feats3) - 1):
            self.convs3.append(ChebConv(hidden_feats3[i], hidden_feats3[i + 1], 4))
            self.bns3.append(GraphNorm(hidden_feats3[i + 1]))

        # Adding the last layer
        self.convs3.append(ChebConv(hidden_feats3[-1], hidden_feats3[0], 4))
        self.bns3.append(GraphNorm(hidden_feats3[0]))

        self.linear3 = nn.Linear(hidden_feats3[0], out_feats)

    def forward(self, edge_index, feat, feat_1, feat_2):

        feat_shape = feat.size()
        batch = torch.zeros(feat_shape[1], dtype=torch.long).to('cuda:0')

        feat1 = feat.clone()
        for conv, bn in zip(self.convs1[0:3], self.bns1[0:3]):
            feat1 = conv(feat1, edge_index)
            feat1 = bn(feat1)
            feat1 = F.leaky_relu(feat1, negative_slope=0.1)

        feat1 = self.convs1[-1](feat1, edge_index)
        feat1 = feat + self.bns1[-1](feat1)
        feat1 = F.relu(feat1)
        feat1 = pyg_nn.global_mean_pool(feat1, batch)
        feat1 = torch.squeeze(feat1)
        feat1 = F.relu(feat1)
        out1 = self.linear1(feat1)
        out1 = F.softplus(out1)

        out11 = F.softmax(out1)
        out12 = F.relu(out1)
        out1 = out11 * out12

        feat2 = feat_1.clone()
        for conv, bn in zip(self.convs2[0:3], self.bns2[0:3]):
            feat2 = conv(feat2, edge_index)
            feat2 = bn(feat2)
            feat2 = F.leaky_relu(feat2, negative_slope=0.1)

        feat2 = self.convs2[-1](feat2, edge_index)
        feat2 = feat_1 + self.bns2[-1](feat2)
        feat2 = F.relu(feat2)

        feat2 = pyg_nn.global_mean_pool(feat2, batch)
        feat2 = torch.squeeze(feat2)
        feat2 = F.relu(feat2)
        out2 = self.linear2(feat2)

        out21 = F.softmax(out2)
        out22 = F.relu(out2)
        out2 = out21 * out22


        feat3 = feat_2.clone()
        for conv, bn in zip(self.convs3[0:3], self.bns3[0:3]):
            feat3 = conv(feat3, edge_index)
            feat3 = bn(feat3)
            feat3 = F.leaky_relu(feat3, negative_slope=0.1)
        feat3 = self.convs3[-1](feat3, edge_index)
        feat3 = feat_2 + self.bns3[-1](feat3)
        feat3 = F.relu(feat3)
        feat3 = pyg_nn.global_mean_pool(feat3, batch)
        feat3 = torch.squeeze(feat3)
        feat3 = F.relu(feat3)
        out3 = self.linear3(feat3)
        out31 = F.softmax(out3)
        out32 = F.relu(out3)
        out3 = out31*out32

        return out1, out2, out3

