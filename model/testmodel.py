
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import DenseGCNConv, ChebConv, BatchNorm, PairNorm, GraphNorm,GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, feat):


        feat_shape = feat.size()
        batch = torch.zeros(feat_shape[1], dtype=torch.long)
        source_node = []
        target_node = []

        for node in range(feat_shape[1] - 1):
            source_node.append(node)
            target_node.append(node + 1)

        edge_index = torch.tensor((source_node, target_node), dtype=torch.long)

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


class ResGCN(nn.Module):
    def __init__(self,hidden_feats,out_feats):
        super(ResGCN, self).__init__()
        self.conv1 = ChebConv(hidden_feats[0], hidden_feats[1], 2)
        self.bn1 = GraphNorm(hidden_feats[1])
        self.conv2 = ChebConv(hidden_feats[1], hidden_feats[2], 3)
        self.bn2 = GraphNorm(hidden_feats[2])
        self.conv3 = ChebConv(hidden_feats[2], hidden_feats[1], 3)
        self.bn3 = GraphNorm(hidden_feats[1])
        self.conv4 = ChebConv(hidden_feats[1], hidden_feats[0], 2)


        self.conv5 = ChebConv(hidden_feats[0], hidden_feats[1], 2)
        self.bn5 = GraphNorm(hidden_feats[1])
        self.conv6 = ChebConv(hidden_feats[1], hidden_feats[2], 3)
        self.bn6 = GraphNorm(hidden_feats[2])
        self.conv7 = ChebConv(hidden_feats[2], hidden_feats[1], 3)
        self.bn7 = GraphNorm(hidden_feats[1])
        self.conv8 = ChebConv(hidden_feats[1], hidden_feats[0], 2)


        self.conv9 = ChebConv(hidden_feats[0], hidden_feats[1], 2)
        self.bn9 = GraphNorm(hidden_feats[1])
        self.conv10 = ChebConv(hidden_feats[1], hidden_feats[2], 3)
        self.bn10 = GraphNorm(hidden_feats[2])
        self.conv11 = ChebConv(hidden_feats[2], hidden_feats[1], 3)
        self.bn11 = GraphNorm(hidden_feats[1])
        self.conv12 = ChebConv(hidden_feats[1], hidden_feats[0], 2)


        self.conv13 = ChebConv(hidden_feats[0], hidden_feats[1], 2)
        self.bn13 = GraphNorm(hidden_feats[1])
        self.conv14 = ChebConv(hidden_feats[1], hidden_feats[2], 3)
        self.bn14 = GraphNorm(hidden_feats[2])
        self.conv15 = ChebConv(hidden_feats[2], hidden_feats[1], 3)
        self.bn15 = GraphNorm(hidden_feats[1])
        self.conv16 = ChebConv(hidden_feats[1], hidden_feats[0], 2)

        self.linear1 = torch.nn.Linear(hidden_feats[0], out_feats)
        # self.linear2 = torch.nn.Linear(18, out_feats)   #10分类



    def forward(self, x):
        feat_shape = x.size()
        batch = torch.zeros(feat_shape[1], dtype=torch.long)
        source_node = []
        target_node = []

        for node in range(feat_shape[1] - 1):
            source_node.append(node)
            target_node.append(node + 1)

        edge_index = torch.tensor((source_node, target_node), dtype=torch.long)
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, negative_slope=0.2)
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.2)
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3, negative_slope=0.2)
        x4 = self.conv4(x3, edge_index)
        x4 += x
        x4 = F.relu(x4)


        x5 = self.conv5(x4, edge_index)
        x5 = self.bn5(x5)
        x5 = F.leaky_relu(x5, negative_slope=0.2)
        x6 = self.conv6(x5, edge_index)
        x6 = self.bn6(x6)
        x6 = F.leaky_relu(x6, negative_slope=0.2)
        x7 = self.conv7(x6, edge_index)
        x7 = self.bn7(x7)
        x7 = F.leaky_relu(x7, negative_slope=0.2)
        x8 = self.conv8(x7, edge_index)
        x8 += x
        x8 = F.relu(x8)


        x9 = self.conv9(x8, edge_index)
        x9 = self.bn9(x9)
        x9 = F.leaky_relu(x9, negative_slope=0.2)
        x10 = self.conv10(x9, edge_index)
        x10 = self.bn10(x10)
        x10 = F.leaky_relu(x10, negative_slope=0.2)
        x11 = self.conv11(x10, edge_index)
        x11 = self.bn11(x11)
        x11 = F.leaky_relu(x11, negative_slope=0.2)
        x12 = self.conv12(x11, edge_index)
        x12 += x
        x12 = F.relu(x12)


        x13 = self.conv13(x12, edge_index)
        x13 = self.bn13(x13)
        x13 = F.leaky_relu(x13, negative_slope=0.2)
        x14 = self.conv14(x13, edge_index)
        x14 = self.bn14(x14)
        x14 = F.leaky_relu(x14, negative_slope=0.2)
        x15 = self.conv15(x14, edge_index)
        x15 = self.bn15(x15)
        x15 = F.leaky_relu(x15, negative_slope=0.2)
        x16 = self.conv16(x15, edge_index)
        x16 += x
        x16 = F.relu(x16)

        out = pyg_nn.global_mean_pool(x16, batch)  # 平均池化
        out = self.linear1(out)
        out = F.tanh(out)
        # out = self.linear2(out)

        #out = F.tanh(out)
        #out = self.linear3(out)
        #out = F.tanh(out)
        #out = self.linear4(out)
        return out

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

    def forward(self, feat):


        feat_shape = feat.size()
        batch = torch.zeros(feat_shape[1], dtype=torch.long)
        source_node = []
        target_node = []

        for node in range(feat_shape[1] - 1):
            source_node.append(node)
            target_node.append(node + 1)

        edge_index = torch.tensor((source_node, target_node), dtype=torch.long)

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


class CNNModel(nn.Module):
    def __init__(self, hidden_feats, out_feats):
        super(CNNModel, self).__init__()
        self.num_classes = out_feats
        self.convs1 = nn.ModuleList()
        self.bns1 = nn.ModuleList()

        # 使用1D卷积层代替图卷积层
        for i in range(len(hidden_feats) - 1):
            self.convs1.append(nn.Conv1d(hidden_feats[i], hidden_feats[i + 1], kernel_size=3, padding=1))
            self.bns1.append(nn.BatchNorm1d(hidden_feats[i + 1]))

        # 添加最后一层
        self.convs1.append(nn.Conv1d(hidden_feats[-1], hidden_feats[0], kernel_size=3, padding=1))
        self.bns1.append(nn.BatchNorm1d(hidden_feats[0]))

        # 全连接层
        self.linear1 = nn.Linear(hidden_feats[0], out_feats)

    def forward(self, feat):
        # 假设输入的feat形状为 (batch_size, num_features, sequence_length)
        batch_size, num_features, sequence_length = feat.size()

        feat1 = feat.clone()
        for conv, bn in zip(self.convs1[:-1], self.bns1[:-1]):
            feat1 = conv(feat1)
            feat1 = bn(feat1)
            feat1 = F.leaky_relu(feat1, negative_slope=0.1)

        feat1 = self.convs1[-1](feat1)
        feat1 = feat1 + self.bns1[-1](feat1)
        feat1 = F.relu(feat1)

        # 使用全局平均池化
        feat1 = feat1.mean(dim=2)  # (batch_size, hidden_feats[0])

        out1 = self.linear1(feat1)
        out1 = F.softplus(out1)
        return out1




if __name__ == '__main__':


    from calflops import calculate_flops

    # ChebGCN11 = GCN1([45, 90, 180, 90],12)
    # ChebGCN12 = GCN1([68, 136, 272, 136], 12)
    #  ChebGCN2 = GCN1([3, 6, 24, 18],6)
    #
    # GCN = GCN1([9, 36, 72, 36], 6)
    # GCN = GCN1([113, 72, 144, 72], 12)
    GCN = GCN1([21, 42, 84, 42], 12)

    # flops1, macs, params1 = calculate_flops(model=ChebGCN11,
    #                                       input_shape=(1, 30, 45))
    # flops2, macs, params2 = calculate_flops(model=ChebGCN12,
    #                                       input_shape=(1, 30, 68))

    # print("OPPORTUNITY FLOPs:%s %s   Params:%s %s\n" % (flops1,flops2, params1,params2))


    # resgcn = ResGCN([9,18,36],6)
    # flops1, macs, params1 = calculate_flops(model=resgcn,
    #                                       input_shape=(1, 128, 9))
    # print("UCI FLOPs:%s    Params: %s\n" % (flops1, params1))

    # resgcn = CNNModel([128, 256, 128, 64],17)
    flops1, macs, params1 = calculate_flops(model=GCN,
                                            input_shape=(1, 128, 21))
    print("MHEALTH Params:%s    FLOPs: %s\n" % (params1,flops1 ))



# PAMAP2 FLOPs:8.89 MFLOPS 2.23 MFLOPS 2.23 MFLOPS    Params:26.8 K 6.92 K 6.92 K          13.35M     40.64K
# MHEALTH FLOPs:3.5 MFLOPS 1.56 MFLOPS 1.56 MFLOPS   Params:14.2 K 6.44 K 6.44 K           6.62M      27.08K
# OPPORTUNITY FLOPs:9.73 MFLOPS 22.22 MFLOPS   Params:164.17 K 373.2 K                     31.59M     537.37K
# UCI FLOPs:2.33 MFLOPS 670.12 KFLOPS    Params:9.5 K 2.82 K                               3.0M       12.32K

# FLOPs:1.17 MFLOPS338.34 KFLOPS     Params:4.96 K1.52 K


# ResGCN
# PAMAP2 FLOPs:99.51 MFLOPS    Params: 295.95 K
# MHEALTH FLOPs:25.39 MFLOPS    Params: 101.82 K
# OPPORTUNITY FLOPs:171.74 MFLOPS    Params: 2.88 M
# UCI FLOPs:4.69 MFLOPS    Params: 19.39 K


# CNNModel
# UCI FLOPs:4.45 MFLOPS    Params: 248.26 K
# PAMAP2 FLOPs:20.76 MFLOPS    Params: 290.96 K
# OPPORTUNITY FLOPs:38.77 MFLOPS    Params: 166.7 K
#MHEALTH FLOPs:10.37 MFLOPS    Params: 249.68 K



#GCN
# PAMAP2 Params:53.58 K    FLOPs: 13.31 MFLOPS
# OPPORTUNITY Params:196.63 K    FLOPs: 49.29 MFLOPS
# MHEALTH Params:18.66 K    FLOPs: 4.54 MFLOPS
# UCI Params: 12.34 K FLOPs:3.01 MFLOPS