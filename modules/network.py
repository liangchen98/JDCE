import torch.nn as nn
import torch

#  This file is modified according to CC(https://github.com/XLearning-SCU/2021-AAAI-CC)


class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num, drop_rate=0.2):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            # nn.Softmax(dim=1)
        )

    def forward_cluster(self, x):
        h = self.resnet(x)
        f = self.cluster_projector(h)
        c = torch.argmax(f, dim=1)
        return f, c
