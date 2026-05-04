import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnModel(nn.Module):
    def __init__(self, input_size):
        super(LearnModel, self).__init__()
        self.input_sizes = input_size
        self.weidu = 128
        self.device = torch.device('cuda')
        self.fc_view1 = nn.Linear(input_size[0], self.weidu)
        self.fc_view2 = nn.Linear(input_size[1], self.weidu)
        self.fc_view3 = nn.Linear(input_size[2], self.weidu)
        self.fc_view4 = nn.Linear(input_size[3], self.weidu)
        self.fc_view5 = nn.Linear(input_size[4], self.weidu)
        # self.fc_view6 = nn.Linear(input_size[5], self.next)
        # self.fc_view7 = nn.Linear(input_size[6], self.next)
        # self.fc = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(dim, self.weidu)
        #     )
        #     for dim in input_size
        # ])

    def forward(self, x):

        x1 = self.fc_view1(x["view1"])
        x2 = self.fc_view2(x["view2"])
        x3 = self.fc_view3(x["view3"])
        x4 = self.fc_view4(x["view4"])
        x5 = self.fc_view5(x["view5"])
        # x6 = self.fc_view6(x["view6"])
        # x7 = self.fc_view7(x["view7"])
        tmp = [x1, x2, x3, x4, x5]
        # tmp = [x1, x2, x3, x4, x5, x6, x7]
        # outs = [fc_demo(view) for fc_demo, view in zip(self.fc, x.values())]
        return tmp


class DifferentiatedFeatureSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(DifferentiatedFeatureSelector, self).__init__()
        self.device = torch.device('cuda')

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [batch_size, input_dim]
        strength: [batch_size, 1]
        """
        gate = self.gate_net(x)
        x_selected = gate * x
        return x_selected, gate


class MultiViewFeatureSelector(nn.Module):
    def __init__(self, input_dim=128, num_views=5):
        super(MultiViewFeatureSelector, self).__init__()
        self.device = torch.device('cuda')
        self.num_views = num_views
        self.selectors = nn.ModuleList([
            DifferentiatedFeatureSelector(input_dim=input_dim) for _ in range(num_views)
        ])
        self.fusion_layer = nn.Linear(input_dim * num_views, input_dim)  # 可根据任务调整

    def forward(self, views, strengths):
        """
        views: list of tensors, each [batch_size, input_dim]
        strengths: list of tensors, each [batch_size, 1]
        """
        selected_features = []
        gate_masks = []
        sorted_indices = torch.argsort(strengths)
        tmp = strengths[sorted_indices[4]]
        # tmp = 0
        for i in range(self.num_views):
            if strengths[i] <= tmp:
                z, g = self.selectors[i](views[i])
                selected_features.append(z)
                gate_masks.append(g)
            else:
                selected_features.append(views[i])
                ones_mask = torch.ones_like(views[i])
                gate_masks.append(ones_mask)

        return selected_features, gate_masks

class MultiViewClass(nn.Module):
    def __init__(self, label_nums, input_dim=128):
        super(MultiViewClass, self).__init__()
        self.cls = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, label_nums)
        )

    def forward(self, x):
        out = self.cls(x)
        return out


class MultiViewClass_Droupout(nn.Module):
    def __init__(self, label_nums, input_dim=128):
        super(MultiViewClass_Droupout, self).__init__()
        self.cls = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # 加入Dropout层，p是丢弃概率
            nn.Linear(input_dim, label_nums),
        )

    def forward(self, x):
        out = self.cls(x)
        return out





