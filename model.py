import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from scipy.special import digamma
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from layers import *
class UVF(nn.Module):
    def __init__(self, config):
        super(UVF, self).__init__()


        self.dim = 256
        self.label_nums = config['num_class']
        self.view_nums = config['view_nums']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Learn_model = LearnModel(config['input_size'])
        self.multi_cls = nn.ModuleList([MultiViewClass(self.label_nums, self.dim) for _ in range(self.view_nums)])
        self.multi_cls2 = nn.ModuleList([MultiViewClass(self.label_nums, self.dim) for _ in range(self.view_nums)])
        self.multi_cls_CU = nn.ModuleList([MultiViewClass_Droupout(self.label_nums, self.dim) for _ in range(self.view_nums)])
        self.selcetor_model = MultiViewFeatureSelector(input_dim=self.dim, num_views=self.view_nums)
        self.fc_view1 = nn.Linear(self.dim, 64)



    def forward(self, x, train):
        input = self.Learn_model(x)
        os.makedirs('./pig', exist_ok=True)
        wq = []
        for i in range(self.view_nums):
            te_z = self.multi_cls[i](input[i])
            w = self.DM(te_z, self.label_nums)
            wq.append(w)
        wq = torch.stack(wq)
        output, gate_masks = self.selcetor_model(input, wq)

        CU_weights = self.uncertainty_weighted_fusion(output, self.multi_cls_CU)
        selected_features = []
        wq2 = []
        for i in range(self.view_nums):
            z = self.multi_cls2[i](output[i])
            w2 = self.DM(z, self.label_nums)
            wq2.append(w2)
            selected_features.append(z)
        wq2 = torch.stack(wq2)
        wq2_normalized = wq2 / wq2.sum()

        fused_weights = self.fused_weight(CU_weights, wq2_normalized, lambda1 = 0.8, lambda2 = 0.2)

        all = 0
        for i in range(self.view_nums):
            all += fused_weights[i] * selected_features[i]

        return all, gate_masks, input, output


    def DM(self, fm, label_num):

        softmax_outputs = F.softmax(fm, dim=1)
        mu = 1 / label_num
        abs_diff = torch.abs(softmax_outputs - mu)
        loss = torch.mean(abs_diff)
        return loss

    def Adj(self, features):
        features_norm = F.normalize(features, p=2, dim=1)  # 在特征维上做L2归一化
        adjacency_matrix = torch.mm(features_norm, features_norm.t())  # [128, 128]
        return  adjacency_matrix



    def enable_dropout(self, model):
        """ 启用 Dropout 层用于 MC Dropout 推理 """
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def compute_uncertainty(self, model, z_v, T=20):
        """
        model: 单个视图的分类器（含 Dropout）
        z_v: 当前视图的输入特征 [B, D]
        T: MC Dropout 次数
        """
        model.eval()
        self.enable_dropout(model)

        preds = []  # 存储每次预测
        with torch.no_grad():  # 推理时禁用梯度
            for _ in range(T):
                output = model(z_v)  # [B, C]
                prob = F.softmax(output, dim=-1)
                preds.append(prob.unsqueeze(0))  # [1, B, C]

        preds = torch.cat(preds, dim=0)  # [T, B, C]
        mean_pred = preds.mean(dim=0)  # [B, C]
        var_pred = preds.var(dim=0)  # [B, C]
        # print(var_pred)

        # ----- 认知不确定性（CU） -----
        CU = var_pred.sum(dim=1)  # [B]
        CU = CU.mean()

        return CU

    def uncertainty_weighted_fusion(self, z_list, model_list):
        assert len(z_list) == len(model_list)
        cu_scores = []

        for z_v, model in zip(z_list, model_list):
            CU = self.compute_uncertainty(model, z_v)
            cu_scores.append(CU.unsqueeze(0))  # [B, 1]

        cu_scores = torch.cat(cu_scores, dim=0)  # [B, V]
        weights = torch.softmax(-cu_scores, dim=0)  # 不确定性越大，权重越小

        return weights

    def fused_weight(self, CU, PU, lambda1=0.9, lambda2=0.1):
        us = lambda1 * PU + lambda2 * CU
        us = us / us.sum()
        return us

    def zhu(self, fused_weights, name):
        fused_weights = fused_weights.detach().cpu().numpy()  # 转为 numpy array

        # 创建柱状图
        views = np.arange(1, len(fused_weights) + 1)  # [1, 2, ..., V]
        plt.figure(figsize=(10, 8))
        plt.bar(views, fused_weights, color='skyblue')

        # 添加标签和标题
        plt.xlabel("View Index", fontsize=18)
        plt.ylabel("Fused Weight", fontsize=18)
        plt.title("Fused Weights per View", fontsize=18)
        plt.xticks(views, fontsize=18)  # 设置 x 轴刻度为 1 到 V

        plt.yticks(fontsize=18)  # y轴刻度数字变大

        # 显示数值标签
        for i, weight in enumerate(fused_weights):
            plt.text(views[i], weight + 0.01, f"{weight:.2f}", ha='center', va='bottom')

        plt.savefig(f'./pig/{name}.pdf', dpi=300)











