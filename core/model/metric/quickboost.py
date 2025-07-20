import torch
import torch.nn as nn
import torch.nn.functional as F
from .metric_model import MetricModel

class QuickBoostLayer(nn.Module):
    """核心特征交互层，整合RelationNet和局部特征匹配"""
    def __init__(self, input_size, hidden_size, n_k=3):
        super(QuickBoostLayer, self).__init__()
        self.n_k = n_k
        # RelationNet组件
        self.relation_net = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(input_size*3*3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, query_feat, support_feat, way_num, shot_num, query_num):
        """
        Args:
            query_feat: [episode, way*query, c, h, w]
            support_feat: [episode, way*shot, c, h, w]
        Returns:
            combined_score: [episode*way*query, way]
        """
        # 1. RelationNet分支
        relation_pairs = torch.cat([
            support_feat.unsqueeze(1).expand(-1, way_num*query_num, -1, -1, -1),
            query_feat.unsqueeze(2).expand(-1, -1, way_num*shot_num, -1, -1)
        ], dim=3)  # [ep, q, s, 2c, h, w]
        rn_scores = self.relation_net(
            relation_pairs.flatten(0, 2).permute(0, 3, 1, 2)  # [ep*q*s, 2c, h, w]
        ).view(-1, way_num*query_num, way_num*shot_num).mean(dim=2)  # [ep, q, way]

        # 2. 局部特征匹配分支（类似DN4的top-k匹配）
        query_feat = F.normalize(query_feat.flatten(3), p=2, dim=-1)  # [ep, q, c, hw]
        support_feat = F.normalize(
            support_feat.view(-1, way_num, shot_num, *support_feat.shape[-3:]).flatten(3),
            p=2, dim=-1
        )  # [ep, way, s, c, hw]
        
        sim_matrix = torch.einsum('eqcf,ewscf->eqws', query_feat, support_feat)  # [ep, q, way, s, hw]
        topk_sim = torch.topk(sim_matrix, self.n_k, dim=-1).values  # [ep, q, way, s, k]
        local_scores = topk_sim.mean(dim=[-1, -2])  # [ep, q, way]

        # 3. 动态加权融合
        alpha = torch.sigmoid(self.alpha_predictor(
            torch.cat([query_feat.mean((2,3)), support_feat.mean((2,3,4))], dim=-1)
        ))  # [ep, 1]
        combined_score = alpha * rn_scores + (1-alpha) * local_scores
        
        return combined_score.view(-1, way_num)

class QuickBoost(MetricModel):
    def __init__(self, n_k=3, **kwargs):
        super(QuickBoost, self).__init__(**kwargs)
        self.boost_layer = QuickBoostLayer(
            input_size=kwargs.get('inputsize', 64),
            hidden_size=kwargs.get('hidden_size', 8),
            n_k=n_k
        )
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        image, _ = batch
        image = image.to(self.device)
        feat = self.emb_func(image)
        
        # 按episode切分特征
        support_feat, query_feat, _, query_target = self.split_by_episode(
            feat, mode=2
        )
        
        # 计算综合得分
        output = self.boost_layer(
            query_feat, 
            support_feat,
            self.way_num,
            self.shot_num,
            self.query_num
        )
        acc = accuracy(output, query_target.view(-1))
        return output, acc

    def set_forward_loss(self, batch):
        output, acc = self.set_forward(batch)
        loss = self.loss_func(output, batch[-1].view(-1).to(self.device))
        return output, acc, loss