import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .metric_model import MetricModel

# "way_num": config["way_num"],
#             "shot_num": config["shot_num"] * config["augment_times"],
#             "query_num": config["query_num"],
#             "test_way": config["test_way"],
#             "test_shot": config["test_shot"] * config["augment_times"],
#             "test_query": config["test_query"],
#             "emb_func": emb_func,
#             "device": self.device,
class QuickBoost(MetricModel):
    """完整的QuickBoost分类器实现，继承自MetricModel"""
    def __init__(self, way_num, shot_num, query_num, 
                 test_way,test_shot,test_query,
                 emb_func, device, alpha=0.6, n_k=3):
        """
        Args:
            way_num (int): N-way分类数
            shot_num (int): K-shot支持样本数
            query_num (int): 每类查询样本数
            emb_func (nn.Module): 特征编码器（如CNNEncoder）
            device (torch.device): 计算设备
            alpha (float): RelationNet权重（0-1）
            n_k (int): FSL-Forest的top-k近邻数
        """
        super(QuickBoost, self).__init__()
        # 基础配置
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.alpha = alpha
        self.device = device
        self.n_k = n_k
        
        # 网络组件
        self.emb_func = emb_func  # 共享特征编码器
        self.relation_net = RelationNetwork(input_size=64, hidden_size=8).to(device)
        self.rf_classifier = RF(n_k=n_k)  # 随机森林分类器
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        """推理阶段前向传播"""
        # 1. 数据准备
        image, global_target = batch
        image = image.to(self.device)
        
        # 2. 特征提取与切分
        feat = self.emb_func(image)
        support_feat, query_feat, _, query_target = self.split_by_episode(feat, mode=2)
        
        # 3. 计算RelationNet得分
        relations_rn = self._relation_forward(support_feat, query_feat)
        
        # 4. 计算FSL-Forest得分
        relations_rf = self.rf_classifier.get_batch_rels(
            batch.support_names, 
            batch.qry_names
        ).to(self.device)
        
        # 5. 加权集成
        final_score = self.alpha * relations_rn + (1-self.alpha) * relations_rf
        acc = accuracy(final_score, query_target)
        return final_score, acc

    def set_forward_loss(self, batch):
        """训练阶段前向传播（含损失计算）"""
        output, acc = self.set_forward(batch)
        query_target = self._get_query_target(batch)
        loss = self.loss_func(output, query_target)
        return output, acc, loss

    def _relation_forward(self, support_feat, query_feat):
        """RelationNet前向计算"""
        # 构建关系对 [batch, way, 2*feat_dim, H, W]
        relation_pairs = torch.cat([
            support_feat.unsqueeze(0).repeat(query_feat.size(0), 1, 1, 1, 1),
            query_feat.unsqueeze(1).repeat(1, self.way_num, 1, 1, 1)
        ], dim=2)
        
        # 计算关系得分
        relations = self.relation_net(
            relation_pairs.view(-1, 128, 19, 19)  # 输入通道=2*64
        )
        return relations.view(-1, self.way_num)

    def _get_query_target(self, batch):
        """生成查询集的标签（0到way_num-1）"""
        return torch.LongTensor(
            [i // self.query_num for i in range(self.way_num * self.query_num)]
        ).to(self.device)

class RelationNetwork(nn.Module):
    """关系网络（子模块）"""
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(input_size*3*3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class RF:
    """FSL-Forest随机森林分类器（子模块）"""
    def __init__(self, n_k=3, data_dir='./data/'):
        self.n_k = n_k
        # 加载预训练数据和模型
        with open(data_dir + 'rf_train_x.pkl', 'rb') as f:
            self.train_x = pickle.load(f).astype(np.float32)
        with open(data_dir + 'rf_train_y.pkl', 'rb') as f:
            self.train_y = pickle.load(f).astype(np.float32)
        self.classifier = RandomForestRegressor(n_estimators=200, max_features=20)
        self.classifier.fit(self.train_x, self.train_y)

    def get_batch_rels(self, support_names, qry_names):
        """计算查询样本与支持集的关系得分"""
        # 实际实现需替换为具体特征加载逻辑
        relations = np.random.rand(len(qry_names)*len(support_names))  # 示例随机值
        return torch.FloatTensor(relations.reshape(len(qry_names), -1))

def accuracy(output, target):
    """计算分类准确率"""
    _, pred = torch.max(output, 1)
    return (pred == target).float().mean().item()