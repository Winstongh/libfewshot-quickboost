import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# 从正确的位置导入
from core.utils import accuracy
from .metric_model import MetricModel  # 相对导入


class RelationNetwork(nn.Module):
    """Relation Network module"""
    def __init__(self, input_channels=128, hidden_dim=8):
        super(RelationNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Calculate the flattened feature size: 64 * 4 * 4 = 1024
        self.fc1 = nn.Linear(1024, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


class QuickBoost(MetricModel):
    def __init__(self, inputsize=64, hidden_size=8, relation_dim=8, **kwargs):
        super(QuickBoost, self).__init__(**kwargs)
        
        self.inputsize = inputsize
        self.hidden_size = hidden_size
        self.relation_dim = relation_dim
        
        # Relation Network
        self.relation_net = RelationNetwork(inputsize * 2, relation_dim)
        
        # Loss function
        self.loss_func = nn.MSELoss()

    def set_forward(self, batch):
        """
        Forward pass for evaluation/testing
        """
        image, global_target = batch
        image = image.to(self.device)
        
        # Extract features
        feat = self.emb_func(image)
        
        # Split features into support and query sets
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=3  # 4D input returning 4D output
        )
        
        # Calculate relations
        relations = self._calculate_relations(support_feat, query_feat)
        
        # Calculate accuracy
        acc = accuracy(relations, query_target.reshape(-1))
        
        return relations, acc

    def set_forward_loss(self, batch):
        """
        Forward pass for training with loss calculation
        """
        image, global_target = batch
        image = image.to(self.device)
        
        # Extract features
        feat = self.emb_func(image)
        
        # Split features into support and query sets
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=3  # 4D input returning 4D output
        )
        
        # Calculate relations
        relations = self._calculate_relations(support_feat, query_feat)
        
        # Construct one-hot labels
        one_hot_labels = self._construct_one_hot(query_target.reshape(-1))
        
        # Calculate loss
        loss = self.loss_func(relations, one_hot_labels)
        
        # Calculate accuracy
        acc = accuracy(relations, query_target.reshape(-1))
        
        return relations, acc, loss

    def _calculate_relations(self, support_features, query_features):
        """
        Calculate relations between support and query features
        """
        # Get dimensions
        way_num = self.way_num
        shot_num = self.shot_num
        query_num = self.query_num
        
        # Reshape support features: average over shots
        support_features = support_features.view(way_num, shot_num, 
                                               support_features.size(1),
                                               support_features.size(2),
                                               support_features.size(3))
        support_features = torch.mean(support_features, dim=1)  # [way_num, c, h, w]
        
        # Reshape query features
        query_features = query_features.view(way_num * query_num,
                                           query_features.size(1),
                                           query_features.size(2),
                                           query_features.size(3))
        
        # Expand features for relation calculation
        support_features_ext = support_features.unsqueeze(0).repeat(way_num * query_num, 1, 1, 1, 1)
        query_features_ext = query_features.unsqueeze(1).repeat(1, way_num, 1, 1, 1)
        
        # Concatenate features
        relation_pairs = torch.cat((support_features_ext, query_features_ext), 2)
        relation_pairs = relation_pairs.view(-1, self.inputsize * 2,
                                           relation_pairs.size(3),
                                           relation_pairs.size(4))
        
        # Pass through relation network
        relations = self.relation_net(relation_pairs)
        relations = relations.view(way_num * query_num, way_num)
        
        return torch.sigmoid(relations)

    def _construct_one_hot(self, labels):
        """Construct one-hot labels"""
        way_num = self.way_num
        one_hot = torch.zeros(labels.size(0), way_num).to(labels.device)
        one_hot.scatter_(1, labels.long().view(-1, 1), 1)
        return one_hot