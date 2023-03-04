import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.transformer.attention import MultiHeadAttention
import random


class SAModule(nn.Module):
    def __init__(self, in_size=128 + 64 + 32 + 16, hidden_size=256, head=4, depth=2):
        super().__init__()
        self.use_box_embedding = True
        self.use_dist_weight_matrix = True
        self.use_obj_embedding = True

        self.features_concat = nn.Sequential(
            nn.Conv1d(in_size, hidden_size, 1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
        )
        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))

        self.bbox_embedding = nn.Linear(6, hidden_size)
        self.obj_embedding = nn.Sequential(  # 128 128 128 128
            nn.Linear(135, hidden_size),  # 128+3+3+1
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
        self.self_attn_fc = nn.Sequential(  # 4 32 32 4(head)
            nn.Linear(4, 32),  # xyz, dist
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 4)
        )

    def forward(self, obj_feature, obj_center, obj_size, obj_mask=None):
        """
        Args:
            obj_feature: (max_n_obj, obj_feature_dim)
            obj_center: (max_n_obj, 3)
            obj_size: (max_n_obj, 3)
            obj_mask: (max_n_obj, max_n_obj)
        Returns:
            contextual obj feat
        """
        if self.use_dist_weight_matrix:
            # Attention Weight
            N_K = obj_center.shape[0]
            center_A = obj_center[None, :, :].repeat(N_K, 1, 1)
            center_B = obj_center[:, None, :].repeat(1, N_K, 1)
            center_dist = (center_A - center_B)
            dist = center_dist.pow(2)

            dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
            weights = torch.cat([center_dist, dist], dim=-1).detach().unsqueeze(0)  # 1 N N 4
            dist_weights = self.self_attn_fc(weights).permute(0,3,1,2)  # 1 4 N N
            attention_matrix_way = 'add'
        else:
            dist_weights = None
            attention_matrix_way = 'mul'

        # object size embedding
        features = self.features_concat(obj_feature.unsqueeze(0).permute(0,2,1)).permute(0,2,1)

        if self.use_box_embedding:
            # attention weight
            manual_bbox_feat = torch.cat([obj_center, obj_size], dim=-1).float()
            bbox_embedding = self.bbox_embedding(manual_bbox_feat)
            # print(objects_center.shape, '<< centers shape', flush=True)
            features = features + bbox_embedding

        features = self.self_attn[0](features, features, features, attention_weights=dist_weights, way=attention_matrix_way)

        return features


