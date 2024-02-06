import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import clip
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GCNConv
from torch_scatter import scatter

from model.model_utils.model_base import BaseModel
from clip_adapter.model import AdapterModel
from src.model.transformer.attention import MultiHeadAttention
from src.utils.eva_utils_acc import (evaluate_topk_object,
                                 evaluate_topk_predicate,
                                 evaluate_triplet_topk, get_gt)
from utils import op_utils

################################
# Just Copy From SGGpoint Repo #
################################

#####################################################
#                                                   #
#                                                   #
#   Backbone network - PointNet                     #
#                                                   #
#                                                   #
#####################################################

class PointNet(nn.Module):
    # from DGCNN's repo
    def __init__(self, input_channel, embeddings):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, embeddings, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(embeddings)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        return x

#####################################################
#                                                   #
#                                                   #
#   Backbone network - DGCNN (and its components)   #
#                                                   #
#                                                   #
#####################################################

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class DGCNN(nn.Module):
    # official DGCNN
    def __init__(self, input_channel, embeddings):
        super(DGCNN, self).__init__()
        self.k = 20
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel * 2, 64, kernel_size=1, bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),nn.BatchNorm2d(64),nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),nn.BatchNorm2d(256),nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, embeddings, kernel_size=1, bias=False),nn.BatchNorm1d(embeddings),nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        #x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv5(x)
        return x

##################################################
#                                                #
#                                                #
#  Core Network: EdgeGCN                         #
#                                                #
#                                                #
##################################################

class EdgeGCN(torch.nn.Module):
    def __init__(self, num_node_in_embeddings, num_edge_in_embeddings, AttnEdgeFlag, AttnNodeFlag):
        super(EdgeGCN, self).__init__()

        self.node_GConv1 = GCNConv(num_node_in_embeddings, num_node_in_embeddings // 2, add_self_loops=True)
        self.node_GConv2 = GCNConv(num_node_in_embeddings // 2, num_node_in_embeddings, add_self_loops=True)

        self.edge_MLP1 = nn.Sequential(nn.Conv1d(num_edge_in_embeddings, num_edge_in_embeddings // 2, 1), nn.ReLU())
        self.edge_MLP2 = nn.Sequential(nn.Conv1d(num_edge_in_embeddings // 2, num_edge_in_embeddings, 1), nn.ReLU())

        self.AttnEdgeFlag = AttnEdgeFlag # boolean (for ablaiton studies)
        self.AttnNodeFlag = AttnNodeFlag # boolean (for ablaiton studies)

        # multi-dimentional (N-Dim) node/edge attn coefficients mappings
        self.edge_attentionND = nn.Linear(num_edge_in_embeddings, num_node_in_embeddings // 2) if self.AttnEdgeFlag else None
        self.node_attentionND = nn.Linear(num_node_in_embeddings, num_edge_in_embeddings // 2) if self.AttnNodeFlag else None

        self.node_indicator_reduction = nn.Linear(num_edge_in_embeddings, num_edge_in_embeddings // 2) if self.AttnNodeFlag else None

    def concate_NodeIndicator_for_edges(self, node_indicator, batchwise_edge_index):
        node_indicator = node_indicator.squeeze(0)
        
        edge_index_list = batchwise_edge_index.t()
        subject_idx_list = edge_index_list[:, 0]
        object_idx_list = edge_index_list[:, 1]

        subject_indicator = node_indicator[subject_idx_list]  # (num_edges, num_mid_channels)
        object_indicator = node_indicator[object_idx_list]    # (num_edges, num_mid_channels)

        edge_concat = torch.cat((subject_indicator, object_indicator), dim=-1)
        return edge_concat  # (num_edges, num_mid_channels * 2)

    def forward(self, node_feats, edge_feats, edge_index):
        # prepare node_feats & edge_feats in the following formats
        # node_feats: (1, num_nodes,  num_embeddings)
        # edge_feats: (1, num_edges,  num_embeddings)
        # (num_embeddings = num_node_in_embeddings = num_edge_in_embeddings) = 2 * num_mid_channels
        
        #### Deriving Edge Attention
        if self.AttnEdgeFlag:
            edge_indicator = self.edge_attentionND(edge_feats.squeeze(0)).unsqueeze(0).permute(0, 2, 1)  # (1, num_mid_channels, num_edges)
            raw_out_row = scatter(edge_indicator, edge_index.t()[:, 0].squeeze(0), dim=2, reduce='mean', dim_size=node_feats.size(0)) # (1, num_mid_channels, num_nodes)
            raw_out_col = scatter(edge_indicator, edge_index.t()[:, 1].squeeze(0), dim=2, reduce='mean', dim_size=node_feats.size(0)) # (1, num_mid_channels, num_nodes)
            agg_edge_indicator_logits = raw_out_row * raw_out_col                                        # (1, num_mid_channels, num_nodes)
            agg_edge_indicator = torch.sigmoid(agg_edge_indicator_logits).permute(0, 2, 1).squeeze(0)    # (num_nodes, num_mid_channels)
        else:
            agg_edge_indicator = 1

        #### Node Evolution Stream (NodeGCN)
        node_feats = F.relu(self.node_GConv1(node_feats, edge_index)) * agg_edge_indicator # applying EdgeAttn on Nodes
        node_feats = F.dropout(node_feats, training=self.training)
        node_feats = F.relu(self.node_GConv2(node_feats, edge_index))
        node_feats = node_feats.unsqueeze(0)  # (1, num_nodes, num_embeddings)

        #### Deriving Node Attention
        if self.AttnNodeFlag:
            node_indicator = F.relu(self.node_attentionND(node_feats.squeeze(0)).unsqueeze(0))                  # (1, num_mid_channels, num_nodes)
            agg_node_indicator = self.concate_NodeIndicator_for_edges(node_indicator, edge_index)               # (num_edges, num_mid_channels * 2)
            agg_node_indicator = self.node_indicator_reduction(agg_node_indicator).unsqueeze(0).permute(0,2,1)  # (1, num_mid_channels, num_edges)
            agg_node_indicator = torch.sigmoid(agg_node_indicator)  # (1, num_mid_channels, num_edges)
        else:
            agg_node_indicator = 1

        #### Edge Evolution Stream (EdgeMLP)
        edge_feats = edge_feats.unsqueeze(0).permute(0, 2, 1)                  # (1, num_embeddings, num_edges)
        edge_feats = self.edge_MLP1(edge_feats)                   # (1, num_mid_channels, num_edges)
        edge_feats = F.dropout(edge_feats, training=self.training) * agg_node_indicator    # applying NodeAttn on Edges
        edge_feats = self.edge_MLP2(edge_feats).permute(0, 2, 1)  # (1, num_edges, num_embeddings)

        return node_feats.squeeze(0), edge_feats.squeeze(0)

class MMEdgeGCN(torch.nn.Module):
    
    def __init__(self, num_node_in_embeddings, num_edge_in_embeddings, AttnEdgeFlag, AttnNodeFlag, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.self_attn = MultiHeadAttention(d_model=num_node_in_embeddings, d_k=num_node_in_embeddings // num_heads, d_v=num_node_in_embeddings // num_heads, h=num_heads)
        self.cross_attn = MultiHeadAttention(d_model=num_node_in_embeddings, d_k=num_node_in_embeddings // num_heads, d_v=num_node_in_embeddings // num_heads, h=num_heads) 
        self.cross_attn_rel = MultiHeadAttention(d_model=num_edge_in_embeddings, d_k=num_edge_in_embeddings // num_heads, d_v=num_edge_in_embeddings // num_heads, h=num_heads)

        self.edgegcn_2d = EdgeGCN(num_node_in_embeddings, num_edge_in_embeddings, AttnEdgeFlag, AttnNodeFlag)
        self.edgegcn_3d = EdgeGCN(num_node_in_embeddings, num_edge_in_embeddings, AttnEdgeFlag, AttnNodeFlag)
        
        self.self_attn_fc = nn.Sequential(  # 11 32 32 4(head)
            nn.Linear(4, 32),  # xyz, dist
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_heads)
        )
    
    def forward(self, obj_feature_3d, obj_feature_2d, edge_feature_3d, edge_feature_2d, edge_index, batch_ids, obj_center=None):
        # compute weight for obj
        if obj_center is not None:
            # get attention weight for object
            batch_size = batch_ids.max().item() + 1
            N_K = obj_feature_3d.shape[0]
            obj_mask = torch.zeros(1, 1, N_K, N_K).cuda()
            obj_distance_weight = torch.zeros(1, self.num_heads, N_K, N_K).cuda()
            count = 0

            for i in range(batch_size):

                idx_i = torch.where(batch_ids == i)[0]
                obj_mask[:, :, count:count + len(idx_i), count:count + len(idx_i)] = 1
            
                center_A = obj_center[None, idx_i, :].clone().detach().repeat(len(idx_i), 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, len(idx_i), 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
                weights = torch.cat([center_dist, dist], dim=-1).unsqueeze(0)  # 1 N N 4
                dist_weights = self.self_attn_fc(weights).permute(0,3,1,2)  # 1 num_heads N N
                
                attention_matrix_way = 'add'
                obj_distance_weight[:, :, count:count + len(idx_i), count:count + len(idx_i)] = dist_weights

                count += len(idx_i)
        else:
            obj_mask = None
            obj_distance = None
            attention_matrix_way = 'mul'
        
        obj_feature_3d = obj_feature_3d.unsqueeze(0)
        obj_feature_2d = obj_feature_2d.unsqueeze(0)
        
        obj_feature_3d = self.self_attn(obj_feature_3d, obj_feature_3d, obj_feature_3d, attention_weights=obj_distance_weight, way=attention_matrix_way, attention_mask=obj_mask, use_knn=False)
        obj_feature_2d = self.cross_attn(obj_feature_2d, obj_feature_3d, obj_feature_3d, attention_weights=obj_distance_weight, way=attention_matrix_way, attention_mask=obj_mask, use_knn=False)
        
        obj_feature_3d = obj_feature_3d.squeeze(0)
        obj_feature_2d = obj_feature_2d.squeeze(0) 

        obj_feature_3d, edge_feature_3d = self.edgegcn_3d(obj_feature_3d, edge_feature_3d, edge_index)
        obj_feature_2d, edge_feature_2d = self.edgegcn_2d(obj_feature_2d, edge_feature_2d, edge_index)

        edge_feature_2d = edge_feature_2d.unsqueeze(0)
        edge_feature_3d = edge_feature_3d.unsqueeze(0)
        
        edge_feature_2d = self.cross_attn_rel(edge_feature_2d, edge_feature_3d, edge_feature_3d)
        
        edge_feature_2d = edge_feature_2d.squeeze(0)
        edge_feature_3d = edge_feature_3d.squeeze(0)

        return obj_feature_3d, edge_feature_3d, obj_feature_2d, edge_feature_2d

###############################################
#                                             #
#                                             #
#   Tail Classification - NodeMLP & EdgeMLP   #
#                                             #
#                                             #
###############################################

# class NodeMLP(nn.Module):
#     def __init__(self, embeddings, nObjClasses, negative_slope=0.2):
#         super(NodeMLP, self).__init__()
#         mid_channels = embeddings // 2
#         self.node_linear1 = nn.Linear(embeddings, mid_channels, bias=False)
#         self.node_BnReluDp = nn.Sequential(nn.BatchNorm1d(mid_channels), nn.LeakyReLU(negative_slope), nn.Dropout())
#         self.node_linear2 = nn.Linear(mid_channels, nObjClasses, bias=False)

#     def forward(self, node_feats):
#         # node_feats: (1, nodes, embeddings)  => node_logits: (1, nodes, nObjClasses)
#         x = self.node_linear1(node_feats.unsqueeze(0))
#         x = self.node_BnReluDp(x.permute(0, 2, 1)).permute(0, 2, 1)
#         node_logits = self.node_linear2(x)
#         return node_logits.squeeze(0)

class EdgeMLP(nn.Module):
    def __init__(self, embeddings, nRelClasses, negative_slope=0.2):
        super(EdgeMLP, self).__init__()
        mid_channels = embeddings // 2
        self.edge_linear1 = nn.Linear(embeddings, mid_channels, bias=False)
        self.edge_BnReluDp = nn.Sequential(nn.BatchNorm1d(mid_channels), nn.LeakyReLU(negative_slope), nn.Dropout())
        self.edge_linear2 = nn.Linear(mid_channels, nRelClasses, bias=False)

    def forward(self, edge_feats):
        # edge_feats: (1, edges, embeddings)  => edge_logits: (1, edges, nRelClasses)
        x = self.edge_linear1(edge_feats.unsqueeze(0))
        x = self.edge_BnReluDp(x.permute(0, 2, 1)).permute(0, 2, 1)
        edge_logits = self.edge_linear2(x)
        # we treat it as multi-label classification
        edge_logits = torch.sigmoid(edge_logits)
        return edge_logits.squeeze(0)

#####################################################
#                                                   #
#                                                   #
#   SGGpoint Model                                  #
#                                                   #
#                                                   #
#####################################################

def edge_feats_initialization(node_feats, batchwise_edge_index):

    connections_from_subject_to_object = batchwise_edge_index.t()
    subject_idx = connections_from_subject_to_object[:, 0]
    object_idx = connections_from_subject_to_object[:, 1]

    subject_feats = node_feats[subject_idx]
    object_feats = node_feats[object_idx]
    diff_feats = object_feats - subject_feats

    edge_feats = torch.cat((subject_feats, diff_feats), dim=1)  # equivalent to EdgeConv (with in DGCNN)

    return edge_feats  # (num_Edges, Embeddings * 2)

class SGGpoint(BaseModel):
    # architecture
    def __init__(self, config, num_obj_class, num_rel_class):
        super().__init__('SGGpoint', config)
        self.config = config
        self.mconfig = config.MODEL
        self.backbone = nn.Sequential(
            #PointNet(input_channel=9, embeddings=128),
            DGCNN(input_channel=3, embeddings=768)
        )
        self.mlp_3d = torch.nn.Linear(512 + 256, 512 - 8)
        self.clip_adapter = AdapterModel(input_size=512, output_size=512, alpha=0.5)
        
        self.edge_mlp_2d = torch.nn.Linear(512 * 2, 512 - 11)
        self.edge_mlp_3d = torch.nn.Linear(512 * 2, 512 - 11)
        self.edge_gcn = MMEdgeGCN(num_node_in_embeddings=512, num_edge_in_embeddings=512, AttnNodeFlag=True, AttnEdgeFlag=True, num_heads=self.mconfig.NUM_HEADS)
        
        self.obj_mlp_2d = nn.Linear(512 * 2, 512)
        self.rel_mlp_2d = nn.Linear(512 * 2, 512)
        self.obj_mlp_3d = nn.Linear(512 * 2, 512)
        self.rel_mlp_3d = nn.Linear(512 * 2, 512)
        
        self.triplet_projector_3d = torch.nn.Sequential(
            torch.nn.Linear(512 * 3, 512 * 2),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512 * 2, 512)
        )
        self.triplet_projector_2d = torch.nn.Sequential(
            torch.nn.Linear(512 * 3, 512 * 2),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512 * 2, 512)
        )
        
        self.obj_classifier_2d = torch.nn.Linear(512, num_obj_class, bias=False)
        self.rel_classifier_2d = EdgeMLP(embeddings=512, nRelClasses=num_rel_class)
        self.obj_classifier_3d = torch.nn.Linear(512, num_obj_class, bias=False)
        self.rel_classifier_3d = EdgeMLP(embeddings=512, nRelClasses=num_rel_class)

        self.obj_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.init_weight(obj_label_path='/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/classes.txt', \
                         rel_label_path='/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/relations.txt', \
                         adapter_path='/data/caidaigang/project/3DSSG_Repo/clip_adapter/checkpoint/origin_mean.pth')

        self.optimizer = optim.Adam([
            {'params':self.backbone.parameters(), 'lr':float(1e-4), 'weight_decay':float(1e-5), 'amsgrad':False},
            {'params':self.mlp_3d.parameters(), 'lr':float(1e-4), 'weight_decay':False, 'amsgrad':False},
            {'params':self.edge_mlp_2d.parameters(), 'lr':float(1e-4), 'weight_decay':False, 'amsgrad':False},
            {'params':self.edge_mlp_3d.parameters(), 'lr':float(1e-4), 'weight_decay':False, 'amsgrad':False},
            {'params':self.edge_gcn.parameters(), 'lr':float(1e-5), 'weight_decay':float(1e-5), 'amsgrad':False},
            {'params':self.triplet_projector_3d.parameters(), 'lr':float(1e-4), 'weight_decay':False, 'amsgrad':False},
            {'params':self.triplet_projector_2d.parameters(), 'lr':float(1e-4), 'weight_decay':False, 'amsgrad':False},
            {'params':self.obj_mlp_2d.parameters(), 'lr':float(1e-4), 'weight_decay':False, 'amsgrad':False},
            {'params':self.rel_mlp_2d.parameters(), 'lr':float(1e-4), 'weight_decay':float(1e-4), 'amsgrad':False},
            {'params':self.obj_mlp_3d.parameters(), 'lr':float(1e-4), 'weight_decay':float(1e-4), 'amsgrad':False},
            {'params':self.rel_mlp_3d.parameters(), 'lr':float(1e-4), 'weight_decay':False, 'amsgrad':False},
            {'params':self.obj_classifier_2d.parameters(), 'lr':float(1e-5), 'weight_decay':False, 'amsgrad':False},
            {'params':self.rel_classifier_2d.parameters(), 'lr':float(1e-4), 'weight_decay':False, 'amsgrad':False},
            {'params':self.obj_classifier_3d.parameters(), 'lr':float(1e-5), 'weight_decay':False, 'amsgrad':False},
            {'params':self.rel_classifier_3d.parameters(), 'lr':float(1e-4), 'weight_decay':False, 'amsgrad':False},
            {'params':self.obj_logit_scale, 'lr':float(1e-4), 'weight_decay':False, 'amsgrad':False},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()
    
    def init_weight(self, obj_label_path, rel_label_path, adapter_path):
        torch.nn.init.xavier_uniform_(self.mlp_3d.weight)
        torch.nn.init.xavier_uniform_(self.obj_mlp_3d.weight)
        torch.nn.init.xavier_uniform_(self.obj_mlp_2d.weight)
        torch.nn.init.xavier_uniform_(self.rel_mlp_3d.weight)
        torch.nn.init.xavier_uniform_(self.rel_mlp_2d.weight)
        torch.nn.init.xavier_uniform_(self.edge_mlp_2d.weight)
        torch.nn.init.xavier_uniform_(self.edge_mlp_3d.weight)

        obj_text_features, rel_text_feature = self.get_label_weight(obj_label_path, rel_label_path)
        # node feature classifier        
        self.obj_classifier_2d.weight.data.copy_(obj_text_features)
        for param in self.obj_classifier_2d.parameters():
            param.requires_grad = True

        self.obj_classifier_3d.weight.data.copy_(obj_text_features)
        for param in self.obj_classifier_3d.parameters():
            param.requires_grad = True

        self.clip_adapter.load_state_dict(torch.load(adapter_path, 'cpu'))
        # freeze clip adapter
        for param in self.clip_adapter.parameters():
            param.requires_grad = False
        
        self.obj_logit_scale.requires_grad = True
    
    def get_label_weight(self, obj_label_path, rel_label_path):
        
        self.obj_label_list = []
        self.rel_label_list = []
        self.clip_model, preprocess = clip.load("ViT-B/32", device='cuda')

        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        with open(obj_label_path, "r") as f:
            data = f.readlines()
        for line in data:
            self.obj_label_list.append(line.strip())
        
        with open(rel_label_path, "r") as f:
            data = f.readlines()
        for line in data:
            self.rel_label_list.append(line.strip())
        
        # get norm clip weight
        obj_prompt = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.obj_label_list]).cuda()
        rel_prompt = torch.cat([clip.tokenize(f"{c}") for c in self.rel_label_list]).cuda()

        with torch.no_grad():
            obj_text_features = self.clip_model.encode_text(obj_prompt)
            rel_text_features = self.clip_model.encode_text(rel_prompt)
        
        obj_text_features = obj_text_features / obj_text_features.norm(dim=-1, keepdim=True)
        rel_text_features = rel_text_features / rel_text_features.norm(dim=-1, keepdim=True)

        return obj_text_features.float(), rel_text_features.float() 
    
    def cosine_loss(self, A, B, t=1):
        return torch.clamp(t - F.cosine_similarity(A, B, dim=-1), min=0).mean()
    
    def get_rel_emb(self, objs_target, rels_target, edges):
        
        target_rel_tokens, target_rel_feats = [], []
        rel_index =  []
        for edge_index in range(len(edges)):
            idx_eo = edges[edge_index][0]
            idx_os = edges[edge_index][1]
            target_eo = self.obj_label_list[objs_target[idx_eo]]
            target_os = self.obj_label_list[objs_target[idx_os]]
            assert rels_target.ndim == 2
            if rels_target[edge_index].sum() == 0:
                target_rel_tokens.append(clip.tokenize(f"the {target_eo} and the {target_os} has no relation in the point cloud"))
                rel_index.append(edge_index)
            else:
                for i in range(rels_target.shape[-1]):
                    if rels_target[edge_index][i] == 1:
                        target_rel = self.rel_label_list[i]
                        # target_rel_tokens.append(clip.tokenize(f"a {target_eo} is {target_rel} a {target_os}"))
                        target_rel_tokens.append(clip.tokenize(f"a point cloud of a {target_eo} {target_rel} a {target_os}"))
                        rel_index.append(edge_index)

        prompt_features = torch.cat(target_rel_tokens).cuda()
        with torch.no_grad():
            triplet_feats = self.clip_model.encode_text(prompt_features)
        
        # deal with multi label
        for edge_index in range(len(edges)):
            multi_rel_idxs = torch.where(torch.tensor(rel_index) == edge_index)[0]
            target_rel_feats.append(triplet_feats[multi_rel_idxs].reshape(len(multi_rel_idxs), 512).mean(0))
        
        assert len(target_rel_feats) ==  len(edges)
        target_rel_feats = torch.vstack(target_rel_feats)
        target_rel_feats = target_rel_feats / target_rel_feats.norm(dim=-1, keepdim=True)
        
        return target_rel_feats.float()
  
    def generate_object_pair_features(self, obj_feats, edges_feats, edge_indice):
        obj_pair_feats = []
        for (edge_feat, edge_index) in zip(edges_feats, edge_indice.t()):
            obj_pair_feats.append(torch.cat([obj_feats[edge_index[0]], obj_feats[edge_index[1]], edge_feat], dim=-1))
        obj_pair_feats = torch.vstack(obj_pair_feats)
        return obj_pair_feats
    
    def forward(self, obj_points, obj_2d_feats, edge_indices, descriptor=None, batch_ids=None, istrain=False):
        # Generate 3d node initial feature
        x = self.backbone(obj_points)
        obj_3d_feats = torch.max(x, 2)[0] # perform maxpooling
        if istrain:
            obj_feature_3d_mimic = obj_3d_feats[..., :512].clone()
        obj_3d_feats = self.mlp_3d(obj_3d_feats)
        if self.mconfig.USE_SPATIAL:
            tmp = descriptor[:,3:].clone()
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            obj_3d_feats = torch.cat([obj_3d_feats, tmp],dim=-1)
        
        # Generate 3d edge initial feature
        edge_feats_3d = edge_feats_initialization(obj_3d_feats.clone(), edge_indices)
        edge_feats_3d = self.edge_mlp_3d(edge_feats_3d)
        # Genrate location infomation
        with torch.no_grad():
            x_i = descriptor[edge_indices[0]]
            x_j = descriptor[edge_indices[1]]
            edge_feats_3d_des = torch.zeros_like(x_i)
            edge_feats_3d_des[:,0:3] = x_i[:,0:3]-x_j[:,0:3]
            # std  offset
            edge_feats_3d_des[:,3:6] = x_i[:,3:6]-x_j[:,3:6]
            # dim log ratio
            edge_feats_3d_des[:,6:9] = torch.log(x_i[:,6:9] / x_j[:,6:9])
            # volume log ratio
            edge_feats_3d_des[:,9] = torch.log( x_i[:,9] / x_j[:,9])
            # length log ratio
            edge_feats_3d_des[:,10] = torch.log( x_i[:,10] / x_j[:,10])
            edge_feats_2d_des = edge_feats_3d_des.clone()
        edge_feats_3d = torch.cat((edge_feats_3d, edge_feats_3d_des), dim=-1)
        
        # Generate 2d node initial feature
        with torch.no_grad():
            obj_2d_feats = self.clip_adapter(obj_2d_feats)
        obj_features_2d_mimic = obj_2d_feats.clone()
        
        # Generate 2d edge initial feature
        edge_feats_2d = edge_feats_initialization(obj_2d_feats.clone(), edge_indices)
        edge_feats_2d = self.edge_mlp_2d(edge_feats_2d)
        edge_feats_2d = torch.cat((edge_feats_2d, edge_feats_2d_des), dim=-1)
        
        obj_3d_feats_ = obj_3d_feats.clone()
        obj_2d_feats_ = obj_2d_feats.clone()
        rel_3d_feats_ = edge_feats_3d.clone()
        rel_2d_feats_ = edge_feats_2d.clone()
        
        # EdgeGCN
        obj_center = descriptor[:, :3].clone()
        gcn_obj_feature_3d, gcn_edge_feature_3d, gcn_obj_feature_2d, gcn_edge_feature_2d \
             = self.edge_gcn(obj_3d_feats, obj_2d_feats, edge_feats_3d, edge_feats_2d, edge_indices, batch_ids, obj_center)
        
        gcn_obj_feature_3d = self.obj_mlp_3d(torch.cat([obj_3d_feats_, gcn_obj_feature_3d.squeeze(0)], dim=-1))
        gcn_obj_feature_2d = self.obj_mlp_2d(torch.cat([obj_2d_feats_, gcn_obj_feature_2d.squeeze(0)], dim=-1))
        gcn_edge_feature_3d = self.rel_mlp_3d(torch.cat([rel_3d_feats_, gcn_edge_feature_3d.squeeze(0)], dim=-1))
        gcn_edge_feature_2d = self.rel_mlp_2d(torch.cat([rel_2d_feats_, gcn_edge_feature_2d.squeeze(0)], dim=-1))
        
        # Generate triplet features
        gcn_edge_feature_3d_dis = self.generate_object_pair_features(gcn_obj_feature_3d, gcn_edge_feature_3d, edge_indices)
        gcn_edge_feature_2d_dis = self.generate_object_pair_features(gcn_obj_feature_2d, gcn_edge_feature_2d, edge_indices)
        
        gcn_edge_feature_3d_dis = self.triplet_projector_3d(gcn_edge_feature_3d_dis)
        gcn_edge_feature_2d_dis = self.triplet_projector_2d(gcn_edge_feature_2d_dis)
        
        logit_scale = self.obj_logit_scale.exp()
        obj_logits_3d = logit_scale * self.obj_classifier_3d(gcn_obj_feature_3d / gcn_obj_feature_3d.norm(dim=-1, keepdim=True))
        obj_logits_2d = logit_scale * self.obj_classifier_2d(gcn_obj_feature_2d / gcn_obj_feature_2d.norm(dim=-1, keepdim=True))

        rel_logits_3d = self.rel_classifier_3d(gcn_edge_feature_3d.squeeze(0))
        rel_logits_2d = self.rel_classifier_2d(gcn_edge_feature_2d.squeeze(0))
        
        if istrain:
            return obj_logits_3d, obj_logits_2d, rel_logits_3d, rel_logits_2d, obj_feature_3d_mimic, obj_features_2d_mimic, gcn_edge_feature_3d_dis, gcn_edge_feature_2d_dis, logit_scale
        else:
            return obj_logits_3d, obj_logits_2d, rel_logits_3d, rel_logits_2d
    
    def process_train(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration += 1 

        obj_logits_3d, obj_logits_2d, rel_cls_3d, rel_cls_2d, obj_feature_3d, obj_feature_2d, edge_feature_3d, edge_feature_2d, obj_logit_scale\
             = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=True)
        
        # compute object loss
        loss_obj_3d = F.cross_entropy(obj_logits_3d, gt_cls)
        loss_obj_2d = F.cross_entropy(obj_logits_2d, gt_cls)
        
        # compute predicate loss
        batch_mean = torch.sum(gt_rel_cls, dim=(0))
        zeros = (gt_rel_cls.sum(-1) ==0).sum().unsqueeze(0)
        batch_mean = torch.cat([zeros,batch_mean],dim=0)
        weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
            
        weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
        weight = weight[1:]
        loss_rel_3d = F.binary_cross_entropy(rel_cls_3d, gt_rel_cls, weight=weight)
        loss_rel_2d = F.binary_cross_entropy(rel_cls_2d, gt_rel_cls, weight=weight)

        # compute mimic loss
        obj_feature_3d = obj_feature_3d / obj_feature_3d.norm(dim=-1, keepdim=True)
        obj_feature_2d = obj_feature_2d / obj_feature_2d.norm(dim=-1, keepdim=True)
        loss_mimic = self.cosine_loss(obj_feature_3d, obj_feature_2d, t=0.8)

        # compute triplet loss
        rel_text_feat = self.get_rel_emb(gt_cls, gt_rel_cls, edge_indices)
        edge_feature_2d = edge_feature_2d / edge_feature_2d.norm(dim=-1, keepdim=True)

        rel_mimic_2d = F.l1_loss(edge_feature_2d, rel_text_feat)
        
        loss = 0.1 * (loss_obj_3d + loss_obj_2d) + 3 * (loss_rel_3d + loss_rel_2d) + 0.1 * (loss_mimic + rel_mimic_2d)
        self.backward(loss)

        # compute 3d metric
        top_k_obj = evaluate_topk_object(obj_logits_3d.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_cls_3d.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]

        # compute 2d metric
        top_k_obj = evaluate_topk_object(obj_logits_2d.detach(), gt_cls, topk=11)
        top_k_rel = evaluate_topk_predicate(rel_cls_2d.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        obj_topk_2d_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_2d_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]       
        
        log = [("train/rel_loss", loss_rel_3d.detach().item()),
                ("train/obj_loss", loss_obj_3d.detach().item()),
                ("train/2d_rel_loss", loss_rel_2d.detach().item()),
                ("train/2d_obj_loss", loss_obj_2d.detach().item()),
                ("train/mimic_loss", loss_mimic.detach().item()),
                ("train/logit_scale", obj_logit_scale.detach().item()),
                ("train/rel_mimic_loss_2d", rel_mimic_2d.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
                ("train/Obj_R1_2d", obj_topk_2d_list[0]),
                ("train/Obj_R5_2d", obj_topk_2d_list[1]),
                ("train/Obj_R10_2d", obj_topk_2d_list[2]),
                ("train/Pred_R1_2d", rel_topk_2d_list[0]),
                ("train/Pred_R3_2d", rel_topk_2d_list[1]),
                ("train/Pred_R5_2d", rel_topk_2d_list[2]),
            ]
        return log
    
    def process_val(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, use_triplet=False):
        obj_logits_3d, obj_logits_2d, rel_cls_3d, rel_cls_2d = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
         # compute metric
        top_k_obj = evaluate_topk_object(obj_logits_3d.detach().cpu(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_cls_3d.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)

        top_k_obj_2d = evaluate_topk_object(obj_logits_2d.detach().cpu(), gt_cls, topk=11)
        top_k_rel_2d = evaluate_topk_predicate(rel_cls_2d.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        
        if use_triplet:
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_logits_3d.detach().cpu(), rel_cls_3d.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=True, obj_topk=top_k_obj)
            top_k_2d_triplet, _, _, _, _ = evaluate_triplet_topk(obj_logits_2d.detach().cpu(), rel_cls_2d.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=True, obj_topk=top_k_obj)
        else:
            top_k_triplet = [101]
            cls_matrix = None
            sub_scores = None
            obj_scores = None
            rel_scores = None
        
        return top_k_obj, top_k_obj_2d, top_k_rel, top_k_rel_2d, top_k_triplet, top_k_2d_triplet, cls_matrix, sub_scores, obj_scores, rel_scores
  
    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()
