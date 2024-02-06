import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GCNConv
from torch_scatter import scatter

from model.model_utils.model_base import BaseModel
from src.model.model_utils.network_PointNet import PointNetfeat
from src.utils.eva_utils_acc import ( evaluate_topk_object,
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

        return node_feats, edge_feats

###############################################
#                                             #
#                                             #
#   Tail Classification - NodeMLP & EdgeMLP   #
#                                             #
#                                             #
###############################################

class NodeMLP(nn.Module):
    def __init__(self, embeddings, nObjClasses, negative_slope=0.2):
        super(NodeMLP, self).__init__()
        mid_channels = embeddings // 2
        self.node_linear1 = nn.Linear(embeddings, mid_channels, bias=False)
        self.node_BnReluDp = nn.Sequential(nn.BatchNorm1d(mid_channels), nn.LeakyReLU(negative_slope), nn.Dropout())
        self.node_linear2 = nn.Linear(mid_channels, nObjClasses, bias=False)

    def forward(self, node_feats):
        # node_feats: (1, nodes, embeddings)  => node_logits: (1, nodes, nObjClasses)
        x = self.node_linear1(node_feats.unsqueeze(0))
        x = self.node_BnReluDp(x.permute(0, 2, 1)).permute(0, 2, 1)
        node_logits = self.node_linear2(x)
        return node_logits.squeeze(0)

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
            DGCNN(input_channel=3, embeddings=256 - 8)
        )
        self.edge_mlp = nn.Linear(512, 512 - 11)
        self.edge_gcn = EdgeGCN(num_node_in_embeddings=256, num_edge_in_embeddings=512, AttnNodeFlag=True, AttnEdgeFlag=True)
        self.obj_mlp = nn.Linear(256 * 2, 256)
        self.rel_mlp = nn.Linear(512 * 2, 512)
        self.obj_classifier = NodeMLP(embeddings=256, nObjClasses=num_obj_class)
        self.rel_classifier = EdgeMLP(embeddings=512, nRelClasses=num_rel_class)
        
        self.optimizer = optim.Adam([
            {'params':self.backbone.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
            {'params':self.edge_gcn.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
            {'params':self.edge_mlp.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
            {'params':self.obj_mlp.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
            {'params':self.rel_mlp.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
            {'params':self.obj_classifier.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
            {'params':self.rel_classifier.parameters(), 'lr':float(1e-3), 'weight_decay':float(1e-4), 'amsgrad':False},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()
    
    def forward(self, obj_points, obj_2d_feats, edge_indices, descriptor=None, batch_ids=None, istrain=False):

        # Generate node initial feature
        x = self.backbone(obj_points)
        node_feats = torch.max(x, 2)[0] # perform maxpooling
        # Generate edge initial feature
        if self.mconfig.USE_SPATIAL:
            tmp = descriptor[:,3:].clone()
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            node_feats = torch.cat([node_feats, tmp],dim=-1)
        
        edge_feats = edge_feats_initialization(node_feats.clone(), edge_indices)
        edge_feats = self.edge_mlp(edge_feats)
        with torch.no_grad():
            x_i = descriptor[edge_indices[0]]
            x_j = descriptor[edge_indices[1]]
            edge_feats_des = torch.zeros_like(x_i)
            edge_feats_des[:,0:3] = x_i[:,0:3]-x_j[:,0:3]
            # std  offset
            edge_feats_des[:,3:6] = x_i[:,3:6]-x_j[:,3:6]
            # dim log ratio
            edge_feats_des[:,6:9] = torch.log(x_i[:,6:9] / x_j[:,6:9])
            # volume log ratio
            edge_feats_des[:,9] = torch.log( x_i[:,9] / x_j[:,9])
            # length log ratio
            edge_feats_des[:,10] = torch.log( x_i[:,10] / x_j[:,10])

        edge_feats = torch.cat((edge_feats, edge_feats_des), dim=-1)
        
        node_feats_ = node_feats.clone()
        edge_feats_ = edge_feats.clone()
        # EdgeGCN
        node_feats_gcn, edge_feats_gcn = self.edge_gcn(node_feats, edge_feats, edge_indices)
        # Add residual cat
        node_feats = self.obj_mlp(torch.cat([node_feats_, node_feats_gcn.squeeze(0)], dim=-1))
        edge_feats = self.rel_mlp(torch.cat([edge_feats_, edge_feats_gcn.squeeze(0)], dim=-1))
        
        obj_logits = self.obj_classifier(node_feats)
        rel_logits = self.rel_classifier(edge_feats)
        
        return obj_logits, rel_logits
    
    def process_train(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration += 1 

        obj_logits_3d, rel_cls_3d = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=True)
        loss_obj_3d = F.cross_entropy(obj_logits_3d, gt_cls)
        
        batch_mean = torch.sum(gt_rel_cls, dim=(0))
        zeros = (gt_rel_cls.sum(-1) ==0).sum().unsqueeze(0)
        batch_mean = torch.cat([zeros,batch_mean],dim=0)
        weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
            
        weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
        weight = weight[1:]
        loss_rel_3d = F.binary_cross_entropy(rel_cls_3d, gt_rel_cls, weight=weight)

        loss = 0.1 * loss_obj_3d + 3 * loss_rel_3d
        self.backward(loss)

        top_k_obj = evaluate_topk_object(obj_logits_3d.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_cls_3d.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]        
        
        log = [("train/rel_loss", loss_rel_3d.detach().item()),
                ("train/obj_loss", loss_obj_3d.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
            ]
        return log
    
    def process_val(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, use_triplet=False):
        obj_logits_3d, rel_cls_3d = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
         # compute metric
        top_k_obj = evaluate_topk_object(obj_logits_3d.detach().cpu(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_cls_3d.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_logits_3d.detach().cpu(), rel_cls_3d.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=False, obj_topk=top_k_obj)
        
        return top_k_obj, top_k_obj, top_k_rel, top_k_rel, top_k_triplet, top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores
  
    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()

