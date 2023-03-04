import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model_utils.network_util import (MLP, Aggre_Index, Gen_Index,
                                                build_mlp)
from src.model.transformer.attention import MultiHeadAttention


class GraphEdgeAttenNetwork(torch.nn.Module):
    def __init__(self, num_heads, dim_node, dim_edge, dim_atten, aggr= 'max', use_bn=False,
                 flow='target_to_source',attention = 'fat',use_edge:bool=True, **kwargs):
        super().__init__() #  "Max" aggregation.
        self.name = 'edgeatten'
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.index_get = Gen_Index(flow=flow)
        if attention == 'fat':        
            self.index_aggr = Aggre_Index(aggr=aggr,flow=flow)
        elif attention == 'distance':
            aggr = 'add'
            self.index_aggr = Aggre_Index(aggr=aggr,flow=flow)
        else:
            raise NotImplementedError()

        self.edgeatten = MultiHeadedEdgeAttention(
            dim_node=dim_node,dim_edge=dim_edge,dim_atten=dim_atten,
            num_heads=num_heads,use_bn=use_bn,attention=attention,use_edge=use_edge, **kwargs)
        self.prop = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                            do_bn= use_bn, on_last=False)

    def forward(self, x, edge_feature, edge_index, weight=None, istrain=False):
        assert x.ndim == 2
        assert edge_feature.ndim == 2
        x_i, x_j = self.index_get(x, edge_index)
        xx, gcn_edge_feature, prob = self.edgeatten(x_i, edge_feature, x_j, weight, istrain=istrain)
        xx = self.index_aggr(xx, edge_index, dim_size = x.shape[0])
        xx = self.prop(torch.cat([x,xx],dim=1))
        return xx, gcn_edge_feature
  

class MultiHeadedEdgeAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_node: int, dim_edge: int, dim_atten: int, use_bn=False,
                 attention = 'fat', use_edge:bool = True, **kwargs):
        super().__init__()
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.name = 'MultiHeadedEdgeAttention'
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.d_n = d_n = dim_node // num_heads
        self.d_e = d_e = dim_edge // num_heads
        self.d_o = d_o = dim_atten // num_heads
        self.num_heads = num_heads
        self.use_edge = use_edge
        self.nn_edge = build_mlp([dim_node*2+dim_edge,(dim_node+dim_edge),dim_edge],
                          do_bn= use_bn, on_last=False)
        self.mask_obj = 0.5
        
        DROP_OUT_ATTEN = None
        if 'DROP_OUT_ATTEN' in kwargs:
            DROP_OUT_ATTEN = kwargs['DROP_OUT_ATTEN']
            # print('drop out in',self.name,'with value',DROP_OUT_ATTEN)
        
        self.attention = attention
        assert self.attention in ['fat']
        
        if self.attention == 'fat':
            if use_edge:
                self.nn = MLP([d_n+d_e, d_n+d_e, d_o],do_bn=use_bn,drop_out = DROP_OUT_ATTEN)
            else:
                self.nn = MLP([d_n, d_n*2, d_o],do_bn=use_bn,drop_out = DROP_OUT_ATTEN)
                
            self.proj_edge  = build_mlp([dim_edge,dim_edge])
            self.proj_query = build_mlp([dim_node,dim_node])
            self.proj_value = build_mlp([dim_node,dim_atten])
        elif self.attention == 'distance':
            self.proj_value = build_mlp([dim_node,dim_atten])

        
    def forward(self, query, edge, value, weight=None, istrain=False):
        batch_dim = query.size(0)
        
        edge_feature = torch.cat([query, edge, value],dim=1)
        # avoid overfitting by mask relation input object feature
        # if random.random() < self.mask_obj and istrain: 
        #     feat_mask = torch.cat([torch.ones_like(query),torch.zeros_like(edge), torch.ones_like(value)],dim=1)
        #     edge_feature = torch.where(feat_mask == 1, edge_feature, torch.zeros_like(edge_feature))
        
        edge_feature = self.nn_edge( edge_feature )#.view(b, -1, 1)

        if self.attention == 'fat':
            value = self.proj_value(value)
            query = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads)
            edge = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)
            if self.use_edge:
                prob = self.nn(torch.cat([query,edge],dim=1)) # b, dim, head    
            else:
                prob = self.nn(query) # b, dim, head 
            prob = prob.softmax(1)
            x = torch.einsum('bm,bm->bm', prob.reshape_as(value), value)
        
        elif self.attention == 'distance':
            raise NotImplementedError()
        
        else:
            raise NotImplementedError('')
        
        return x, edge_feature, prob
    
    
class MMG(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr= 'max', 
                 use_bn=False,flow='target_to_source', attention = 'fat', 
                 hidden_size=512, depth=1, use_edge:bool=True, **kwargs,
                 ):
        
        super().__init__()

        self.num_heads = num_heads
        self.depth = depth

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads) for i in range(depth))

        self.cross_attn = nn.ModuleList(
            MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads) for i in range(depth))

        self.cross_attn_rel = nn.ModuleList(
            MultiHeadAttention(d_model=dim_edge, d_k=dim_edge // num_heads, d_v=dim_edge // num_heads, h=num_heads) for i in range(depth))
        
        self.gcn_2ds = torch.nn.ModuleList()
        self.gcn_3ds = torch.nn.ModuleList()
        
        for _ in range(self.depth):

            self.gcn_2ds.append(GraphEdgeAttenNetwork(
                            num_heads,
                            dim_node,
                            dim_edge,
                            dim_atten,
                            aggr,
                            use_bn=use_bn,
                            flow=flow,
                            attention=attention,
                            use_edge=use_edge, 
                            **kwargs))
            
            self.gcn_3ds.append(GraphEdgeAttenNetwork(
                            num_heads,
                            dim_node,
                            dim_edge,
                            dim_atten,
                            aggr,
                            use_bn=use_bn,
                            flow=flow,
                            attention=attention,
                            use_edge=use_edge, 
                            **kwargs))
           
        self.self_attn_fc = nn.Sequential(  # 11 32 32 4(head)
            nn.Linear(4, 32),  # xyz, dist
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_heads)
        )
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
    
    
    def forward(self, obj_feature_3d, obj_feature_2d, edge_feature_3d, edge_feature_2d, edge_index, batch_ids, obj_center=None, discriptor=None, istrain=False):

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


        for i in range(self.depth):

            obj_feature_3d = obj_feature_3d.unsqueeze(0)
            obj_feature_2d = obj_feature_2d.unsqueeze(0)
            
            obj_feature_3d = self.self_attn[i](obj_feature_3d, obj_feature_3d, obj_feature_3d, attention_weights=obj_distance_weight, way=attention_matrix_way, attention_mask=obj_mask, use_knn=False)
            obj_feature_2d = self.cross_attn[i](obj_feature_2d, obj_feature_3d, obj_feature_3d, attention_weights=obj_distance_weight, way=attention_matrix_way, attention_mask=obj_mask, use_knn=False)
            
            obj_feature_3d = obj_feature_3d.squeeze(0)
            obj_feature_2d = obj_feature_2d.squeeze(0)  


            obj_feature_3d, edge_feature_3d = self.gcn_3ds[i](obj_feature_3d, edge_feature_3d, edge_index, istrain=istrain)
            obj_feature_2d, edge_feature_2d = self.gcn_2ds[i](obj_feature_2d, edge_feature_2d, edge_index, istrain=istrain)

            
            edge_feature_2d = edge_feature_2d.unsqueeze(0)
            edge_feature_3d = edge_feature_3d.unsqueeze(0)
            
            edge_feature_2d = self.cross_attn_rel[i](edge_feature_2d, edge_feature_3d, edge_feature_3d, use_knn=False)
            
            edge_feature_2d = edge_feature_2d.squeeze(0)
            edge_feature_3d = edge_feature_3d.squeeze(0)

            if i < (self.depth-1) or self.depth==1:
                
                obj_feature_3d = F.relu(obj_feature_3d)
                obj_feature_3d = self.drop_out(obj_feature_3d)
                
                obj_feature_2d = F.relu(obj_feature_2d)
                obj_feature_2d = self.drop_out(obj_feature_2d)

                edge_feature_3d = F.relu(edge_feature_3d)
                edge_feature_3d = self.drop_out(edge_feature_3d)

                edge_feature_2d = F.relu(edge_feature_2d)
                edge_feature_2d = self.drop_out(edge_feature_2d)
        
        return obj_feature_3d, obj_feature_2d, edge_feature_3d, edge_feature_2d


class MMG_single(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr= 'max', 
                 use_bn=False,flow='target_to_source', attention = 'fat', 
                 hidden_size=512, depth=1, use_edge:bool=True, **kwargs,
                 ):
        
        super().__init__()

        self.num_heads = num_heads
        self.depth = depth

        self.gcn_3ds = torch.nn.ModuleList()
        
        for _ in range(self.depth):

            self.gcn_3ds.append(GraphEdgeAttenNetwork(
                            num_heads,
                            dim_node,
                            dim_edge,
                            dim_atten,
                            aggr,
                            use_bn=use_bn,
                            flow=flow,
                            attention=attention,
                            use_edge=use_edge, 
                            **kwargs))
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
    
    def forward(self, obj_feature_3d, edge_feature_3d, edge_index, batch_ids, obj_center=None, istrain=False):

        for i in range(self.depth):
            obj_feature_3d, edge_feature_3d = self.gcn_3ds[i](obj_feature_3d, edge_feature_3d, edge_index, istrain=istrain)
            if i < (self.depth-1) or self.depth==1:
                
                obj_feature_3d = F.relu(obj_feature_3d)
                obj_feature_3d = self.drop_out(obj_feature_3d)
                
                edge_feature_3d = F.relu(edge_feature_3d)
                edge_feature_3d = self.drop_out(edge_feature_3d)
        
        return obj_feature_3d, edge_feature_3d

 
class MMG_teacher(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr= 'max', 
                 use_bn=False,flow='target_to_source', attention = 'fat', 
                 hidden_size=512, depth=1, use_edge:bool=True, **kwargs,
                 ):
        
        super().__init__()

        self.num_heads = num_heads
        self.depth = depth

        self.self_attn_3d = MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads)

        self.cross_attn_3d = MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads)
        
        self.self_attn_2d = MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads)

        self.cross_attn_2d = MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads)

        self.fusion_module = nn.Sequential(
            nn.Linear(512 * 4, 512 * 2),
            nn.ReLU(),
            nn.BatchNorm1d(512 * 2),
            nn.Dropout(0.5),
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )

        self.gcns = torch.nn.ModuleList()   
        for _ in range(self.depth):
            
            self.gcns.append(GraphEdgeAttenNetwork(
                            num_heads,
                            dim_node,
                            dim_edge,
                            dim_atten,
                            aggr,
                            use_bn=use_bn,
                            flow=flow,
                            attention=attention,
                            use_edge=use_edge, 
                            **kwargs))
           
        self.self_attn_fc = nn.Sequential(  # 4 32 32 4(head)
            nn.Linear(4, 32),  # xyz, dist
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_heads)
        )
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
    
    def forward(self, obj_feature_3d, obj_feature_2d, edge_feature, edge_index, batch_ids, obj_center=None, istrain=False):

        if obj_center is not None:
            # get attention weight
            batch_size = batch_ids.max().item() + 1
            N_K = obj_feature_3d.shape[0]
            mask = torch.zeros(1, 1, N_K, N_K).cuda()
            distance = torch.zeros(1, self.num_heads, N_K, N_K).cuda()
            count = 0

            for i in range(batch_size):

                idx_i = torch.where(batch_ids == i)[0]
                mask[:, :, count:count + len(idx_i), count:count + len(idx_i)] = 1
            
                center_A = obj_center[None, idx_i, :].clone().detach().repeat(len(idx_i), 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, len(idx_i), 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
                weights = torch.cat([center_dist, dist], dim=-1).unsqueeze(0)  # 1 N N 4
                dist_weights = self.self_attn_fc(weights).permute(0,3,1,2)  # 1 num_heads N N
                
                attention_matrix_way = 'add'
                distance[:, :, count:count + len(idx_i), count:count + len(idx_i)] = dist_weights

                count += len(idx_i)
        else:
            mask = None
            distance = None
            attention_matrix_way = 'mul'
         
        obj_feature_3d = obj_feature_3d.unsqueeze(0)
        obj_feature_2d = obj_feature_2d.unsqueeze(0)
        
        obj_feature_3d_sa = self.self_attn_3d(obj_feature_3d, obj_feature_3d, obj_feature_3d, attention_weights=distance, way=attention_matrix_way, attention_mask=mask)
        obj_feature_2d_sa = self.self_attn_2d(obj_feature_2d, obj_feature_2d, obj_feature_2d, attention_weights=distance, way=attention_matrix_way, attention_mask=mask)
        obj_feature_3d_ca = self.cross_attn_3d(obj_feature_3d_sa, obj_feature_2d_sa, obj_feature_2d_sa, attention_weights=distance, way=attention_matrix_way, attention_mask=mask)
        obj_feature_2d_ca = self.cross_attn_2d(obj_feature_2d_sa, obj_feature_3d_sa, obj_feature_3d_sa, attention_weights=distance, way=attention_matrix_way, attention_mask=mask)
        
        obj_feature_3d_sa = obj_feature_3d_sa.squeeze(0)
        obj_feature_2d_sa = obj_feature_2d_sa.squeeze(0)
        obj_feature_3d_ca = obj_feature_3d_ca.squeeze(0)
        obj_feature_2d_ca = obj_feature_2d_ca.squeeze(0)
        
        # fusion 3d and 2d
        obj_feature = self.fusion_module(torch.cat([obj_feature_3d_sa, obj_feature_2d_sa, obj_feature_3d_ca, obj_feature_2d_ca], dim=-1))
        obj_feature_mimic = obj_feature.clone().detach()
        
        for i in range(self.depth):

            obj_feature, edge_feature = self.gcns[i](obj_feature, edge_feature, edge_index, istrain=istrain)

            if i < (self.depth-1) or self.depth==1:
                
                obj_feature = F.relu(obj_feature)
                obj_feature = self.drop_out(obj_feature)
                
                edge_feature = F.relu(edge_feature)
                edge_feature = self.drop_out(edge_feature)
        
        return obj_feature, edge_feature, obj_feature_mimic


class MMG_student(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr= 'max', 
                 use_bn=False,flow='target_to_source', attention = 'fat', 
                 hidden_size=512, depth=1, use_edge:bool=True, **kwargs,
                 ):
        
        super().__init__()

        self.num_heads = num_heads
        self.depth = depth

        self.self_attn_before = MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads)
        self.self_attn_after = MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads)
        self.gcns = torch.nn.ModuleList()
        
        for _ in range(self.depth):
            
            self.gcns.append(GraphEdgeAttenNetwork(
                            num_heads,
                            dim_node,
                            dim_edge,
                            dim_atten,
                            aggr,
                            use_bn=use_bn,
                            flow=flow,
                            attention=attention,
                            use_edge=use_edge, 
                            **kwargs))
           
        self.self_attn_fc = nn.Sequential(  # 4 32 32 4(head)
            nn.Linear(4, 32),  # xyz, dist
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_heads)
        )

        self.modality_learner = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512)
        )

        self.mlp = nn.Linear(512 * 2, 512)
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
    
    def forward(self, obj_feature, edge_feature, edge_index, batch_ids, obj_center=None, istrain=False):
        
        if obj_center is not None:
            # get attention weight
            batch_size = batch_ids.max().item() + 1
            N_K = obj_feature.shape[0]
            mask = torch.zeros(1, 1, N_K, N_K).cuda()
            distance = torch.zeros(1, self.num_heads, N_K, N_K).cuda()
            count = 0

            for i in range(batch_size):

                idx_i = torch.where(batch_ids == i)[0]
                mask[:, :, count:count + len(idx_i), count:count + len(idx_i)] = 1
            
                center_A = obj_center[None, idx_i, :].clone().detach().repeat(len(idx_i), 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, len(idx_i), 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
                weights = torch.cat([center_dist, dist], dim=-1).unsqueeze(0)  # 1 N N 4
                dist_weights = self.self_attn_fc(weights).permute(0,3,1,2)  # 1 num_heads N N
                
                attention_matrix_way = 'add'
                distance[:, :, count:count + len(idx_i), count:count + len(idx_i)] = dist_weights

                count += len(idx_i)
        else:
            mask = None
            distance = None
            attention_matrix_way = 'mul'
 
        
        obj_feature = obj_feature.unsqueeze(0)
        obj_feature = self.self_attn_before(obj_feature, obj_feature, obj_feature, attention_weights=distance, way=attention_matrix_way, attention_mask=mask)
        obj_feature = obj_feature.squeeze(0)

        #obj_feature_tmp = obj_feature.clone()
        #obj_feature_mimic = self.modality_learner(obj_feature)
        obj_feature_mimic = obj_feature.clone()
        # obj_feature_tmp = torch.cat([obj_feature_tmp, obj_feature_mimic.detach()], dim=-1)
        # obj_feature = self.mlp(obj_feature_tmp)
        
        obj_feature = obj_feature.unsqueeze(0)
        obj_feature = self.self_attn_after(obj_feature, obj_feature, obj_feature, attention_weights=distance, way=attention_matrix_way, attention_mask=mask)
        obj_feature = obj_feature.squeeze(0)
        
        for i in range(self.depth):

            obj_feature, edge_feature = self.gcns[i](obj_feature, edge_feature, edge_index, istrain=istrain)

            if i < (self.depth-1) or self.depth==1:
                
                obj_feature = F.relu(obj_feature)
                obj_feature = self.drop_out(obj_feature)
                
                edge_feature = F.relu(edge_feature)
                edge_feature = self.drop_out(edge_feature)
        
        return obj_feature, edge_feature, obj_feature_mimic
