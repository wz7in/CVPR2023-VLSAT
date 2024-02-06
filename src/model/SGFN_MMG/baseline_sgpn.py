import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model.model_utils.model_base import BaseModel
from utils import op_utils
from src.utils.eva_utils_acc import get_gt, evaluate_topk_object, evaluate_topk_predicate, evaluate_triplet_topk
from src.model.model_utils.network_GNN import GraphEdgeAttenNetworkLayers
from src.model.model_utils.network_PointNet import PointNetfeat, PointNetCls, PointNetRelCls, PointNetRelClsMulti

class SGPN(BaseModel):
    """
    512 + 256 baseline
    """
    def __init__(self, config, num_obj_class, num_rel_class, dim_descriptor=11):
        super().__init__('SGPN', config)

        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN

        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        
        dim_point = 3
        dim_point_rel = 3
        if mconfig.USE_RGB:
            dim_point +=3
            dim_point_rel+=3
        if mconfig.USE_NORMAL:
            dim_point +=3
            dim_point_rel+=3
            
        if mconfig.USE_CONTEXT:
            dim_point_rel += 1

        dim_point_feature = 512
        
        # Object Encoder
        self.obj_encoder = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)      
        
        # Relationship Encoder
        self.rel_encoder = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.edge_feature_size)
        

        self.obj_predictor = PointNetCls(num_obj_class, in_size=dim_point_feature,
                                 batch_norm=with_bn, drop_out=True)

        if mconfig.multi_rel_outputs:
            self.rel_predictor = PointNetRelClsMulti(
                num_rel_class, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
        else:
            self.rel_predictor = PointNetRelCls(
                num_rel_class, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
        
        self.optimizer = optim.AdamW([
            {'params':self.obj_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            #{'params':self.mlp.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()


    def forward(self, obj_points, rel_points):

        obj_feature = self.obj_encoder(obj_points)
        
        rel_feature = self.rel_encoder(rel_points)

        rel_cls = self.rel_predictor(rel_feature)

        obj_logits = self.obj_predictor(obj_feature)

        return obj_logits, rel_cls

    def process_train(self, obj_points, gt_cls, rel_points, gt_rel_cls, edge_indices):
        self.iteration +=1    
        
        obj_pred, rel_pred = self(obj_points, rel_points)
        
        # compute loss for obj
        loss_obj = F.nll_loss(obj_pred, gt_cls)

         # compute loss for rel
        if self.mconfig.multi_rel_outputs:
            loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls)
        else:
            loss_rel = F.nll_loss(rel_pred, gt_rel_cls)

        
        loss = 0.1 * loss_obj + loss_rel
        self.backward(loss)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        

        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        
        
        log = [("train/rel_loss", loss_rel.detach().item()),
                ("train/obj_loss", loss_obj.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
            ]
        return log
           
    def process_val(self, obj_points, gt_cls, rel_points, gt_rel_cls, edge_indices, use_triplet=False):
 
        obj_pred, rel_pred = self(obj_points, rel_points)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach().cpu(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        if use_triplet:
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=False, obj_topk=top_k_obj)
        else:
            top_k_triplet = [101]
            cls_matrix = None
            sub_scores = None
            obj_scores = None
            rel_scores = None

        return top_k_obj, top_k_obj, top_k_rel, top_k_rel, top_k_triplet, top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores 
    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()
