import torch
import clip
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model.model_utils.model_base import BaseModel
from utils import op_utils
from src.utils.eva_utils_acc import get_gt, evaluate_topk_object, evaluate_topk_predicate, evaluate_triplet_topk
from src.model.model_utils.network_MMG import MMG_single
from src.model.model_utils.network_PointNet import PointNetfeat, PointNetRelCls, PointNetRelClsMulti
from clip_adapter.model import AdapterModel

class Mmgnet(BaseModel):
    def __init__(self, config, num_obj_class, num_rel_class, dim_descriptor=11):
        '''
        3d cat location, 2d
        '''
        
        super().__init__('Mmgnet2', config)

        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN

        dim_point = 3
        if not mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        
        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial

        self.dim_point=dim_point
        self.dim_edge=dim_point_rel
        self.num_class=num_obj_class
        self.num_rel=num_rel_class
        self.flow = 'target_to_source'
        self.clip_feat_dim = self.config.MODEL.clip_feat_dim
        dim_point_feature = 768
        self.momentum = 0.1
        self.model_pre = None
        
        # Object Encoder
        self.obj_encoder = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)      
           
        self.rel_encoder_3d = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=512)
        
        self.mmg = MMG_single(
            dim_node=512,
            dim_edge=512,
            dim_atten=self.mconfig.DIM_ATTEN,
            depth=self.mconfig.N_LAYERS, 
            num_heads=self.mconfig.NUM_HEADS,
            aggr=self.mconfig.GCN_AGGR,
            flow=self.flow,
            attention=self.mconfig.ATTENTION,
            use_edge=self.mconfig.USE_GCN_EDGE,
            DROP_OUT_ATTEN=self.mconfig.DROP_OUT_ATTEN)

        self.triplet_projector_3d = torch.nn.Sequential(
            torch.nn.Linear(512 * 3, 512 * 2),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512 * 2, 512)
        )
        
        # object adapter
        self.clip_adapter = AdapterModel(input_size=512, output_size=512, alpha=0.5)
        self.obj_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.mlp_3d = torch.nn.Sequential(
            torch.nn.Linear(512 + 256, 512 - 8),
            torch.nn.BatchNorm1d(512 - 8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        
        if mconfig.multi_rel_outputs:
            self.rel_predictor_3d = PointNetRelClsMulti(
                num_rel_class, 
                in_size=512, 
                batch_norm=with_bn,drop_out=True)
        else:
            self.rel_predictor_3d = PointNetRelCls(
                num_rel_class, 
                in_size=512, 
                batch_norm=with_bn,drop_out=True)
            
        self.init_weight(obj_label_path=self.mconfig.obj_label_path, \
                         rel_label_path=self.mconfig.rel_label_path, \
                         adapter_path=self.mconfig.adapter_path)
        
        mmg_obj, mmg_rel = [], []
        for name, para in self.mmg.named_parameters():
            if 'nn_edge' in name:
                mmg_rel.append(para)
            else:
                mmg_obj.append(para)
        
        self.optimizer = optim.AdamW([
            {'params':self.obj_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder_3d.parameters() , 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':mmg_obj, 'lr':float(config.LR) / 4, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':mmg_rel, 'lr':float(config.LR) / 2, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_predictor_3d.parameters(), 'lr':float(config.LR) / 10, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_predictor_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.mlp_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.triplet_projector_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_logit_scale, 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()

    def init_weight(self, obj_label_path, rel_label_path, adapter_path):
        torch.nn.init.xavier_uniform_(self.mlp_3d[0].weight)
        torch.nn.init.xavier_uniform_(self.triplet_projector_3d[0].weight)
        torch.nn.init.xavier_uniform_(self.triplet_projector_3d[-1].weight)
        obj_text_features, rel_text_feature = self.get_label_weight(obj_label_path, rel_label_path)
        # node feature classifier        
        self.obj_predictor_3d = torch.nn.Linear(self.mconfig.clip_feat_dim, self.num_class)
        self.obj_predictor_3d.weight.data.copy_(obj_text_features)
        for param in self.obj_predictor_3d.parameters():
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

    def cosine_loss(self, A, B, t=1):
        return torch.clamp(t - F.cosine_similarity(A, B, dim=-1), min=0).mean()
    
    def generate_object_pair_features(self, obj_feats, edges_feats, edge_indice):
        obj_pair_feats = []
        for (edge_feat, edge_index) in zip(edges_feats, edge_indice.t()):
            obj_pair_feats.append(torch.cat([obj_feats[edge_index[0]], obj_feats[edge_index[1]], edge_feat], dim=-1))
        obj_pair_feats = torch.vstack(obj_pair_feats)
        return obj_pair_feats
    
    def compute_triplet_loss(self, obj_logits_3d, rel_cls_3d, obj_logits_2d, rel_cls_2d, edge_indices):
        triplet_loss = []
        obj_logits_3d_softmax = F.softmax(obj_logits_3d, dim=-1)
        obj_logits_2d_softmax = F.softmax(obj_logits_2d, dim=-1)
        for idx, i in enumerate(edge_indices):
            obj_score_3d = obj_logits_3d_softmax[i[0]]
            obj_score_2d = obj_logits_2d_softmax[i[0]]
            sub_score_3d = obj_logits_3d_softmax[i[1]]
            sub_score_2d = obj_logits_2d_softmax[i[1]]
            rel_score_3d = rel_cls_3d[idx]
            rel_score_2d = rel_cls_2d[idx]
            node_score_3d = torch.einsum('n,m->nm', obj_score_3d, sub_score_3d)
            node_score_2d = torch.einsum('n,m->nm', obj_score_2d, sub_score_2d)
            triplet_score_3d = torch.einsum('nl,m->nlm', node_score_3d, rel_score_3d).reshape(-1)
            triplet_score_2d = torch.einsum('nl,m->nlm', node_score_2d, rel_score_2d).reshape(-1)
            triplet_loss.append(F.l1_loss(triplet_score_3d, triplet_score_2d.detach(), reduction='sum')) 
            
            # triplet_logits_3d_kl = F.softmax(triplet_score_3d.clone(), dim=-1)
            # triplet_logits_2d_kl = F.softmax(triplet_score_2d.clone().detach(), dim=-1)
            # triplet_loss.append(F.kl_div(triplet_logits_3d_kl.log(), triplet_logits_2d_kl, reduction='sum'))
            
        #return torch.sum(torch.tensor(triplet_loss))
        return torch.mean(torch.tensor(triplet_loss))
    
    def forward(self, obj_points, edge_indices, descriptor=None, batch_ids=None, istrain=False):

        obj_feature = self.obj_encoder(obj_points)
        obj_feature = self.mlp_3d(obj_feature)

        if self.mconfig.USE_SPATIAL:
            tmp = descriptor[:,3:].clone()
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            obj_feature = torch.cat([obj_feature, tmp],dim=-1)
        
        ''' Create edge feature '''
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)
        
        rel_feature_3d = self.rel_encoder_3d(edge_feature)

        obj_center = descriptor[:, :3].clone()
        #obj_center = torch.randn_like(obj_feature[:, :3]).cuda()
        gcn_obj_feature_3d, gcn_edge_feature_3d \
            = self.mmg(obj_feature, rel_feature_3d, edge_indices, batch_ids, obj_center, istrain=istrain)

        gcn_edge_feature_3d_dis = self.generate_object_pair_features(gcn_obj_feature_3d, gcn_edge_feature_3d, edge_indices)
        
        gcn_edge_feature_3d_dis = self.triplet_projector_3d(gcn_edge_feature_3d_dis)

        rel_cls_3d = self.rel_predictor_3d(gcn_edge_feature_3d)
        
        #if istrain:
        logit_scale = self.obj_logit_scale.exp()
        #else:
        #    logit_scale = 63.74

        obj_logits_3d = logit_scale * self.obj_predictor_3d(gcn_obj_feature_3d / gcn_obj_feature_3d.norm(dim=-1, keepdim=True))

        if istrain:
            return obj_logits_3d, rel_cls_3d, gcn_edge_feature_3d_dis, logit_scale
        else:
            return obj_logits_3d, rel_cls_3d

    def process_train(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration +=1    
        obj_logits_3d, rel_cls_3d, edge_feature_3d, obj_logit_scale = self(obj_points, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=True)
        
        # compute loss for obj
        # obj_logits_3d_kl = F.softmax(obj_logits_3d.clone(), dim=-1)
        # obj_logits_2d_kl = F.softmax(obj_logits_2d.clone().detach(), dim=-1)
        # loss_obj_KL_3d = F.kl_div(obj_logits_3d_kl.log(), obj_logits_2d_kl, reduction='batchmean')
        
        # compute loss for obj
        loss_obj_3d = F.cross_entropy(obj_logits_3d, gt_cls)

         # compute loss for rel
        if self.mconfig.multi_rel_outputs:
            if self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                batch_mean = torch.sum(gt_rel_cls, dim=(0))
                zeros = (gt_rel_cls.sum(-1) ==0).sum().unsqueeze(0)
                batch_mean = torch.cat([zeros,batch_mean],dim=0)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
                if ignore_none_rel:
                    weight[0] = 0
                    weight *= 1e-2 # reduce the weight from ScanNet
                    # print('set weight of none to 0')
                if 'NONE_RATIO' in self.mconfig:
                    weight[0] *= self.mconfig.NONE_RATIO
                    
                weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
                weight = weight[1:]                
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            elif self.mconfig.WEIGHT_EDGE == 'NONE':
                weight = None
            else:
                raise NotImplementedError("unknown weight_edge type")

            # rel_logits_3d_kl = F.softmax(rel_cls_3d.clone().detach(), dim=-1)
            # rel_logits_2d_kl = F.softmax(rel_cls_2d.clone(), dim=-1)
            # loss_rel_KL_2d = F.kl_div(rel_logits_2d_kl.log(), rel_logits_3d_kl, reduction='sum')

            loss_rel_3d = F.binary_cross_entropy(rel_cls_3d, gt_rel_cls, weight=weight)
        else:
            if self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                one_hot_gt_rel = torch.nn.functional.one_hot(gt_rel_cls,num_classes = self.num_rel)
                batch_mean = torch.sum(one_hot_gt_rel, dim=(0), dtype=torch.float)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf
                if ignore_none_rel: 
                    weight[0] = 0 # assume none is the first relationship
                    weight *= 1e-2 # reduce the weight from ScanNet
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            elif self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'NONE':
                weight = None
            else:
                raise NotImplementedError("unknown weight_edge type")

            if 'ignore_entirely' in self.mconfig and (self.mconfig.ignore_entirely and ignore_none_rel):
                loss_rel_3d = torch.zeros(1, device=rel_cls_3d.device, requires_grad=False)
            else:
                loss_rel_3d = F.nll_loss(rel_cls_3d, gt_rel_cls, weight = weight)

        lambda_r = 1.0
        lambda_o = self.mconfig.lambda_o
        lambda_max = max(lambda_r,lambda_o)
        lambda_r /= lambda_max
        lambda_o /= lambda_max

        # compute similarity between visual with text
        rel_text_feat = self.get_rel_emb(gt_cls, gt_rel_cls, edge_indices)

        edge_feature_3d = edge_feature_3d / edge_feature_3d.norm(dim=-1, keepdim=True)
        rel_mimic_3d = F.l1_loss(edge_feature_3d, rel_text_feat)
        #rel_mimic_3d = self.cosine_loss(edge_feature_3d, rel_text_feat, t=0.9)

        # compute triplet loss
        # triplet_loss = self.compute_triplet_loss(obj_logits_3d, rel_cls_3d, obj_logits_2d, rel_cls_2d, edge_indices)
               
        loss = lambda_o * (loss_obj_3d) + 3 * lambda_r * (loss_rel_3d) + 0.1 * (rel_mimic_3d)
        #loss = lambda_o * (loss_obj_2d + loss_obj_3d) + 3 * lambda_r * (loss_rel_2d + loss_rel_3d) + 0.1 * (loss_mimic + rel_mimic_2d)
        self.backward(loss)
        
        # compute 3d metric
        top_k_obj = evaluate_topk_object(obj_logits_3d.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_cls_3d.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        
        
        log = [("train/rel_loss", loss_rel_3d.detach().item()),
                ("train/obj_loss", loss_obj_3d.detach().item()),
                ("train/logit_scale", obj_logit_scale.detach().item()),
                #("train/loss_rel_KL_2d", loss_rel_KL_2d.detach().item()),
                ("train/rel_mimic_loss_3d", rel_mimic_3d.detach().item()),
                #("train/triplet_loss", triplet_loss.detach().item()),
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
 
        obj_logits_3d, rel_cls_3d = self(obj_points, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_logits_3d.detach().cpu(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_cls_3d.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        
        if use_triplet:
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_logits_3d.detach().cpu(), rel_cls_3d.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=True, obj_topk=top_k_obj)
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
