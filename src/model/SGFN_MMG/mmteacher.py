import torch
import clip
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model.model_utils.model_base import BaseModel
from utils import op_utils
from src.utils.eva_utils_acc import get_gt, evaluate_topk_object, evaluate_topk_predicate, evaluate_triplet_topk
from src.model.model_utils.network_MMG import MMG_student, MMG_teacher
from src.model.model_utils.network_PointNet import PointNetfeat, PointNetRelCls, PointNetRelClsMulti
from clip_adapter.model import AdapterModel

class MMteacher(BaseModel):
    def __init__(self, config, num_obj_class, num_rel_class, dim_descriptor=11):
        '''
        multi modality teacher
        '''
        
        super().__init__('MMteacher', config)

        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN

        dim_point = 3
        if mconfig.USE_RGB:
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
        self.obj_encoder_student = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=768) 
        
        self.mlp_student = torch.nn.Sequential(
            torch.nn.Linear(512 + 256, 512 - 8),
            torch.nn.BatchNorm1d(512 - 8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )

        self.obj_encoder_teacher = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=512 - 8)  
        
        self.triplet_projector_teacher = torch.nn.Sequential(
            torch.nn.Linear(512 * 3, 512 * 2),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512 * 2, 512)
        )

        self.triplet_projector_student = torch.nn.Sequential(
            torch.nn.Linear(512 * 3, 512 * 2),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512 * 2, 512)
        )
        
        # Relationship Encoder
        self.rel_encoder_student = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=512)
        
        self.rel_encoder_teacher = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=512)
        
        self.mmg_teacher = MMG_teacher(
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

        self.mmg_student = MMG_student(
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

        # object adapter
        self.clip_adapter = AdapterModel(input_size=512, output_size=512, alpha=0.5)
        self.obj_teacher_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_student_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        if mconfig.multi_rel_outputs:
            self.rel_predictor_teacher = PointNetRelClsMulti(
                num_rel_class, 
                in_size=512, 
                batch_norm=with_bn,drop_out=True)
            self.rel_predictor_student = PointNetRelClsMulti(
                num_rel_class, 
                in_size=512, 
                batch_norm=with_bn,drop_out=True)
        else:
            self.rel_predictor_teacher = PointNetRelCls(
                num_rel_class, 
                in_size=512, 
                batch_norm=with_bn,drop_out=True)
            self.rel_predictor_student = PointNetRelCls(
                num_rel_class, 
                in_size=512, 
                batch_norm=with_bn,drop_out=True)
            
        self.init_weight(obj_label_path=self.mconfig.obj_label_path, \
                         rel_label_path=self.mconfig.rel_label_path, \
                         adapter_path=self.mconfig.adapter_path)
        
        mmg_obj_teacher, mmg_rel_teacher = [], []
        for name, para in self.mmg_teacher.named_parameters():
            if 'nn_edge' in name:
                mmg_rel_teacher.append(para)
            else:
                mmg_obj_teacher.append(para)
        
        mmg_obj_student, mmg_rel_student = [], []
        for name, para in self.mmg_student.named_parameters():
            if 'nn_edge' in name:
                mmg_rel_student.append(para)
            else:
                mmg_obj_student.append(para)
        
        self.optimizer = optim.AdamW([
            {'params':self.obj_encoder_teacher.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_encoder_student.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder_student.parameters() , 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder_teacher.parameters() , 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':mmg_obj_teacher, 'lr':float(config.LR) / 4, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':mmg_rel_teacher, 'lr':float(config.LR) / 2, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':mmg_obj_student, 'lr':float(config.LR) / 4, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':mmg_rel_student, 'lr':float(config.LR) / 2, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_predictor_student.parameters(), 'lr':float(config.LR) / 10, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_predictor_student.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_predictor_teacher.parameters(), 'lr':float(config.LR) / 10, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_predictor_teacher.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.triplet_projector_teacher.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.triplet_projector_student.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.mlp_student.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_teacher_logit_scale, 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_student_logit_scale, 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()

    def init_weight(self, obj_label_path, rel_label_path, adapter_path):
        torch.nn.init.xavier_uniform_(self.mlp_student[0].weight)
        torch.nn.init.xavier_uniform_(self.triplet_projector_teacher[0].weight)
        torch.nn.init.xavier_uniform_(self.triplet_projector_teacher[-1].weight)
        torch.nn.init.xavier_uniform_(self.triplet_projector_student[0].weight)
        torch.nn.init.xavier_uniform_(self.triplet_projector_student[-1].weight)
        obj_text_features, rel_text_feature = self.get_label_weight(obj_label_path, rel_label_path)
        # node feature classifier        
        self.obj_predictor_teacher = torch.nn.Linear(self.mconfig.clip_feat_dim, self.num_class)
        self.obj_predictor_teacher.weight.data.copy_(obj_text_features)
        for param in self.obj_predictor_teacher.parameters():
            param.requires_grad = True
        
        self.obj_predictor_student = torch.nn.Linear(self.mconfig.clip_feat_dim, self.num_class)
        self.obj_predictor_student.weight.data.copy_(obj_text_features)
        for param in self.obj_predictor_student.parameters():
            param.requires_grad = True

        self.clip_adapter.load_state_dict(torch.load(adapter_path, 'cpu'))
        # freeze clip adapter
        for param in self.clip_adapter.parameters():
            param.requires_grad = False
        
        self.obj_teacher_logit_scale.requires_grad = True
        self.obj_student_logit_scale.requires_grad = True
    
    def update_model_pre(self, new_model):
        self.model_pre = new_model
    
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

    def smooth_loss(self, pred, gold):
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss

    def cosine_loss(self, A, B, t=1):
        return (t - F.cosine_similarity(A, B, dim=-1)).mean()
    
    def generate_object_pair_features(self, obj_feats, edges_feats, edge_indice):
        obj_pair_feats = []
        for (edge_feat, edge_index) in zip(edges_feats, edge_indice.t()):
            obj_pair_feats.append(torch.cat([obj_feats[edge_index[0]], obj_feats[edge_index[1]], edge_feat], dim=-1))
        obj_pair_feats = torch.vstack(obj_pair_feats)
        return obj_pair_feats
 
    def forward(self, obj_points, obj_2d_feats, edge_indices, descriptor=None, batch_ids=None, istrain=False):

        obj_feature_teacher = self.obj_encoder_teacher(obj_points) 
        obj_feature_student = self.obj_encoder_student(obj_points)
        obj_feature_student_mimic_before = obj_feature_student[:, :512].clone()
        obj_feature_student = self.mlp_student(obj_feature_student)
        if self.mconfig.USE_SPATIAL:
            tmp = descriptor[:,3:].clone()
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            obj_feature_teacher = torch.cat([obj_feature_teacher, tmp.clone()],dim=-1)
            obj_feature_student = torch.cat([obj_feature_student, tmp.clone()],dim=-1)
        
        ''' Create edge feature '''
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)
        rel_feature_teacher = self.rel_encoder_teacher(edge_feature)
        rel_feature_student = self.rel_encoder_student(edge_feature)

        ''' Create 2d feature '''
        with torch.no_grad():
            obj_2d_feats = self.clip_adapter(obj_2d_feats)
        obj_2d_feats_mimic = obj_2d_feats.clone()

        ''' Create multi modalities teacher feature '''
        obj_center = descriptor[:, :3].clone()
        gcn_obj_feature_teacher, gcn_edge_feature_teacher, obj_feature_teacher_mimic \
            = self.mmg_teacher(obj_feature_teacher, obj_2d_feats, rel_feature_teacher, edge_indices, batch_ids, obj_center, istrain=istrain)
        
        ''' Create student feature '''
        gcn_obj_feature_student, gcn_edge_feature_student, obj_feature_student_mimic \
            = self.mmg_student(obj_feature_student, rel_feature_student, edge_indices, batch_ids, obj_center, istrain=istrain)

        gcn_edge_feature_teacher_dis = self.generate_object_pair_features(gcn_obj_feature_teacher, gcn_edge_feature_teacher, edge_indices)
        gcn_edge_feature_teacher_dis = self.triplet_projector_teacher(gcn_edge_feature_teacher_dis)
        gcn_edge_feature_student_dis = self.generate_object_pair_features(gcn_obj_feature_student, gcn_edge_feature_student, edge_indices)
        gcn_edge_feature_student_dis = self.triplet_projector_student(gcn_edge_feature_student_dis)

        
        rel_cls_teacher = self.rel_predictor_teacher(gcn_edge_feature_teacher)
        rel_cls_student = self.rel_predictor_student(gcn_edge_feature_student)
        
        logit_scale_teacher = self.obj_teacher_logit_scale.exp()
        logit_scale_student = self.obj_student_logit_scale.exp()

        obj_logits_teacher = logit_scale_teacher * self.obj_predictor_teacher(gcn_obj_feature_teacher / gcn_obj_feature_teacher.norm(dim=-1, keepdim=True))
        obj_logits_student = logit_scale_student * self.obj_predictor_student(gcn_obj_feature_student / gcn_obj_feature_student.norm(dim=-1, keepdim=True))

        if istrain:
            return obj_logits_teacher, obj_logits_student, rel_cls_teacher, rel_cls_student, obj_feature_teacher_mimic, \
                obj_feature_student_mimic, obj_feature_student_mimic_before, obj_2d_feats_mimic, gcn_edge_feature_teacher_dis, gcn_edge_feature_student_dis, logit_scale_teacher, logit_scale_student
        else:
            return obj_logits_teacher, obj_logits_student, rel_cls_teacher, rel_cls_student

    def process_train(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration +=1    
        
        obj_logits_teacher, obj_logits_student, rel_cls_teacher, rel_cls_student, obj_feature_teacher_mimic, obj_feature_student_mimic, obj_feature_student_mimic_before, obj_2d_feats_mimic, \
            edge_feature_teacher, edge_feature_student, logit_scale_teacher, logit_scale_student = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=True)
                
        # compute loss for obj
        loss_obj_teacher = F.cross_entropy(obj_logits_teacher, gt_cls)
        loss_obj_student = F.cross_entropy(obj_logits_student, gt_cls)

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

            # rel_logits_teacher_kl = F.softmax(rel_cls_teacher.clone(), dim=-1)
            # rel_logits_student_kl = F.softmax(rel_cls_student.clone().detach(), dim=-1)
            # rel_KL_loss = F.kl_div(rel_logits_teacher_kl.log(), rel_logits_student_kl, reduction='sum')

            loss_rel_teacher = F.binary_cross_entropy(rel_cls_teacher, gt_rel_cls, weight=weight)
            loss_rel_student = F.binary_cross_entropy(rel_cls_student, gt_rel_cls, weight=weight)
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
                loss_rel_2d = loss_rel_3d = torch.zeros(1, device=rel_cls_teacher.device, requires_grad=False)
            else:
                loss_rel_teacher = F.nll_loss(rel_cls_teacher, gt_rel_cls, weight = weight)
                loss_rel_student = F.nll_loss(rel_cls_student, gt_rel_cls, weight = weight)
        
        lambda_r = 1.0
        lambda_o = self.mconfig.lambda_o
        lambda_max = max(lambda_r,lambda_o)
        lambda_r /= lambda_max
        lambda_o /= lambda_max

        # loss_mimic = F.mse_loss(obj_feature_3d, obj_feature_2d, reduction='sum')
        # loss_mimic /= obj_feature_3d.shape[0]
        obj_feature_teacher_mimic = obj_feature_teacher_mimic / obj_feature_teacher_mimic.norm(dim=-1, keepdim=True)
        obj_feature_student_mimic = obj_feature_student_mimic / obj_feature_student_mimic.norm(dim=-1, keepdim=True)
        obj_feature_student_mimic_before = obj_feature_student_mimic_before / obj_feature_student_mimic_before.norm(dim=-1, keepdim=True)
        obj_feature_teacher_mimic_before = obj_2d_feats_mimic / obj_2d_feats_mimic.norm(dim=-1, keepdim=True)
        
        loss_mimic_before = self.cosine_loss(obj_feature_student_mimic_before, obj_feature_teacher_mimic_before, t=0.8)
        loss_mimic_after = self.cosine_loss(obj_feature_student_mimic, obj_feature_teacher_mimic, t=0.8)
        #loss_mimic = F.l1_loss(obj_feature_3d, obj_feature_2d)

        rel_text_feat = self.get_rel_emb(gt_cls, gt_rel_cls, edge_indices)
        
        edge_feature_teacher = edge_feature_teacher / edge_feature_teacher.norm(dim=-1, keepdim=True)
        rel_mimic_teacher = F.l1_loss(edge_feature_teacher, rel_text_feat)

        edge_feature_student = edge_feature_student / edge_feature_student.norm(dim=-1, keepdim=True)
        rel_mimic_student = F.l1_loss(edge_feature_student, rel_text_feat)
               
        loss = lambda_o * (loss_obj_student + loss_obj_teacher) + 3 * lambda_r * (loss_rel_student + loss_rel_teacher) + 0.1 * (loss_mimic_before + loss_mimic_after + rel_mimic_teacher + rel_mimic_student)
        #loss = lambda_o * (loss_obj_2d + loss_obj_3d) + 3 * lambda_r * (loss_rel_2d + loss_rel_3d) + 0.1 * (loss_mimic + obj_kl_loss + rel_KL_loss)loss_mimic_before
        self.backward(loss)
        
        # compute student metric
        top_k_obj = evaluate_topk_object(obj_logits_student.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_cls_student.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]

        # compute teacher metric
        top_k_obj = evaluate_topk_object(obj_logits_teacher.detach(), gt_cls, topk=11)
        top_k_rel = evaluate_topk_predicate(rel_cls_teacher.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        obj_topk_teacher_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_teacher_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        
        
        log = [("train/rel_loss", loss_rel_student.detach().item()),
                ("train/obj_loss", loss_obj_student.detach().item()),
                ("train/2d_rel_loss", loss_rel_teacher.detach().item()),
                ("train/2d_obj_loss", loss_obj_teacher.detach().item()),
                ("train/mimic_loss_after", loss_mimic_after.detach().item()),
                ("train/logit_scale_teacher", logit_scale_teacher.detach().item()),
                ("train/logit_scale_student", logit_scale_student.detach().item()),
                ("train/loss_mimic_before", loss_mimic_before.detach().item()),
                #("train/obj_kl_loss", obj_kl_loss.detach().item()),
                ("train/rel_mimic_teacher", rel_mimic_teacher.detach().item()),
                ("train/rel_mimic_student", rel_mimic_student.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
                ("train/Obj_R1_2d", obj_topk_teacher_list[0]),
                ("train/Obj_R5_2d", obj_topk_teacher_list[1]),
                ("train/Obj_R10_2d", obj_topk_teacher_list[2]),
                ("train/Pred_R1_2d", rel_topk_teacher_list[0]),
                ("train/Pred_R3_2d", rel_topk_teacher_list[1]),
                ("train/Pred_R5_2d", rel_topk_teacher_list[2]),
            ]
        return log
           
    def process_val(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, use_triplet=False):
 
        obj_logits_teacher, obj_logits_student, rel_cls_teacher, rel_cls_student = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_logits_student.detach().cpu(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_cls_student.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)

        top_k_obj_2d = evaluate_topk_object(obj_logits_teacher.detach().cpu(), gt_cls, topk=11)
        top_k_rel_2d = evaluate_topk_predicate(rel_cls_teacher.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        
        if use_triplet:
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_logits_student.detach().cpu(), rel_cls_student.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=True, obj_topk=top_k_obj)
            top_k_2d_triplet, _, _, _, _ = evaluate_triplet_topk(obj_logits_teacher.detach().cpu(), rel_cls_teacher.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=True, obj_topk=top_k_obj)
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
