from pydoc import describe
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
import numpy as np

class CustomSingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):
    def __init__(self,loader):
        super().__init__(loader)
    def IndexIter(self):
        return self._sampler_iter
    
class CustomMultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self,loader):
        super().__init__(loader)
    def IndexIter(self):
        return self._sampler_iter


class CustomDataLoader(DataLoader):
    def __init__(self, config, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        if worker_init_fn is None:
            worker_init_fn = self.init_fn
        super().__init__(dataset, batch_size, shuffle, sampler,
                 batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context)
        self.config = config
        
    def init_fn(self, worker_id):
        np.random.seed(self.config.SEED + worker_id)
        
    def __iter__(self):
        if self.num_workers == 0:
            return CustomSingleProcessDataLoaderIter(self)
        else:
            return CustomMultiProcessingDataLoaderIter(self)

def collate_fn_obj(batch):
    # batch
    
    name_list, instance2mask_list, obj_point_list, obj_label_list = [], [], [], []
    for i in batch:
        name_list.append(i[0])
        instance2mask_list.append(i[1])
        obj_point_list.append(i[2])
        obj_label_list.append(i[4])
    return name_list, instance2mask_list, torch.cat(obj_point_list, dim=0), torch.cat(obj_label_list, dim=0)

def collate_fn_rel(batch):
    # batch
    name_list, instance2mask_list, obj_label_list, rel_point_list, rel_label_list, edge_indices = [], [], [], [], [], []
    for i in batch:
        assert len(i) == 7
        name_list.append(i[0])
        instance2mask_list.append(i[1])
        obj_label_list.append(i[4])
        rel_point_list.append(i[3])
        rel_label_list.append(i[5])
        edge_indices.append(i[6])
    return name_list, instance2mask_list, torch.cat(obj_label_list, dim=0), torch.cat(rel_point_list, dim=0), torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0)

def collate_fn_obj_new(batch):
    # batch
    obj_point_list, obj_label_list = [], []
    for i in batch:
        obj_point_list.append(i[0])
        obj_label_list.append(i[2])
    return torch.cat(obj_point_list, dim=0), torch.cat(obj_label_list, dim=0)

def collate_fn_rel_new(batch):
    # batch
    rel_point_list, rel_label_list = [], []
    for i in batch:
        rel_point_list.append(i[1])
        rel_label_list.append(i[3])
    return torch.cat(rel_point_list, dim=0), torch.cat(rel_label_list, dim=0)


def collate_fn_all(batch):
    # batch
    obj_point_list, obj_label_list = [], []
    rel_point_list, rel_label_list = [], []
    edge_indices = []
    for i in batch:
        obj_point_list.append(i[0])
        obj_label_list.append(i[3])
        rel_point_list.append(i[2])
        rel_label_list.append(i[4])
        edge_indices.append(i[5])

    return torch.cat(obj_point_list, dim=0), torch.cat(obj_label_list, dim=0), torch.cat(rel_point_list, dim=0), torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0)

def collate_fn_all_des(batch):
    # batch
    obj_point_list, obj_label_list = [], []
    rel_label_list = []
    edge_indices, descriptor = [], []
    count = 0
    for i in batch:
        obj_point_list.append(i[0])
        obj_label_list.append(i[2])
        #rel_point_list.append(i[1])
        rel_label_list.append(i[3])
        edge_indices.append(i[4] + count)
        descriptor.append(i[5])
        # accumulate batch number to make edge_indices match correct object index
        count += i[0].shape[0]

    return torch.cat(obj_point_list, dim=0), torch.cat(obj_label_list, dim=0), torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0)

def collate_fn_all_2d(batch):
    # batch
    obj_point_list, obj_label_list, obj_2d_feats = [], [], []
    rel_label_list = []
    edge_indices, descriptor = [], []
    
    count = 0
    for i in batch:
        obj_point_list.append(i[0])
        obj_2d_feats.append(i[1])
        obj_label_list.append(i[3])
        #rel_point_list.append(i[2])
        rel_label_list.append(i[4])
        edge_indices.append(i[5] + count)
        descriptor.append(i[6])
        # accumulate batch number to make edge_indices match correct object index
        count += i[0].shape[0]

    return torch.cat(obj_point_list, dim=0), torch.cat(obj_2d_feats, dim=0), torch.cat(obj_label_list, dim=0), \
         torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0)

def collate_fn_det(batch):
    assert len(batch) == 1
    scene_points, obj_boxes, obj_labels, point_votes, point_votes_mask = [], [], [], [], []
    for i in range(len(batch)):
        scene_points.append(batch[i][0])
        obj_boxes.append(batch[i][1])
        obj_labels.append(batch[i][2])
        point_votes.append(batch[i][3])
        point_votes_mask.append(batch[i][4])
    
    scene_points = torch.stack(scene_points, dim=0)
    obj_boxes = torch.stack(obj_boxes, dim=0)
    obj_labels = torch.stack(obj_labels, dim=0)
    point_votes = torch.stack(point_votes, dim=0)
    point_votes_mask = torch.stack(point_votes_mask, dim=0)

    return scene_points, obj_boxes, obj_labels, point_votes, point_votes_mask


def collate_fn_mmg(batch):
    # batch
    obj_point_list, obj_label_list, obj_2d_feats = [], [], []
    rel_label_list = []
    edge_indices, descriptor = [], []
    batch_ids = []
    
    count = 0
    for i, b in enumerate(batch):
        obj_point_list.append(b[0])
        obj_2d_feats.append(b[1])
        obj_label_list.append(b[3])
        #rel_point_list.append(i[2])
        rel_label_list.append(b[4])
        edge_indices.append(b[5] + count)
        descriptor.append(b[6])
        # accumulate batch number to make edge_indices match correct object index
        count += b[0].shape[0]
        # get batchs location
        batch_ids.append(torch.full((b[0].shape[0], 1), i))


    return torch.cat(obj_point_list, dim=0), torch.cat(obj_2d_feats, dim=0), torch.cat(obj_label_list, dim=0), \
         torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0), torch.cat(batch_ids, dim=0)
