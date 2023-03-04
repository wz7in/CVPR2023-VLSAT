import numpy as np
import torch
import torch.nn.functional as F
import numpy as np


def evaluate_triplet_recallk(objs_pred, rels_pred, gt_rel, edges, multi_rel_outputs, topk, topk_each, use_clip=False, evaluate='triplet'):
    # objs_pred: N_o * 160
    # rels_pred: N_r * 26
    # gt_rel: N_r * 26, multiple
    # edges: N_r * 2, 0 - N_o-1 (obj index)
    # confidence_threshold: no
    res, triplet = [], []
    if not use_clip:
        # convert the score from log_softmax to softmax
        objs_pred = np.exp(objs_pred)
    else:
        # convert the score to softmax
        objs_pred = F.softmax(objs_pred, dim=-1)
    
    if not multi_rel_outputs:
        rels_pred = np.exp(rels_pred)

    all_topk_conf_matrix, all_topk_id = None, None
    topk_list = topk if isinstance(topk, list) else [topk]
    topk = np.max(topk_list)

    for edge in range(len(edges)):
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        sub = objs_pred[edge_from]
        obj = objs_pred[edge_to]

        node_score = torch.einsum('n,m->nm',sub,obj)
        if evaluate == 'triplet':
            conf_matrix = torch.einsum('nl,m->nlm',node_score,rel_predictions)
            conf_matrix_1d = conf_matrix.reshape(-1)
        elif evaluate == 'rels':
            conf_matrix_1d = rel_predictions
        else:
            raise NotImplementedError('evaluate type', evaluate)

        curr_topk_conf_matrix, curr_topk_conf_id = conf_matrix_1d.topk(min(topk_each, conf_matrix_1d.shape[0]), largest=True)
        curr_edge_id = torch.zeros_like(curr_topk_conf_id) + edge
        # (edgeid, topk-conf-id)
        # edgeid represents (object and subject) id
        # topk-conf-id represents (sub type, obj type, rel type)
        curr_topk_id = torch.stack([curr_edge_id, curr_topk_conf_id], dim=-1)

        # select all --- matrix
        if all_topk_conf_matrix is None:
            all_topk_conf_matrix = curr_topk_conf_matrix
            all_topk_id = curr_topk_id
        else:
            # print(all_topk_conf_matrix.shape, all_topk_id.shape)
            all_topk_conf_matrix = torch.cat([all_topk_conf_matrix, curr_topk_conf_matrix], dim=0)
            all_topk_id = torch.cat([all_topk_id, curr_topk_id], dim=0)
        all_topk_conf_matrix, select_id = all_topk_conf_matrix.topk(min(topk, all_topk_conf_matrix.shape[0]), largest=True, sorted=True)
        all_topk_id = all_topk_id[select_id]

        # sorted_conf_matrix, sorted_args_1d = torch.sort(conf_matrix_1d, descending=True)
    # print(all_topk_id, all_topk_conf_matrix)
    # import ipdb; ipdb.set_trace()

    pred_triplets = []
    correct_number, all_number = [0 for i in topk_list], 0  # all_number: all correct number in the gt
    #all_number = sum([len(gt_edge[2]) for gt_edge in gt_rel])
    all_number = sum([min(1,len(gt_edge[2])) for gt_edge in gt_rel])
    # print(all_number, '<< gt edge number')
    # print(all_topk_conf_matrix, 'all conf')

    size_o, size_r = objs_pred.shape[1], rels_pred.shape[1]
    iscompute = [{} for _ in range(len(topk_list))]
    for idk, [edge, idx_1d] in enumerate(all_topk_id):  # calculate for each predicted edge
        conf_score = all_topk_conf_matrix[idk]

        # same edge id (same object and subject)
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]

        sub_gt= gt_rel[edge][0]
        obj_gt = gt_rel[edge][1]
        rel_gt = gt_rel[edge][2]

        edge = int(edge)
        if evaluate == 'triplet':
            idx = np.unravel_index(idx_1d, (size_o, size_o, size_r))
            if sub_gt == idx[0] and obj_gt == idx[1] and (idx[2] in rel_gt):
                for _, k in enumerate(topk_list):
                    if idk < k and edge not in iscompute[_].keys():
                        correct_number[_] += 1
                        iscompute[_][edge] = 1
                # print(conf_score, edge, 'idx', idx, 'edge from and to', edge_from, edge_to, 'gt type', sub_gt, obj_gt, rel_gt)
            pred_triplets.append(((edge_from, edge_to), idx, conf_score))  # edge, object&predicate cls type
        elif evaluate == 'rels':
            idx = idx_1d
            if idx in rel_gt:
                for _, k in enumerate(topk_list):
                    if idk < k and edge not in iscompute[_].keys():
                        correct_number[_] += 1
                        iscompute[_][edge] = 1
            pred_triplets.append(((edge_from, edge_to), (-1, -1, idx)))  # edge, object&predicate cls type
        else:
            raise NotImplementedError()

    # print(correct_number, all_number)
    correct_number = np.array(correct_number)
    #return pred_triplets, correct_number/all_number
    # print(correct_number, all_number, iscompute)
    return correct_number/all_number

def evaluate_triplet_mrecallk(objs_pred, rels_pred, gt_rel, edges, multi_rel_outputs, topk, topk_each, use_clip=False, evaluate='triplet'):
    # objs_pred: N_o * 160
    # rels_pred: N_r * 26
    # gt_rel: N_r * 26, multiple
    # edges: N_r * 2, 0 - N_o-1 (obj index)
    # confidence_threshold: no
    res, triplet = [], []
    if not use_clip:
        # convert the score from log_softmax to softmax
        objs_pred = np.exp(objs_pred)
    else:
        # convert the score to softmax
        objs_pred = F.softmax(objs_pred, dim=-1)
    
    if not multi_rel_outputs:
        rels_pred = np.exp(rels_pred)

    all_topk_conf_matrix, all_topk_id = None, None
    topk_list = topk if isinstance(topk, list) else [topk]
    topk = np.max(topk_list)

    for edge in range(len(edges)):
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        sub = objs_pred[edge_from]
        obj = objs_pred[edge_to]

        node_score = torch.einsum('n,m->nm',sub,obj)
        if evaluate == 'triplet':
            conf_matrix = torch.einsum('nl,m->nlm',node_score,rel_predictions)
            conf_matrix_1d = conf_matrix.reshape(-1)
        elif evaluate == 'rels':
            conf_matrix_1d = rel_predictions
        else:
            raise NotImplementedError('evaluate type', evaluate)

        curr_topk_conf_matrix, curr_topk_conf_id = conf_matrix_1d.topk(min(topk_each, conf_matrix_1d.shape[0]), largest=True)
        curr_edge_id = torch.zeros_like(curr_topk_conf_id) + edge
        # (edgeid, topk-conf-id)
        # edgeid represents (object and subject) id
        # topk-conf-id represents (sub type, obj type, rel type)
        curr_topk_id = torch.stack([curr_edge_id, curr_topk_conf_id], dim=-1)

        # select all --- matrix
        if all_topk_conf_matrix is None:
            all_topk_conf_matrix = curr_topk_conf_matrix
            all_topk_id = curr_topk_id
        else:
            # print(all_topk_conf_matrix.shape, all_topk_id.shape)
            all_topk_conf_matrix = torch.cat([all_topk_conf_matrix, curr_topk_conf_matrix], dim=0)
            all_topk_id = torch.cat([all_topk_id, curr_topk_id], dim=0)
        all_topk_conf_matrix, select_id = all_topk_conf_matrix.topk(min(topk, all_topk_conf_matrix.shape[0]), largest=True, sorted=True)
        all_topk_id = all_topk_id[select_id]

        # sorted_conf_matrix, sorted_args_1d = torch.sort(conf_matrix_1d, descending=True)
    # print(all_topk_id, all_topk_conf_matrix)
    # import ipdb; ipdb.set_trace()

    pred_triplets = []
    correct_number = [[0 for _ in topk_list] for _ in range(26)] 
    #all_number = sum([len(gt_edge[2]) for gt_edge in gt_rel])
    all_number_perclass = [ sum([1 if i in gt_edge[2] else 0 for gt_edge in gt_rel]) for i in range(26)]
    # print(all_number, '<< gt edge number')
    # print(all_topk_conf_matrix, 'all conf')

    size_o, size_r = objs_pred.shape[1], rels_pred.shape[1]
    iscompute = [{} for _ in range(len(topk_list))]
    
    for idk, [edge, idx_1d] in enumerate(all_topk_id):  # calculate for each predicted edge
        conf_score = all_topk_conf_matrix[idk]

        # same edge id (same object and subject)
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]

        sub_gt= gt_rel[edge][0]
        obj_gt = gt_rel[edge][1]
        rel_gt = gt_rel[edge][2]

        edge = int(edge)
        if evaluate == 'triplet':
            idx = np.unravel_index(idx_1d, (size_o, size_o, size_r))
            if sub_gt == idx[0] and obj_gt == idx[1] and (idx[2] in rel_gt):
                for _, k in enumerate(topk_list):
                    #for cls in range(26):
                    if idk < k and edge not in iscompute[_].keys():
                        for r in rel_gt:
                            correct_number[r][_] += 1
                        iscompute[_][edge] = 1
                # print(conf_score, edge, 'idx', idx, 'edge from and to', edge_from, edge_to, 'gt type', sub_gt, obj_gt, rel_gt)
            pred_triplets.append(((edge_from, edge_to), idx, conf_score))  # edge, object&predicate cls type
        elif evaluate == 'rels':
            idx = idx_1d
            if idx in rel_gt:
                for _, k in enumerate(topk_list):
                    if idk < k and edge not in iscompute[_].keys():
                        for r in rel_gt:
                            correct_number[r][_] += 1
                        iscompute[_][edge] = 1
                    # for cls in range(26):
                    #     if idk < k and cls == idx:
                    #         correct_number[cls][_] += 1

            #pred_triplets.append(((edge_from, edge_to), (-1, -1, idx)))  # edge, object&predicate cls type
        else:
            raise NotImplementedError()

    # print(correct_number, all_number)
    correct_number = np.array(correct_number)

    #return pred_triplets, correct_number/all_number
    # print(correct_number, all_number, iscompute)
    return [[correct_number[j][i] / all_number_perclass[j] if all_number_perclass[j]!=0 else -1 for i in range(3)] for j in range(26)]
