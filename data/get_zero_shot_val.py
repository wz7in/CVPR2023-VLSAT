import json
from tqdm import tqdm
import numpy as np

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def read_json(split):
    """
    Reads a json file and returns points with instance label.
    """
    selected_scans = set()
    if split == 'train' :
        selected_scans = selected_scans.union(read_txt_to_list('/data/wangziqin/project/CVPR2023-VLSAT/data/3DSSG_subset/train_scans.txt'))
        with open("/data/wangziqin/project/CVPR2023-VLSAT/data/3DSSG_subset/relationships_train.json", "r") as read_file:
            data = json.load(read_file)
    elif split == 'val':
        selected_scans = selected_scans.union(read_txt_to_list('/data/wangziqin/project/CVPR2023-VLSAT/data/3DSSG_subset/validation_scans.txt'))
        with open("/data/wangziqin/project/CVPR2023-VLSAT/data/3DSSG_subset/relationships_validation.json", "r") as read_file:
            data = json.load(read_file)
    else:
        raise RuntimeError('unknown split type:',split)

    return data

def get_zero_shot_recall(triplet_rank, cls_matrix, obj_names, rel_name):
   
    train_data = read_json('train')
    scene_data = dict()
    for i in train_data['scans']:
        objs = i['objects']
        for rel in i['relationships']:
            if str(rel[0]) not in objs.keys():
                print(f'{rel[0]} not in objs in scene {i["scan"]} split {i["split"]}')
                continue
            if str(rel[1]) not in objs.keys():
                print(f'{rel[1]} not in objs in scene {i["scan"]} split {i["split"]}')
                continue
            triplet_name = str(obj_names.index(objs[str(rel[0])])) + ' ' + str(obj_names.index(objs[str(rel[1])])) + ' ' + str(rel_name.index(rel[-1]))
            if triplet_name not in scene_data.keys():
                scene_data[triplet_name] = 1
            scene_data[triplet_name] += 1
    
    val_data = read_json('val')
    res = []
    count = 0
    for i in tqdm(val_data['scans']):
        objs = i['objects']
        for rel in i['relationships']:
            count += 1
            triplet_name = str(obj_names.index(objs[str(rel[0])])) + ' ' + str(obj_names.index(objs[str(rel[1])])) + ' ' + str(rel_name.index(rel[-1]))
            if triplet_name not in scene_data.keys():
                res.append(triplet_name)
    
    # get valid triplet which not appears in train data
    valid_triplet = []
    for i in range(len(cls_matrix)):
        if cls_matrix[i, -1] == -1:
            continue
        if len(cls_matrix[i]) == 5:
            triplet_name = str(cls_matrix[i][0]) + ' ' + str(cls_matrix[i][2]) + ' ' + str(cls_matrix[i][-1])
        elif len(cls_matrix[i]) == 3:
            triplet_name = str(cls_matrix[i][0]) + ' ' + str(cls_matrix[i][1]) + ' ' + str(cls_matrix[i][-1])
        else:
            raise RuntimeError('unknown triplet length:', len(cls_matrix[i]))

        if triplet_name in res:
            valid_triplet.append(triplet_rank[i])
    
    import ipdb; ipdb.set_trace()
    
    return np.array(valid_triplet)

if __name__ == '__main__':

    with open('/data/wangziqin/project/CVPR2023-VLSAT/data/3DSSG_subset/classes.txt') as f:
        obj_names = f.read().splitlines()
    with open('/data/wangziqin/project/CVPR2023-VLSAT/data/3DSSG_subset/relationships.txt') as f:
        rel_name = f.read().splitlines()
    rel_name.pop(0)
    a = np.load('/data/wangziqin/project/CVPR2023-VLSAT/config/results/Mmgnet/fill_mimic_ca/cls_matrix_list.npy')
    aa = np.load('/data/wangziqin/project/CVPR2023-VLSAT/config/results/Mmgnet/fill_baseline_obj512/cls_matrix_list.npy')
    b = np.load('/data/wangziqin/project/CVPR2023-VLSAT/config/results/Mmgnet/fill_mimic_ca/topk_triplet_list.npy')
    bb = np.load('/data/wangziqin/project/CVPR2023-VLSAT/config/results/Mmgnet/fill_baseline_obj512/topk_triplet_list.npy')
    res = get_zero_shot_recall(b, a, obj_names, rel_name)
    import ipdb; ipdb.set_trace()


            