import json, os
import clip, torch, numpy as np
import argparse
from tqdm import tqdm


model, preprocess = clip.load("ViT-B/32", device='cuda')

def read_json(split):
    """
    Reads a json file and returns points with instance label.
    """
    selected_scans = set()
    if split == 'train_scans_1' or split == 'train_scans_2' or split == 'train_scans_3' or split == 'train_scans_4':
        selected_scans = selected_scans.union(read_txt_to_list(f'/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/{split}.txt'))
        with open("/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/relationships_train.json", "r") as read_file:
            data = json.load(read_file)
    elif split == 'train' :
        selected_scans = selected_scans.union(read_txt_to_list('/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/train_scans.txt'))
        with open("/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/relationships_train.json", "r") as read_file:
            data = json.load(read_file)
    elif split == 'val':
        selected_scans = selected_scans.union(read_txt_to_list('/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/validation_scans.txt'))
        with open("/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/relationships_validation.json", "r") as read_file:
            data = json.load(read_file)
    else:
        raise RuntimeError('unknown split type:',split)

    # convert data to dict('scene_id': {'obj': [], 'rel': []})
    scene_data = dict()
    for i in data['scans']:
        if i['scan'] not in scene_data.keys():
            scene_data[i['scan']] = {'obj': dict(), 'rel': list()}
        scene_data[i['scan']]['obj'].update(i['objects'])
        scene_data[i['scan']]['rel'].extend(i['relationships'])

    return scene_data, selected_scans

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def get_label(label_path):
    label_list = []
    with open(label_path, "r") as f:
        data = f.readlines()
    for line in data:
        label_list.append(line.strip())
    # get norm clip weight
    text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in label_list]).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return label_list, text_features

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', type=str, default='train', help='train or test')
    args = argparser.parse_args()
    print("========= Deal with {} ========".format(args.mode))
    scene_data, selected_scans = read_json(args.mode)
    class_list, class_weight = get_label('/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/classes.txt')
    
    res = []
    mode = 'mean'
    for i in tqdm(selected_scans):
        instances = scene_data[i]['obj']
        for instance_id in instances.keys():
            class_idx = class_list.index(instances[instance_id])
            path = f'/data/caidaigang/project/3DSSG_Repo/data/3RScan/{i}/multi_view_depth_clip/instance_{instance_id}_class_{instances[instance_id]}_origin_view_{mode}.npy'
            if os.path.exists(path):
                instance_feat = torch.from_numpy(np.load(path)).cuda()
                instance_feat /= instance_feat.norm(dim=-1, keepdim=True)
                similarity = (instance_feat @ class_weight.T).argsort(descending=True)
                res.append(torch.where(similarity == class_idx)[1].item())
    
    res = np.array(res)
    topk1 = 100.0 * (res < 1).sum() / len(res)
    topk5 = 100.0 * (res < 5).sum() / len(res)
    topk10 = 100.0 * (res < 10).sum() / len(res)

    print("======== mode : {} ========".format(mode))
    print("Top 1: {:.2f}%".format(topk1))
    print("Top 5: {:.2f}%".format(topk5))
    print("Top 10: {:.2f}%".format(topk10))