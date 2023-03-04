import os
import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    model, preprocess = clip.load("RN101", device='cuda')
    res_croped, res_origin = [], []

    with open('/data/caidaigang/project/3DSSG_Repo/data/3RScan/val_all_quanlity.txt') as f:
        lines = f.readlines()
    
    for i in tqdm(lines):
        items = i.strip().split(':')
        scene_id = items[1].split(' ')[0]
        instance_id = items[2].split(' ')[0]
        label_name = ' '.join(items[3].split(' ')[0:-1])
        path_origin = f'instance_{instance_id}_class_{label_name}_view'
        
        root = f'/data/caidaigang/project/3DSSG_Repo/data/3RScan/'
        files = os.listdir(os.path.join(root,f'{scene_id}/multi_view'))
        tmp_origin = []
        for j in files:
            if path_origin in j and 'npy' not in j:
                tmp_origin.append(os.path.join(f'{scene_id}/multi_view',j))
        
        # compute best view for each instance
        vision_feat = [preprocess(Image.open(os.path.join(root, t)).rotate(Image.ROTATE_270)) for t in tmp_origin]
        vision_feat = torch.stack(vision_feat, dim=0).cuda()
        with torch.no_grad():
            vision_feat = model.encode_image(vision_feat)
        
        vision_feat = vision_feat / vision_feat.norm(dim=-1, keepdim=True)
        vision_feat = vision_feat.mean(dim=0, keepdim=True)
        save_path = f'instance_{instance_id}_class_{label_name}_origin_view_mean_2.npy'
        save_path = os.path.join(root, f'{scene_id}/multi_view', save_path)
        np.save(save_path, vision_feat.cpu().numpy())