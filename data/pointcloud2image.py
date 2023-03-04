from asyncio import sleep
import json, os, trimesh, argparse
from xml.dom import INDEX_SIZE_ERR
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL.Image as Image
import clip, torch, numpy as np


model, preprocess = clip.load("ViT-B/32", device='cuda')

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def process_imgs(imgs):
    # rotate images
    a = torch.stack([preprocess(Image.fromarray(img).transpose(Image.ROTATE_270)).cuda()for img in imgs], dim=0)
    return a

def read_pointcloud(scan_id):
    """
    Reads a pointcloud from a file and returns points with instance label.
    """
    plydata = trimesh.load(os.path.join('/data/wangziqin/project/CVPR2023-VLSAT/data/3RScan', scan_id, 'labels.instances.annotated.v2.ply'), process=False)
    points = np.array(plydata.vertices)
    labels = np.array(plydata.metadata['ply_raw']['vertex']['data']['objectId'])

    return points, labels

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

    # convert data to dict('scene_id': {'obj': [], 'rel': []})
    scene_data = dict()
    for i in data['scans']:
        if i['scan'] not in scene_data.keys():
            scene_data[i['scan']] = {'obj': dict(), 'rel': list()}
        scene_data[i['scan']]['obj'].update(i['objects'])
        scene_data[i['scan']]['rel'].extend(i['relationships'])

    return scene_data, selected_scans

def read_intrinsic(intrinsic_path, mode='rgb'):
    with open(intrinsic_path, "r") as f:
        data = f.readlines()
    
    m_versionNumber = data[0].strip().split(' ')[-1]
    m_sensorName = data[1].strip().split(' ')[-2]
    
    if mode == 'rgb':
        m_Width = int(data[2].strip().split(' ')[-1])
        m_Height = int(data[3].strip().split(' ')[-1])
        m_Shift = None
        m_intrinsic = np.array([float(x) for x in data[7].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))
    else:
        m_Width = int(data[4].strip().split(' ')[-1])
        m_Height = int(data[5].strip().split(' ')[-1])
        m_Shift = int(data[6].strip().split(' ')[-1])
        m_intrinsic = np.array([float(x) for x in data[9].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))
    
    m_frames_size = int(data[11].strip().split(' ')[-1])
    
    return dict(
        m_versionNumber=m_versionNumber,
        m_sensorName=m_sensorName,
        m_Width=m_Width,
        m_Height=m_Height,
        m_Shift=m_Shift,
        m_intrinsic=m_intrinsic,
        m_frames_size=m_frames_size
    )

def read_extrinsic(extrinsic_path):
    pose = []
    with open(extrinsic_path) as f:
        lines = f.readlines()
    for line in lines:
        pose.append([float(i) for i in line.strip().split(' ')])
    return pose

def read_scan_info(scan_id, mode='rgb'):
    scan_path = os.path.join("/data/wangziqin/project/CVPR2023-VLSAT/data/3RScan", scan_id)
    sequence_path = os.path.join(scan_path, "sequence")
    intrinsic_path = os.path.join(sequence_path, "_info.txt")
    intrinsic_info = read_intrinsic(intrinsic_path, mode='rgb')
    mode_template = 'color.jpg' if mode == 'rgb' else 'depth.pgm'
    
    image_list, extrinsic_list = [], []
    
    for i in range(0, intrinsic_info['m_frames_size']):
        frame_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6)+ mode_template)
        extrinsic_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6)+ "pose.txt")
        assert os.path.exists(frame_path) and os.path.exists(extrinsic_path)
        image_list.append(np.array(plt.imread(frame_path)))
        # inverce the extrinsic matrix, from camera_2_world to world_2_camera
        extrinsic = np.matrix(read_extrinsic(extrinsic_path))
        extrinsic_list.append(extrinsic.I)
        sleep(1)
    
    return np.array(image_list), np.array(extrinsic_list), intrinsic_info

def map_pc_to_image(points, instances, image_list, instance_names, extrinsics, intrinsic, width, height, class_list, class_weight, save_path, scene_id, fin_all, topk=10):
    """
    Maps a pointcloud to an image.
    """
    instance_id = set(instance_names.keys())
    
    # get clip match rate to filter some transport noise
    image_input = process_imgs(image_list)
    with torch.no_grad():
        image_feature = model.encode_image(image_input)
    image_feature /= image_feature.norm(dim=-1, keepdim=True)
    similarity = (image_feature @ class_weight.T).softmax(dim=-1)

    # compute max number of instances class
    name_count = {}
    for i in instance_id:
        name = instance_names[i]
        if name not in name_count.keys():
            name_count[name] = 0
        name_count[name] += 1
    max_num = max(name_count.values())
    topk = min(30, max(topk, max_num * int(np.ceil(image_feature.shape[0] / len(instance_id)))), int(image_feature.shape[0] / 10))
    
    for i in instance_id:
        # record the project quality
        fin = open(os.path.join(save_path, 'project_quality.txt'), 'a')

        # found the instance points, convert to homogeneous coordinates
        points_i = points[(instances==int(i)).flatten()]
        if points_i.shape[0] == 0:
            continue

        points_i = np.concatenate((points_i, np.ones((points_i.shape[0],1))), axis=-1)
        # transform to camera coordinates
        w_2_c = (extrinsics @ points_i.T)   # n_frames x 4 x n_points
        # transform to image coordinates
        c_2_i = intrinsic[:3, :] @ w_2_c    # n_frames x 3 x n_points
        c_2_i = c_2_i.transpose(0, 2, 1)    # n_frames x n_points x 3
        c_2_i = c_2_i[...,:2] / c_2_i[..., 2:] # n_frames x n_points x 2
        # find the points in the image
        indexs = ((c_2_i[...,0]< width) & (c_2_i[...,0]>0) & (c_2_i[...,1]< height) & (c_2_i[...,1]>0))
        # select top-k frames
        # topk_index = np.argsort(-indexs.mean(-1))
        # we use clip to filter some images that obviously donot belong to the class
        class_idx = class_list.index(instance_names[i])
        topk_index = (-similarity[:, class_idx]).argsort()[:topk]
        # select top-k points
        idx = 0
        # scores = []
        quanlity = None
        # for k in topk_index:
        #     #scores.append(indexs.mean(-1)[k])
        #     c_2_i_k = c_2_i[k][indexs[k].reshape(-1)]
        #     if len(c_2_i_k) == 0:
        #         scores.append(0)
        #         continue
        #     padding_x = min(height * 0.3, 20)
        #     padding_y = min(width * 0.3, 20)
        #     left_up_x = max(0, int(c_2_i_k[...,1].min()) - padding_x)
        #     left_up_y = max(0, int(c_2_i_k[...,0].min()) - padding_y)
        #     right_down_x = min(int(c_2_i_k[...,1].max()) + padding_x, height)
        #     right_down_y = min(int(c_2_i_k[...,0].max()) + padding_y, width)           
        #     scores.append((right_down_x - left_up_x)*(right_down_y - left_up_y) / (width * height))

        
        # # sort the score
        # scores_idx = (-np.array(scores)).argsort()
        # topk_index = topk_index[scores_idx]


        # save the image
        croped_image_feats = []
        origin_image_feats = []
        
        # clip filter + quality filter : quanlity A
        for k in topk_index:
            c_2_i_k = c_2_i[k][indexs[k].reshape(-1)]
            image_i = image_list[k]
            object_pc_ratio = indexs.mean(-1)[k]
            if len(c_2_i_k) == 0:
                continue
            padding_x = min(height * 0.3, 20)
            padding_y = min(width * 0.3, 20)
            left_up_x = max(0, int(c_2_i_k[...,1].min()) - padding_x)
            left_up_y = max(0, int(c_2_i_k[...,0].min()) - padding_y)
            right_down_x = min(int(c_2_i_k[...,1].max()) + padding_x, height)
            right_down_y = min(int(c_2_i_k[...,0].max()) + padding_y, width)
            
            img_ratio = (right_down_x - left_up_x)*(right_down_y - left_up_y) / (width * height)
            croped_image_i = image_i[left_up_x:right_down_x, left_up_y:right_down_y]
            plt.imsave(os.path.join(save_path, f'instance_{i}_class_{instance_names[i]}_croped_view{idx}_score_{object_pc_ratio}_ratio_{img_ratio}_A.jpg'),croped_image_i)
            plt.imsave(os.path.join(save_path, f'instance_{i}_class_{instance_names[i]}_view{idx}_{k}_A.jpg'),image_i)
            
            # get image clip feature
            with torch.no_grad():
                croped_image_feats.append(model.encode_image(preprocess(Image.fromarray(croped_image_i).transpose(Image.ROTATE_270)).unsqueeze(0).cuda()).cpu().numpy())
                origin_image_feats.append(model.encode_image(preprocess(Image.fromarray(image_i).transpose(Image.ROTATE_270)).unsqueeze(0).cuda()).cpu().numpy())
            
            quanlity = 'A'
            idx += 1
            if idx >= 5:
                break
            
        # quanlity filter : quanlity B
        if idx == 0:
            topk_index = np.argsort(-indexs.mean(-1))
            for k in topk_index:
                c_2_i_k = c_2_i[k][indexs[k].reshape(-1)]
                image_i = image_list[k]
                if len(c_2_i_k) == 0:
                    continue

                padding_x = min(height * 0.3, 20)
                padding_y = min(width * 0.3, 20)
                left_up_x = max(0, int(c_2_i_k[...,1].min()) - padding_x)
                left_up_y = max(0, int(c_2_i_k[...,0].min()) - padding_y)
                right_down_x = min(int(c_2_i_k[...,1].max()) + padding_x, height)
                right_down_y = min(int(c_2_i_k[...,0].max()) + padding_y, width)
                img_ratio = (right_down_x - left_up_x)*(right_down_y - left_up_y) / (width*height)
                
                croped_image_i = image_i[left_up_x:right_down_x, left_up_y:right_down_y]
                plt.imsave(os.path.join(save_path, f'instance_{i}_class_{instance_names[i]}_croped_view{idx}_score_{indexs.mean(-1)[k]}_ratio_{img_ratio}_B.jpg'),croped_image_i)
                plt.imsave(os.path.join(save_path, f'instance_{i}_class_{instance_names[i]}_view{idx}_{k}_B.jpg'),image_i)
                
                # get image clip feature
                with torch.no_grad():
                    croped_image_feats.append(model.encode_image(preprocess(Image.fromarray(croped_image_i).transpose(Image.ROTATE_270)).unsqueeze(0).cuda()).cpu().numpy())
                    origin_image_feats.append(model.encode_image(preprocess(Image.fromarray(image_i).transpose(Image.ROTATE_270)).unsqueeze(0).cuda()).cpu().numpy())
                
                idx += 1
                if quanlity is None:
                    quanlity  = 'B'
                if idx >= 5:
                    break 
        
        # clip filter : quanlity C
        if idx == 0:
            assert indexs.mean() == 0
            # there is no map to the image, just use clip feature
            topk_index = (-similarity[:, class_idx]).argsort()
            for k in topk_index:
                image_i = image_list[k]
                plt.imsave(os.path.join(save_path, f'instance_{i}_class_{instance_names[i]}_croped_view{idx}_score_0_C.jpg'), image_i)
                plt.imsave(os.path.join(save_path, f'instance_{i}_class_{instance_names[i]}_view{idx}_{k}_C.jpg'),image_i)
                
                # get image clip feature             
                with torch.no_grad():
                    tmp_feat = model.encode_image(preprocess(Image.fromarray(image_i).transpose(Image.ROTATE_270)).unsqueeze(0).cuda()).cpu().numpy()
                croped_image_feats.append(tmp_feat)
                origin_image_feats.append(tmp_feat) 
                idx += 1
                if quanlity is None:
                    quanlity = 'C'
                if idx == 3:
                    break
            with open(os.path.join(save_path, f'no_view.txt'), 'a') as f:
                f.write(f'instance {i} has no view found, use clip feature sort instead \n')

        assert quanlity is not None, 'there is no view found'
        # record the quanlity
        fin.write(f'Quanlity:{quanlity} instance:{i} label:{instance_names[i]} \n')
        fin.close()
        fin_all.write(f'Scene:{scene_id} Instance:{i} Label:{instance_names[i]} Quanlity:{quanlity} \n')
        fin_all.flush()
        
        # store multi-view feature
        croped_image_feats_mean = np.concatenate(croped_image_feats, axis=0).mean(axis=0, keepdims=True)
        origin_image_feats_mean = np.concatenate(origin_image_feats, axis=0).mean(axis=0, keepdims=True)

        np.save(os.path.join(save_path, f'instance_{i}_class_{instance_names[i]}_croped_view_mean.npy'), croped_image_feats_mean)
        np.save(os.path.join(save_path, f'instance_{i}_class_{instance_names[i]}_origin_view_mean.npy'), origin_image_feats_mean)

        # save clip-sorted feature
        # clip_feat = []
        # topk_index = (-similarity[:, class_idx]).argsort()[:3]
        # for k in topk_index:
        #     image_i = image_list[k]
        #     with torch.no_grad():
        #         clip_feat.append(model.encode_image(preprocess(Image.fromarray(image_i).transpose(Image.ROTATE_270)).unsqueeze(0).cuda()).cpu().numpy())
        
        # clip_feats_mean = np.concatenate(clip_feat, axis=0).mean(axis=0, keepdims=True)
        # clip_feats_max = np.concatenate(clip_feat, axis=0).max(axis=0, keepdims=True)
        # np.save(os.path.join(save_path, f'instance_{i}_class_{instance_names[i]}_origin_clip_mean.npy'), clip_feats_mean)
        # np.save(os.path.join(save_path, f'instance_{i}_class_{instance_names[i]}_origin_clip_max.npy'), clip_feats_max)
        

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', type=str, default='train', help='train or test')
    args = argparser.parse_args()
    print("========= Deal with {} ========".format(args.mode))
    scene_data, selected_scans = read_json(args.mode)
    # record global quanlity
    fin_all = open(os.path.join(f'/data/wangziqin/project/CVPR2023-VLSAT/data/3RScan/{args.mode}_all_quanlity.txt'), 'a')
    for i in tqdm(selected_scans):
        instance_names = scene_data[i]['obj']
        pc_i, instances_i = read_pointcloud(i)
        # print(f'======= read image and extrinsic for {i} =========')
        image_list, extrinsic_list, intrinsic_info = read_scan_info(i)
        save_path = f'/data/wangziqin/project/CVPR2023-VLSAT/data/3RScan/{i}/multi_view'
        os.makedirs(save_path, exist_ok=True)
        # print(f'======= map pointcloud to image =========')
        class_list, class_weight = get_label('/data/wangziqin/project/CVPR2023-VLSAT/data/3DSSG_subset/classes.txt')
        map_pc_to_image(pc_i, instances_i, image_list, instance_names, extrinsic_list, intrinsic_info['m_intrinsic'], intrinsic_info['m_Width'], intrinsic_info['m_Height'], class_list, class_weight, save_path, i, fin_all)
    fin_all.close()