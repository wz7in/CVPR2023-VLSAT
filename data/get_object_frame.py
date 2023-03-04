from asyncio import sleep
import json, os, trimesh, argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL.Image as Image
import cv2, torch, numpy as np
from scipy.spatial import cKDTree

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def read_pointcloud(scan_id):
    """
    Reads a pointcloud from a file and returns points with instance label.
    """
    plydata = trimesh.load(os.path.join('/data/wangziqin/project/CVPR2023-VLSAT/data/3RScan', scan_id, 'labels.instances.annotated.v2.ply'), process=False)
    points = np.array(plydata.vertices)
    labels = np.array(plydata.metadata['ply_raw']['vertex']['data']['objectId'])

    return points, labels
   
def read_json(split):
    """
    Reads a json file and returns points with instance label.
    """
    selected_scans = set()
    if split == 'train_scans_1' or split == 'train_scans_2' or split == 'train_scans_3' or split == 'train_scans_4':
        selected_scans = selected_scans.union(read_txt_to_list(f'/data/wangziqin/project/CVPR2023-VLSAT/data/3DSSG_subset/{split}.txt'))
        with open("/data/wangziqin/project/CVPR2023-VLSAT/data/3DSSG_subset/relationships_train.json", "r") as read_file:
            data = json.load(read_file)
    elif split == 'train' :
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
        m_intrinsic=np.matrix(m_intrinsic),
        m_frames_size=m_frames_size
    )

def read_extrinsic(extrinsic_path):
    pose = []
    with open(extrinsic_path) as f:
        lines = f.readlines()
    for line in lines:
        pose.append([float(i) for i in line.strip().split(' ')])
    return pose

def read_scan_info(scan_id, mode='depth'):
    scan_path = os.path.join("/data/wangziqin/project/CVPR2023-VLSAT/data/3RScan", scan_id)
    sequence_path = os.path.join(scan_path, "sequence")
    intrinsic_path = os.path.join(sequence_path, "_info.txt")
    intrinsic_info = read_intrinsic(intrinsic_path, mode=mode)
    mode_template = 'color.jpg' if mode == 'rgb' else 'depth.pgm'
    
    image_list, extrinsic_list = [], []
    
    for i in range(0, intrinsic_info['m_frames_size']):
        frame_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6)+ mode_template)
        extrinsic_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6)+ "pose.txt")
        assert os.path.exists(frame_path) and os.path.exists(extrinsic_path)
        if mode == 'rgb':
            image_list.append(np.array(plt.imread(frame_path)))
        else:
            image_list.append(cv2.imread(frame_path, -1).reshape(-1))
        # inverce the extrinsic matrix, from camera_2_world to world_2_camera
        extrinsic = np.matrix(read_extrinsic(extrinsic_path))
        extrinsic_list.append(extrinsic)
        sleep(1)
    
    return np.array(image_list), np.array(extrinsic_list), intrinsic_info

def get_label(label_path):
    label_list = []
    with open(label_path, "r") as f:
        for line in f:
            label_list.append(line.strip())
    return label_list

def map_depth_to_pc(points, instances, image_list, instance_names, extrinsics, intrinsic, width, height, class_list, scene_id, sample_image=False):
    """
    Maps a depth image to a point cloud.
    """
    instance_id = set(instance_names.keys())
    classes = dict()

    # generate initial depth grid
    x = np.linspace(0, height-1, height)
    y = np.linspace(0, width-1, width)
    yy,xx = np.meshgrid(y,x)
    zz = np.ones((height, width)) 
    # homogeneous coordinates
    depth_grid = torch.from_numpy(np.stack([xx, yy, zz, zz], axis=-1).reshape(-1, 4)).float()
    # convert to gpu
    depth_grid = depth_grid.transpose(0, 1) # (4, height*width)
    if sample_image:
        depth_grid = depth_grid[:, ::3]
    extrinsics = torch.from_numpy(extrinsics).float()
    intrinsic = torch.from_numpy(intrinsic.I).float()
    points = torch.from_numpy(points).float() # (n_points, 3)

    # construct the ckdtree
    ckdtree = cKDTree(np.array(points))
    
    # per frame
    for i, (extrinsic, depth) in enumerate(zip(extrinsics,image_list)):

        i_2_c = torch.matmul(intrinsic, depth_grid).transpose(0, 1)
        if sample_image:
            depth = depth.reshape(-1,1)[::3]
            i_2_c[:,:3] *= depth
        else:
            i_2_c[:,:3] *= depth.reshape(-1,1)
        c_2_w = torch.matmul(extrinsic, i_2_c.transpose(0, 1)).transpose(0, 1) # (height*width, 4)
        # project to 3D
        c_2_w = c_2_w[..., :3] / c_2_w[..., 3:4] # (height*width, 3)
        # get the nearest instance for each point
        c_2_w = c_2_w[:, :3] / 1000 # (height*width, 3)

        # sort the index by distance
        indexs = ckdtree.query(c_2_w)[1]
        
        # gey instance id for each point
        project_id = instances[np.array(indexs)]

        for s in instance_id:
            if s not in classes.keys():
                classes[s] = []
            project_object_points = (project_id == int(s)).sum()
            if project_object_points > 0:
                classes[s].append(f'{str(i)}_{str(project_object_points)}')
    
    # save the result
    json.dump(classes, open(os.path.join('/data/wangziqin/project/CVPR2023-VLSAT/data/3RScan', scene_id, "classes_2.json"), "w"))

        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', type=str, default='train', help='train or test')
    args = argparser.parse_args()
    print("========= Deal with {} ========".format(args.mode))
    scene_data, selected_scans = read_json(args.mode)
    for i in tqdm(selected_scans):
        instance_names = scene_data[i]['obj']
        pc_i, instances_i = read_pointcloud(i)
        # print(f'======= read image and extrinsic for {i} =========')
        image_list, extrinsic_list, intrinsic_info = read_scan_info(i)
        # print(f'======= map pointcloud to image =========')
        class_list = get_label('/data/wangziqin/project/CVPR2023-VLSAT/data/3DSSG_subset/classes.txt')
        map_depth_to_pc(pc_i, instances_i, image_list, instance_names, extrinsic_list, intrinsic_info['m_intrinsic'], intrinsic_info['m_Width'], intrinsic_info['m_Height'], class_list, i)