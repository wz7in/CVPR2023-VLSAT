import torch, numpy as np
import os.path as osp
from PIL import Image
import torchvision.transforms as transforms

class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, data_list_path, labels_path, mode):
        
        self.root_path = root_path
        self.data_list_path = data_list_path
        self.labels_path = labels_path
        self.mode = mode
        self.labels = self.load_labels(labels_path)
        self.data_list = self.load_data_list(data_list_path)

    def __len__(self):
        return len(self.data_list)
    
    def load_labels(self, labels_path):
        label_list = []
        with open(labels_path, "r") as f:
            for line in f:
                label_list.append(line.strip())
        return label_list
    
    def load_data_list(self, data_list_path):
        data_list = []
        with open(data_list_path, 'r') as f:
            for line in f:
                items = line.strip().split(':')
                scene_id = items[1].split(' ')[0]
                instance_id = items[2].split(' ')[0]
                label_name = ' '.join(items[3].split(' ')[0:-1])
                label_id = self.labels.index(label_name)
                path = f'instance_{instance_id}_class_{label_name}_{self.mode}.npy'
                path = osp.join(self.root_path, scene_id, 'multi_view', path)
                data_list.append((path, label_id))
        
        return data_list

    def __getitem__(self, idx):
        path, label = self.data_list[idx]
        image_features = torch.from_numpy(np.load(path)).float()
        return image_features.reshape(-1), int(label)

class MultiViewPCDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_path, data_list_path, labels_path):
        
        self.root_path = root_path
        self.data_list_path = data_list_path
        self.labels_path = labels_path
        self.labels = self.load_labels(labels_path)
        self.data_list = self.load_data_list(data_list_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            lambda image: image.convert("RGB"),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_list)
    
    def load_labels(self, labels_path):
        label_list = []
        with open(labels_path, "r") as f:
            for line in f:
                label_list.append(line.strip())
        return label_list
    
    def load_data_list(self, data_list_path):
        data_list = []
        with open(data_list_path, 'r') as f:
            for line in f:
                items = line.strip().split(':')
                scene_id = items[1].split(' ')[0]
                instance_id = items[2].split(' ')[0]
                label_name = ' '.join(items[3].split(' ')[0:-1])
                label_id = self.labels.index(label_name)
                all_path = []
                for angle in [0, 30, -30, 60, -60]:
                    path = f'{instance_id}_{label_name}_{angle}.jpg'
                    path = osp.join(self.root_path, scene_id, 'multi_view_pc', path)
                    all_path.append(path)
                data_list.append((all_path, label_id))
        
        return data_list
    
    def __getitem__(self, idx):
        paths, label = self.data_list[idx]
        image_features = []
        for path in paths:
            image = Image.open(path)
            image_features.append(self.transform(image))
        image_features = torch.stack(image_features, dim=0)

        return image_features, int(label)

if __name__ == '__main__':
    dataset = MultiViewPCDataset(
        root_path='/data/caidaigang/project/3DSSG_Repo/data/3RScan',
        data_list_path='/data/caidaigang/project/3DSSG_Repo/data/3RScan/train_scans_all_quanlity.txt',
        labels_path='/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/classes.txt',
    )
