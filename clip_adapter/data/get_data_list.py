import os
import clip
import torch
from PIL import Image
from tqdm import tqdm

def read_from_file(filename):
     with open(filename) as f:
         lines = f.readlines()
     return [line.strip() for line in lines]

def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


if __name__ == "__main__":
    model, preprocess = clip.load("RN101", device='cuda')

    classes = read_from_file('/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/classes.txt')
    text_emb = clip_classifier(classes, ['a photo of {}', 'a frame of {}',], model)
    res_croped, res_origin = [], []

    with open('/data/caidaigang/project/3DSSG_Repo/data/3RScan/val_all_quanlity.txt') as f:
        for i in tqdm(f):
            items = i.strip().split(':')
            scene_id = items[1].split(' ')[0]
            instance_id = items[2].split(' ')[0]
            label_name = ' '.join(items[3].split(' ')[0:-1])
            path_croped = f'instance_{instance_id}_class_{label_name}_croped_view'
            path_origin = f'instance_{instance_id}_class_{label_name}_view'
            label_id = classes.index(label_name)
            root = f'/data/caidaigang/project/3DSSG_Repo/data/3RScan/'
            files = os.listdir(os.path.join(root,f'{scene_id}/multi_view'))
            tmp_croped, tmp_origin = [], []
            for j in files:
                if path_origin in j and 'npy' not in j:
                    tmp_origin.append(os.path.join(f'{scene_id}/multi_view',j))
            
            # compute best view for each instance
            vision_feat = [preprocess(Image.open(os.path.join(root, t)).rotate(Image.ROTATE_270)) for t in tmp_croped]
            vision_feat = torch.stack(vision_feat, dim=0).cuda()
            vision_feat = model.encode_image(vision_feat)
            vision_feat = vision_feat / vision_feat.norm(dim=-1, keepdim=True)
            logits = (vision_feat @ text_emb)[:, label_id]
            res_croped.append(tmp_croped[logits.argmax().item()] +' '+str(label_id))
            res_origin.append(tmp_origin[logits.argmax().item()] +' '+str(label_id))

    with open('/data/caidaigang/project/3DSSG_Repo/clip_adapter/data/val_croped.txt', 'w') as f:
        for i in res_croped:
            f.write(i+'\n')

    with open('/data/caidaigang/project/3DSSG_Repo/clip_adapter/data/val_origin.txt', 'w') as f:
        for i in res_origin:
            f.write(i+'\n')
