import torch
from dataset import MultiViewDataset
from torch.utils.data import DataLoader
from model import AdapterModel
from clip import clip
import numpy as np
import torch.nn.functional as F

model, preprocess = clip.load("RN101", device='cuda')

# freeze the model
for name, param in model.named_parameters():
    if 'text_adapter' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

epochs = 20

def smooth_loss(pred, gold):
    eps = 0.2
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1).mean()
    return loss

def train():
    dataset = MultiViewDataset(
        root_path='/data/caidaigang/project/3DSSG_Repo/data/3RScan',
        data_list_path='/data/caidaigang/project/3DSSG_Repo/data/3RScan/train_scans_all_quanlity.txt',
        labels_path='/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/classes.txt',
        mode='origin_view_mean_2'  # ['croped_view_mean', 'origin_view_mean', 'clip_view_mean']
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    adapter = AdapterModel(input_size=512, output_size=512, alpha=0.6, clip_model=model, labels=dataset.labels).cuda()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=1e-2, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * epochs)
    best = 0
    
    for epoch in range(epochs):
        res = []
        adapter.train()
        for i, (data, label) in enumerate(dataloader):
            data = data.cuda()
            label = label.cuda()
            logits = adapter(data)
            loss = smooth_loss(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_list = torch.where(logits.argsort(descending=True) == label.reshape(-1,1))[1].cpu().numpy()
            res.extend(acc_list)
            scheduler.step()
            # if i % 10 == 0:
            #     topk1 = 100.0 * (acc_list < 1).sum() / len(acc_list)
            #     topk5 = 100.0 * (acc_list < 5).sum() / len(acc_list)
            #     topk10 = 100.0 * (acc_list < 10).sum() / len(acc_list)
            #     print(f"epoch: {epoch}, step: {i}, loss: {loss.item()} , topk1: {topk1}, topk5: {topk5}, topk10: {topk10}")
        res = np.array(res)
        print(f"============ epoch: {epoch} ============")
        print(f"topk1: {100.0 * (res < 1).sum() / len(res)}")
        print(f"topk5: {100.0 * (res < 5).sum() / len(res)}")
        print(f"topk10: {100.0 * (res < 10).sum() / len(res)}")
        res = eval(adapter)

        if res > best:
            best = res
            torch.save(adapter.state_dict(), f"/data/caidaigang/project/3DSSG_Repo/clip_adapter/checkpoint/clip_mean_best.pth")
        print(f"best: {best}")
    
    print('done')

def eval(adapter):
    dataset = MultiViewDataset(
        root_path='/data/caidaigang/project/3DSSG_Repo/data/3RScan',
        data_list_path='/data/caidaigang/project/3DSSG_Repo/data/3RScan/val_all_quanlity.txt',
        labels_path='/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/classes.txt',
        mode='origin_view_mean'  # ['croped_view_mean', 'origin_view_mean', 'clip_view_mean']
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    #adapter = AdapterModel(input_size=512, output_size=512, alpha=0.6, clip_model=model, labels=dataset.labels)
    #adapter.load_state_dict(torch.load(f"/data/caidaigang/project/3DSSG_Repo/clip_adapter/checkpoint/clip_mean_best_vit.pth",'cpu'))
    #adapter = adapter.cuda()
    adapter.eval()
    res = []
    for i, (data, label) in enumerate(dataloader):
        data = data.cuda()
        label = label.cuda()
        with torch.no_grad():
            logits = adapter(data)
        acc_list = torch.where(logits.argsort(descending=True) == label.reshape(-1,1))[1].cpu().numpy()
        res.extend(acc_list)
    res = np.array(res)
    print("============ val ============")
    print(f"topk1: {100.0 * (res < 1).sum() / len(res)}")
    print(f"topk5: {100.0 * (res < 5).sum() / len(res)}")
    print(f"topk10: {100.0 * (res < 10).sum() / len(res)}")

    return 100.0 * (res < 1).sum() / len(res)

def get_label(labels_path):
    with torch.no_grad():
        label_list = []
        with open(labels_path, "r") as f:
            data = f.readlines()
        for line in data:
            label_list.append(line.strip())
        text = torch.cat([clip.tokenize(f"there is {c} in scene") for c in label_list]).cuda()

    return text

if __name__ == "__main__":
    train()
    #eval()