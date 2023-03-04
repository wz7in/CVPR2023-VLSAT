import torch
import numpy as np
import torch.nn as nn
from clip import clip

class AdapterModel(torch.nn.Module):
    
    def __init__(self, input_size=512, output_size=512, alpha=0.5):
        super(AdapterModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_size)
        self.init_parameters()
    
    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        res_feat = x.clone()
        global_feat = self.fc1(x)
        global_feat = self.relu(global_feat)
        view_feat = self.fc2(global_feat)
        image_feat = self.alpha * view_feat + (1 - self.alpha) * res_feat

        return image_feat

