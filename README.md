## :book: VL-SAT: Visual-Linguistic Semantics Assisted Training for 3D Semantic Scene Graph Prediction in Point Cloud (CVPR 2023)
<image src="demo.png" width="100%">
<p align="center">
  <small>:fire: If you found the training scheme in VL-SAT is useful, please help to :star: it or recommend it to your friends. Thanks:fire:</small>
</p>


# Introduction
This is a release of the code of our paper **_VL-SAT: Visual-Linguistic Semantics Assisted Training for 3D Semantic Scene Graph Prediction in Point Cloud_** (CVPR 2023).

Authors:
Ziqin Wang,
Bowen Cheng,
Lichen Zhao,
Dong Xu,
Yang Tang,
Lu Sheng* (*corresponding
author)

[[arxiv]](https://arxiv.org/pdf/2303.14408.pdf)  [[code]](https://github.com/wz7in/CVPR2023-VLSAT)

# Dependencies
```bash
conda create -n vlsat python=3.8
conda activate vlsat
pip install -r requirement.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-geometric
pip install git+https://github.com/openai/CLIP.git
```
# Prepare the data
A. Download 3Rscan and 3DSSG-Sub Annotation, you can follow [3DSSG](https://github.com/ShunChengWu/3DSSG#preparation)

B. Generate 2D Multi View Image
```bash
# you should motify the path in pointcloud2image.py into your own path
python data/pointcloud2image.py
```

C. You should arrange the file location like this
```
data
  3DSSG_subset
    relations.txt
    classes.txt
    
  3RScan
    0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca
      multi_view
      labels.instances.align.annotated.v2.ply
    ...  
      
```

D. Train your own clip adapter 

``` python clip_adapter/main.py ```

or just use the checkpoint 

``` clip_adapter/checkpoint/origin_mean.pth ```

# Run Code
```bash
# Train
python -m main --mode train --config <config_path> --exp <exp_name>
# Eval
python -m main --mode eval --config <config_path> --exp <exp_name>
```

# Paper

If you find the code useful please consider citing our [paper](https://arxiv.org/pdf/2303.14408.pdf):
```
@article{wang2023vl,
  title={VL-SAT: Visual-Linguistic Semantics Assisted Training for 3D Semantic Scene Graph Prediction in Point Cloud},
  author={Wang, Ziqin and Cheng, Bowen and Zhao, Lichen and Xu, Dong and Tang, Yang and Sheng, Lu},
  journal={arXiv preprint arXiv:2303.14408},
  year={2023}
}
```


# Acknowledgement
This repository is partly based on [3DSSG](https://github.com/ShunChengWu/3DSSG) and [CLIP](https://github.com/openai/CLIP) repositories.