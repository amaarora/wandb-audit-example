# Checkpoint Saver example with W&B

This repository contains working code to train on ImageNette using in PyTorch.


To be able to run the scripts, please run the following commands first from the root directory of this repository to download the data: 

```
mkdir data && cd data 
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xvf imagenette2-160.tgz
```

Now you should have a `data` directory in the repository whose folder structure looks like:

```
data/
└── imagenette2-160
    ├── train
    │   ├── n01440764
    │   ├── n02102040
    │   ├── n02979186
    │   ├── n03000684
    │   ├── n03028079
    │   ├── n03394916
    │   ├── n03417042
    │   ├── n03425413
    │   ├── n03445777
    │   └── n03888257
    └── val
        ├── n01440764
        ├── n02102040
        ├── n02979186
        ├── n03000684
        ├── n03028079
        ├── n03394916
        ├── n03417042
        ├── n03425413
        ├── n03445777
        └── n03888257
```

## Launch training using PyTorch 
To launch training using PyTorch, run the following command from the root folder of this repository: 

```
python src/train.py 
```