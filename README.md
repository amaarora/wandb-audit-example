# wandb-audit-example

This is a simple example to showcase Weights and Biases artifacts. As part of this example, we train EfficientNet-B3 on Imagenette Dataset.

## Prepare Dataset
To prepare the dataset, run the following lines of code in the root folder of this directory:

```
mkdir data && cd data 
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xvf imagenette2-160.tgz
```

Now you should have a data directory in the repository whose folder structure looks like:
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

## Start Model Training 
From the root of this folder, simply run: 
```
python src/train.py
```
