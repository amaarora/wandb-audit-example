import os
import torch
import argparse

import torchvision
from config import Config
import timm
import torch.nn as nn
from tqdm import tqdm
from wandb_checkpoint import CheckpointSaver
import wandb
import logging 
from timm.utils.log import setup_default_logging

_logger = logging.getLogger('train')

# test

def train_fn(model, train_data_loader, optimizer, epoch):
    model.train()
    fin_loss = 0.0
    tk = tqdm(train_data_loader, desc="Epoch" + " [TRAIN] " + str(epoch + 1))

    for t, data in enumerate(tk):
        data[0] = data[0].to('cuda')
        data[1] = data[1].to('cuda')

        optimizer.zero_grad()
        out = model(data[0])
        loss = nn.CrossEntropyLoss()(out, data[1])
        loss.backward()
        optimizer.step()

        fin_loss += loss.item()
        tk.set_postfix(
            {
                "loss": "%.6f" % float(fin_loss / (t + 1)),
                "LR": optimizer.param_groups[0]["lr"],
            }
        )
    return fin_loss / len(train_data_loader), optimizer.param_groups[0]["lr"]


def eval_fn(model, eval_data_loader, epoch):
    model.eval()
    fin_loss = 0.0
    tk = tqdm(eval_data_loader, desc="Epoch" + " [VALID] " + str(epoch + 1))

    with torch.no_grad():
        for t, data in enumerate(tk):
            data[0] = data[0].to('cuda')
            data[1] = data[1].to('cuda')
            out = model(data[0])
            loss = nn.CrossEntropyLoss()(out, data[1])
            fin_loss += loss.item()
            tk.set_postfix({"loss": "%.6f" % float(fin_loss / (t + 1))})
        return fin_loss / len(eval_data_loader)


def train(args=None, wandb_run=None):
    # train and eval datasets
    train_dataset = torchvision.datasets.ImageFolder(
        Config["TRAIN_DATA_DIR"], transform=Config["TRAIN_AUG"]
    )
    eval_dataset = torchvision.datasets.ImageFolder(
        Config["TEST_DATA_DIR"], transform=Config["TEST_AUG"]
    )

    # train and eval dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config["BS"],
        shuffle=True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=Config["BS"],
    )

    # model
    model = timm.create_model(Config["MODEL"], pretrained=Config["PRETRAINED"])
    model = model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config["LR"])

    # setup checkpoint saver
    saver = CheckpointSaver(model=model, optimizer=optimizer, args=args, decreasing=True, 
                        wandb_run=wandb_run, max_history=args.num_checkpoints)

    for epoch in range(Config["EPOCHS"]):
        avg_loss_train, lr = train_fn(
            model, train_dataloader, optimizer, epoch
        )
        avg_loss_eval = eval_fn(model, eval_dataloader, epoch)
        saver.save_checkpoint(epoch, metric=avg_loss_eval)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='artifacts', required=False)
    parser.add_argument('--num-checkpoints', type=int, default=2, required=False)
    args = parser.parse_args()

    setup_default_logging()

    run = wandb.init(project=args.project)

    train(args=args, wandb_run=run)
