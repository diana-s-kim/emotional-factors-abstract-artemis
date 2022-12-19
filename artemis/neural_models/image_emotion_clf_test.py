"""
Given an image guess a distribution over the emotion labels.

The MIT License (MIT)
Originally created in 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.notebook import tqdm as tqdm_notebook

from ..utils.stats import AverageMeter


class ImageEmotionClassifier(nn.Module):
    def __init__(self, img_encoder, clf_head):
        super(ImageEmotionClassifier, self).__init__()
        self.img_encoder = img_encoder
        self.clf_head = clf_head

    def __call__(self, img):
        feat = self.img_encoder(img)
        logits = self.clf_head(feat)
        return logits


def single_epoch_train(model, data_loader, criterion, optimizer, device):
    epoch_loss = AverageMeter()
    model.train()
    for batch in tqdm_notebook(data_loader):
        img = batch['image'].to(device)
        labels = batch['label'].to(device) # emotion_distribution
        logits = model(img)

        # Calculate loss
        loss = criterion(logits, labels)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        b_size = len(labels)
        epoch_loss.update(loss.item(), b_size)
    return epoch_loss.avg


@torch.no_grad()
def evaluate_on_dataset(model, data_loader, criterion, device, detailed=True, kl_div=True):
    epoch_loss = AverageMeter()
    model.eval()
    epoch_confidence = []
    for batch in tqdm_notebook(data_loader):
        img = batch['image'].to(device)
        labels = batch['label'].to(device) # emotion_distribution
        logits = model(img)

        # Calculate loss
        loss = criterion(logits, labels)

        if detailed:
            if kl_div:
                epoch_confidence.append(torch.exp(logits).cpu())  # logits are log-soft-max
            else:
                epoch_confidence.append(F.softmax(logits, dim=-1).cpu()) # logits are pure logits

        b_size = len(labels)
        epoch_loss.update(loss.item(), b_size)

    if detailed:
        epoch_confidence = torch.cat(epoch_confidence).numpy()

    return epoch_loss.avg, epoch_confidence


@torch.no_grad()
def accuracy_(model,data_loader,criterion,device,detailed=True,kl_div=False,k_top=(1,3,5)):#top1accuracy and top 5

    maxk = max(k_top)
    epoch_loss = [AverageMeter() for i in range(len(k_top))]
    model.eval()#model behavior
    epoch_confidence =[]

    for batch in tqdm_notebook(data_loader):
        img =batch['image'].to(device)
        emotions = batch['emotion'].to(device)
        logits = model(img)
        b_size = len(emotions)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(emotions.view(1, -1).expand_as(pred))


        for idx,k in enumerate(k_top):
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) #number
            accuracy_=torch.mul(correct_k,1.0)/b_size
            print("accuracy",accuracy_)
            epoch_loss[idx].update(accuracy_,b_size)
            print(epoch_loss[idx].avg)
        print("Top-1 Accuracy is ",epoch_loss[0].avg)
        print("Top-3 Accuracy is ",epoch_loss[1].avg)
        print("Top-5 Accuracy is ",epoch_loss[2].avg)

        if detailed:
            if kl_div:
                epoch_confidence.append(torch.exp(logits).cpu()) # logits
            else:
                epoch_confidence.append(F.softmax(logits, dim=-1).cpu()) #

    if detailed:
        epoch_confidence=torch.cat(epoch_confidence).numpy()

    return epoch_loss[0].avg,epoch_confidence

@torch.no_grad()
def collect_embedding_(model,data_loader,device):
    embedding_=[]
    index_=[]
    for batch in tqdm_notebook(data_loader):
        img = batch['image'].to(device)
        embedding_.append(model.img_encoder(img))
        index_.append(batch['index'])
        print(embedding_)
    return torch.cat(embedding_).cpu().numpy(),torch.cat(index_).cpu().numpy() #embedding,index


    
