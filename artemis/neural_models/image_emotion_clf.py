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


class Emotion_embedding(nn.Module):
    def __init__(self, img_encoder, clf_head):
        super(Emotion_embedding, self).__init__()
        self.img_encoder = img_encoder
        self.clf_head = clf_head

    def __call__(self, img):
        feat = self.img_encoder(img)
        return feat



class VAE_emotion_classifier(nn.Module):
    def __init__(self,img_encoder,clf_objective,principle_axes,num_objectives,KL_div):
        super(VAE_emotion_classifier,self).__init__()
        self.dim_embedding=128
        self.rank=2
        self.num_objectives=num_objectives
        self.KL_div=KL_div
        self.KL_softmax=torch.nn.LogSoftmax(dim=-1)
        self.img_encoder=img_encoder
        self.orthogonal_set=principle_axes
        self.clf_head_objective=clf_objective
        self.RBF_clf=nn.Linear(num_objectives,9,bias=False)

        self.mu_=torch.nn.Parameter(torch.rand(num_objectives,self.dim_embedding))
        self.var_=torch.nn.Parameter(torch.rand(num_objectives,self.rank))
        self.RBF_var=torch.nn.Parameter(torch.rand(self.dim_embedding,num_objectives))
        
    def __call__(self,img):
        res_feat=self.img_encoder(img)
        principle_pick=F.gumbel_softmax(self.clf_head_objective(res_feat),tau=0.2)

        principle_mean=torch.matmul(principle_pick,self.mu_)
        principle_variance=torch.matmul(principle_pick,self.var_)
        transformer=torch.matmul(self.orthogonal_set.float(),torch.diag_embed(F.pad(torch.square(principle_variance),(0,self.dim_embedding-self.rank),mode='constant',value=0)))

        #Generation Gaussian
        normal_sample=torch.normal(torch.zeros(img.shape[0],self.dim_embedding),torch.ones(img.shape[0],self.dim_embedding)).cuda() #unit
        normal_embedding=torch.matmul(transformer,normal_sample.unsqueeze(-1)).squeeze()+principle_mean

        #RBF
        dist_to_mu=torch.unsqueeze(normal_embedding,dim=-1).repeat(1,1,self.num_objectives)-self.mu_.transpose(0,1)
        RBF_positive_var=torch.mul(self.RBF_var,self.RBF_var)
        Gaussain_E=torch.matmul(dist_to_mu.transpose(1,2),torch.mul(RBF_positive_var,dist_to_mu)).diagonal(0,dim1=-2,dim2=-1)

        #last RBM recognition
        logits=self.RBF_clf(Gaussain_E)

        #KL_div only when
        if self.KL_div:
            logits=self.KL_softmax(logits)

        return logits
        


class VAE_emotion_embedding(nn.Module):
    def __init__(self,img_encoder,clf_objective,principle_axes,num_objectives,KL_div):
        super(VAE_emotion_embedding,self).__init__()
        self.dim_embedding=128
        self.rank=2
        self.num_objectives=num_objectives
        self.KL_div=KL_div
        self.KL_softmax=torch.nn.LogSoftmax(dim=-1)
        self.img_encoder=img_encoder
        self.orthogonal_set=principle_axes
        self.clf_head_objective=clf_objective
        self.RBF_clf=nn.Linear(num_objectives,9,bias=False)

        self.mu_=torch.nn.Parameter(torch.rand(num_objectives,self.dim_embedding))
        self.var_=torch.nn.Parameter(torch.rand(num_objectives,self.rank))
        self.RBF_var=torch.nn.Parameter(torch.rand(self.dim_embedding,num_objectives))
        
    def __call__(self,img):
        res_feat=self.img_encoder(img)
        principle_pick=F.gumbel_softmax(self.clf_head_objective(res_feat),tau=0.2)
        which_factor=torch.unsqueeze(torch.argmax(principle_pick,dim=1),1)
        print(which_factor)
        principle_mean=torch.matmul(principle_pick,self.mu_)
        principle_variance=torch.matmul(principle_pick,self.var_)
        transformer=torch.matmul(self.orthogonal_set.float(),torch.diag_embed(F.pad(torch.square(principle_variance),(0,self.dim_embedding-self.rank),mode='constant',value=0)))

        #Generation Gaussian
        normal_sample=torch.normal(torch.zeros(img.shape[0],self.dim_embedding),torch.ones(img.shape[0],self.dim_embedding)).cuda() #unit
        normal_embedding=torch.matmul(transformer,normal_sample.unsqueeze(-1)).squeeze()+principle_mean

        return torch.cat((normal_embedding,which_factor),1)
        

class VAE_random_embedding(nn.Module):
    def __init__(self,img_encoder,clf_objective,principle_axes,num_objectives,KL_div):
        super(VAE_random_embedding,self).__init__()
        self.dim_embedding=128
        self.rank=2
        self.num_objectives=num_objectives
        self.KL_div=KL_div
        self.KL_softmax=torch.nn.LogSoftmax(dim=-1)
        self.img_encoder=img_encoder
        self.orthogonal_set=principle_axes
        self.clf_head_objective=clf_objective
        self.RBF_clf=nn.Linear(num_objectives,9,bias=False)

        self.mu_=torch.nn.Parameter(torch.rand(num_objectives,self.dim_embedding))
        self.var_=torch.nn.Parameter(torch.rand(num_objectives,self.rank))
        self.RBF_var=torch.nn.Parameter(torch.rand(self.dim_embedding,num_objectives))
        
    def __call__(self,img):
        random_int=torch.randint(0,self.num_objectives,(img.shape[0],))# (128) batch
        random_pick=F.one_hot(random_int,num_classes=self.num_objectives).float().cuda()
        print(random_pick)
        #principle_pick=F.gumbel_softmax(self.clf_head_objective(random_pick),tau=0.2)

        principle_mean=torch.matmul(random_pick,self.mu_)
        principle_variance=torch.matmul(random_pick,self.var_)
        transformer=torch.matmul(self.orthogonal_set.float(),torch.diag_embed(F.pad(torch.square(principle_variance),(0,self.dim_embedding-self.rank),mode='constant',value=0)))

        #Generation Gaussian
        normal_sample=torch.normal(torch.zeros(img.shape[0],self.dim_embedding),torch.ones(img.shape[0],self.dim_embedding)).cuda() #unit
        normal_embedding=torch.matmul(transformer,normal_sample.unsqueeze(-1)).squeeze()+principle_mean
        return normal_embedding
        


        
def single_epoch_train(model, data_loader, criterion, optimizer,device):
    epoch_loss = AverageMeter()
    model.train()
    for batch in tqdm_notebook(data_loader):
        img = batch['image'].to(device)
        labels = batch['label'].to(device) # emotion_distribution
        logits = model(img)

        # Calculate loss
        loss = criterion(logits, labels)
        print("loss: {:.3f}".format(loss.item()))

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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) #number
            accuracy_=torch.mul(correct_k,1.0)/b_size
            epoch_loss[idx].update(accuracy_,b_size)


        print("Top-1 Accuracy is {:.3f} ".format(epoch_loss[0].avg.item()))
        print("Top-3 Accuracy is {:.3f} ".format(epoch_loss[1].avg.item()))
        print("Top-5 Accuracy is {:.3f} ".format(epoch_loss[2].avg.item()))

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
        embedding_.append(model(img))
        index_.append(batch['index'])
        print(embedding_)
    return torch.cat(embedding_).cpu().numpy(),torch.cat(index_).cpu().numpy()
