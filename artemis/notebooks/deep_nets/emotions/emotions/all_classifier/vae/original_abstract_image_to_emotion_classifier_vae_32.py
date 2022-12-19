#!/usr/bin/env python
# coding: utf-8

# ## Use the emotion-histograms extracted with the corresponding notebook to train an image to an emotion (emotion-distribution) classifier.
# - the notebook to make the histograms is at __analysis/extract_emotion_histogram_per_image.ipynb__ 
# - if you do not want to train the model; you can load the pretrained one (set do_trainining=False)
# - the pretrained one is located at https://www.dropbox.com/s/8dfj3b36q15iieo/best_model.pt?dl=0

# #### Friendly Remarks from Panos :
#     - Predicting the emotional-responses without the text/explanations is very hard. 
#     - Predicting the emotional-responses given the explanations is much easier.  
#     - Predicting the emotional-responses given the text & image is not (significantly) easier, than relying only on text.
#     
# ###### <=> people can have very different emotional-reactions given an image. 
# 
# very fine-grained remarks:
#     - I did also train the image2emotion with "cleaner" data (e.g., drop images for which the emotion maximizer has less than 0.30 of total mass/votes). It does not make better predictions on test images w.r.t. the majority emotion.
#     - These are interesting stuff... if you want to play/work together let me know.

# In[1]:


import torch
import argparse
import pandas as pd
import os.path as osp
import numpy as np
from ast import literal_eval
#from plotly.offline import init_notebook_mode, iplot

from artemis.in_out.neural_net_oriented import torch_load_model, torch_save_model, save_state_dicts
from artemis.in_out.neural_net_oriented import image_emotion_distribution_df_to_pytorch_dataset
from artemis.in_out.basics import create_dir
from artemis.utils.visualization import plot_confusion_matrix
from artemis.emotions import ARTEMIS_EMOTIONS

from artemis.neural_models.mlp import MLP
from artemis.neural_models.resnet_encoder import ResnetEncoder
from artemis.neural_models.image_emotion_clf import VAE_emotion_classifier
from artemis.neural_models.image_emotion_clf import single_epoch_train, evaluate_on_dataset,accuracy_

#diana
#drop rate 0.3->0.1
num_objectives=32
KL_div=True

#init_notebook_mode()
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Load saved histograms of emotion choices as computed with "previous" notebook (see top-README if you are lost)
image_hists_file = './artemis/data/image-emotion-histogram.csv'
image_hists = pd.read_csv(image_hists_file)

# this literal_eval brings the saved string to its corresponding native (list) type
image_hists.emotion_histogram = image_hists.emotion_histogram.apply(literal_eval)

# normalize the histograms
image_hists.emotion_histogram = image_hists.emotion_histogram.apply(lambda x: (np.array(x) / float(sum(x))).astype('float32'))

print(f'Histograms corresponding to {len(image_hists)} images')


# #### In cell below you need to use YOUR PATHS.
# - I will use the pre-processed ArtEmis dataset; as prepared by the script __preprocess_artemis_data.py --preprocess-for-deep-nets True__ (see STEP.1 at top-README) 
# 
# - Specifically this way, I can utilize the same train/test/val splits accross all my neural-based experiments.

# In[3]:


artemis_preprocessed_dir = '/common/home/dsk101/Research/Emotion_Learneing/artemis_official/pre_processed_deep'
save_dir = './output/'
save_csv_dir ='./csv_save/all'
wikiart_img_dir = '/common/home/dsk101/Research/Emotion_Learneing/wikiart'

create_dir(save_dir)
checkpoint_file = osp.join(save_dir, 'all/vae_32/kl_all.pt')

# minor parameters
GPU_ID = 0 


# In[4]:


## Prepare the artemis dataset (merge it with the emotion-histograms.)
artemis_data = pd.read_csv(osp.join(artemis_preprocessed_dir, 'artemis_preprocessed.csv'))
print('Annotations loaded:', len(artemis_data))

## keep each image once.
artemis_data = artemis_data.drop_duplicates(subset=['art_style', 'painting'])
artemis_data.reset_index(inplace=True, drop=True)

# keep only relevant info + merge
artemis_data = artemis_data[['art_style', 'painting', 'split','emotion_label']]
artemis_data = artemis_data.merge(image_hists)
artemis_data = artemis_data.rename(columns={'emotion_histogram': 'emotion_distribution'})

n_emotions = artemis_data['emotion_label'].max()+1
print('Using {} emotion-classes.'.format(n_emotions))
assert all(image_hists.emotion_histogram.apply(len) == n_emotions)


# In[5]:


# to see the emotion_distribution column
artemis_data.head()

#save csv
artemis_data.to_csv(save_csv_dir+"/vae_32/kl_all.csv")
# In[6]:


parser = argparse.ArgumentParser() # use for convenience instead of say a dictionary
args = parser.parse_args([])

# deep-net data-handling params. note if you want to reuse this net with neural-speaker 
# it makes sense to keep some of the (image-oriented) parameters the same accross the nets.
args.lanczos = True
args.img_dim = 256
args.num_workers = 8
args.batch_size = 128
args.gpu_id = 0

args.img_dir = wikiart_img_dir


# In[7]:


## prepare data
data_loaders, datasets = image_emotion_distribution_df_to_pytorch_dataset(artemis_data, args)


# In[8]:


## Prepate the Neural-Net Stuff (model, optimizer etc.)
## This is what I used for the paper with minimal hyper-param-tuning. You can use different nets/configs here...


# In[8]:


device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
criterion = torch.nn.KLDivLoss(reduction='batchmean').to(device)

img_encoder = ResnetEncoder('resnet34', adapt_image_size=1).unfreeze(level=7, verbose=True)
img_emb_dim = img_encoder.embedding_dimension()

# here we make an MLP closing with LogSoftmax since we want to train this net via KLDivLoss
clf_objective = MLP(img_emb_dim, [256,128,64,num_objectives], dropout_rate=0.1, b_norm=True, closure=None)
principle_axes=torch.from_numpy(np.load("./artemis/notebooks/deep_nets/emotions/abstract_classifier/vae/emotion_principle_axes/ortho_128.npz")['orthogonal_set_128']).to(device)
model = VAE_emotion_classifier(img_encoder,clf_objective,principle_axes,num_objectives,KL_div).to(device)
optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': 5e-4}])



# In[9]:


## helper function.
## to evaluate how well the model does according to the class that it finds most likely
## note it only concerns the predictions on examples (images) with a single -unique maximizer- emotion
def evaluate_argmax_prediction(dataset, guesses):
    labels = dataset.labels
    labels = np.vstack(labels.to_numpy())
    unique_max = (labels == labels.max(1, keepdims=True)).sum(1) == 1
    umax_ids = np.where(unique_max)[0]
    gt_max = np.argmax(labels[unique_max], 1)
    max_pred = np.argmax(guesses[umax_ids], 1)
    return (gt_max == max_pred).mean()


# In[10]:


do_training = True# set to True, if you are not using a pretrained model
max_train_epochs = 25
no_improvement = 0
min_eval_loss = np.Inf

if do_training:
    for epoch in range(1, max_train_epochs+1):
        train_loss = single_epoch_train(model, data_loaders['train'], criterion, optimizer, device)
        print('Train Loss: {:.3f}'.format(train_loss))

        eval_loss, _ =         evaluate_on_dataset(model, data_loaders['val'], criterion, device, detailed=False)
        print('Eval Loss: {:.3f}'.format(eval_loss))

        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            no_improvement = 0
            print('Epoch {}. Validation loss improved!'.format(epoch))
            torch_save_model(model.state_dict(), checkpoint_file)
                
            test_loss, test_confidence =             evaluate_on_dataset(model, data_loaders['test'], criterion, device, detailed=True)
            print('Test Loss: {:.3f}'.format(test_loss))                

            dataset = data_loaders['test'].dataset        
            arg_max_acc = evaluate_argmax_prediction(dataset, test_confidence)
            print('Test arg_max_acc: {:.3f}'.format(arg_max_acc))
        else:
            no_improvement += 1
        
        if no_improvement >=5 :
            print('Breaking at epoch {}. Since for 5 epoch we observed no (validation) improvement.'.format(epoch))
            break


# ### Below is rudimentary analysis of the trained system.

# In[ ]:


load_best_model = True

if not do_training or load_best_model:
    model.load_state_dict(torch.load(checkpoint_file))
    test_accuracy, test_confidence = accuracy_(model, data_loaders['test'], criterion, device, detailed=True)
    print('test accuracy',test_accuracy)




# In[18]:


## how often the most & second most, predicted emotions are positive vs. negative?
preds = torch.from_numpy(test_confidence)
top2 = preds.topk(2).indices
has_pos = torch.any(top2 <= 3, -1)
has_neg = torch.any((top2 >=4) & (top2 !=8), -1)
has_else = torch.any(top2 == 8, -1)
pn = (has_pos & has_neg).double().mean().item()
pne = ((has_pos & has_neg) | (has_pos & has_else) | (has_neg & has_else)).double().mean().item()
print('The classifier finds the 1st/2nd most likely emotions to be negative/positive, or contain something-else')
print(pn, pne)


# In[19]:


# How well it does on test images that have strong majority in emotions?
labels = data_loaders['test'].dataset.labels
labels = np.vstack(labels.to_numpy())



for use_strong_domi in [True, False]:
    print('use_strong_domi:', use_strong_domi)
    if use_strong_domi:
        dominant_max = (labels.max(1) > 0.5)
    else:
        dominant_max = (labels.max(1) >= 0.5)

    umax_ids = np.where(dominant_max)[0]
    gt_max = np.argmax(labels[dominant_max], 1)
    max_pred = np.argmax(test_confidence[umax_ids], 1)    

    print('Test images with dominant majority', dominant_max.mean())
    print('Guess-correctly', (gt_max == max_pred).mean(), '\n')


# In[26]:


#iplot(plot_confusion_matrix(ground_truth=gt_max, predictions=max_pred, labels=ARTEMIS_EMOTIONS))


# In[27]:


# For the curious one. Images where people "together" aggree on anger are rare. Why?
#iplot(plot_confusion_matrix(ground_truth=gt_max, predictions=max_pred, labels=ARTEMIS_EMOTIONS, normalize=False))

