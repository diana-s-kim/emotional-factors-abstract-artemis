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
from artemis.in_out.neural_net_oriented import image_emotion_label_df_to_pytorch_dataset
from artemis.in_out.basics import create_dir
from artemis.utils.visualization import plot_confusion_matrix
from artemis.emotions import ARTEMIS_EMOTIONS

from artemis.neural_models.mlp import MLP
from artemis.neural_models.resnet_encoder import ResnetEncoder
from artemis.neural_models.image_emotion_clf import VAE_emotion_embedding
from artemis.neural_models.image_emotion_clf import single_epoch_train, evaluate_on_dataset,accuracy_,collect_embedding_

#diana
#drop rate 0.3->0.1
num_objectives=16
KL_div=False
do_training=False

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
save_dir = './output'
embedding_dir='./embedding'
save_csv_dir ='./csv_save/abstract/softmax_abstract'
wikiart_img_dir = '/common/home/dsk101/Research/Emotion_Learneing/wikiart'

create_dir(save_dir)
checkpoint_file = osp.join(save_dir, 'abstract/vae_16/softmax_abstract.pt')

# minor parameters
GPU_ID = 0 


# In[4]:


## Prepare the artemis dataset (merge it with the emotion-histograms.)
artemis_data = pd.read_csv(osp.join(artemis_preprocessed_dir, 'artemis_preprocessed.csv'))
print('Annotations loaded:', len(artemis_data))

##keep each image once.(allow duplication)
artemis_data = artemis_data.drop_duplicates(subset=['art_style', 'painting'])
artemis_data.reset_index(inplace=True, drop=True)

# keep only relevant info + merge
artemis_data = artemis_data[['art_style','painting', 'split','emotion_label']]
#without something else
#artemis_data = artemis_data[artemis_data['emotion_label']!=8]
#abstract paintings#
artemis_data = artemis_data[artemis_data['art_style'].isin(['Abstract_Expressionism','Cubism','Color_Field_Painting','Minimalism','Action_painting','Synthetic_Cubism','Analytical_Cubism'])]
artemis_data = artemis_data.merge(image_hists)
artemis_data = artemis_data.rename(columns={'emotion_histogram': 'emotion_distribution'})
n_emotions = artemis_data['emotion_label'].max()+1


print('Using {} emotion-classes.'.format(n_emotions))
assert all(image_hists.emotion_histogram.apply(len) == n_emotions)


# In[5]:


# to see the emotion_distribution column
artemis_data.head()

#save csv
artemis_data['dominant_emotion']=artemis_data['emotion_distribution'].apply(np.argmax)
artemis_data.to_csv(save_csv_dir+"/softmax_abstract.csv",float_format='%.3f')


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
data_loaders, datasets = image_emotion_label_df_to_pytorch_dataset(artemis_data, args,do_training)#label feeding instead of distribution


# In[8]:


## Prepate the Neural-Net Stuff (model, optimizer etc.)
## This is what I used for the paper with minimal hyper-param-tuning. You can use different nets/configs here...


# In[8]:


device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)#torch.nn.KLDivLoss(reduction='batchmean').to(device)

img_encoder = ResnetEncoder('resnet34', adapt_image_size=1).unfreeze(level=7, verbose=True) 
img_emb_dim = img_encoder.embedding_dimension()

# here we make an MLP closing with LogSoftmax since we want to train this net via KLDivLoss
clf_objective = MLP(img_emb_dim, [256,128,64,num_objectives], dropout_rate=0.1, b_norm=True, closure=None)
principle_axes=torch.from_numpy(np.load("./artemis/notebooks/deep_nets/emotions/abstract_classifier/vae/emotion_principle_axes/ortho_128.npz")['orthogonal_set_128']).to(device)
model = VAE_emotion_embedding(img_encoder,clf_objective,principle_axes,num_objectives,KL_div).to(device)


# collect embedding
model.load_state_dict(torch.load(checkpoint_file))
embedding_,index_=collect_embedding_(model,data_loaders['train'],device)
np.savez(embedding_dir+"/abstract/vae_16/vae_16_train_embedding_softmax.npz",embedding=embedding_,index=index_)
print("finished!")
