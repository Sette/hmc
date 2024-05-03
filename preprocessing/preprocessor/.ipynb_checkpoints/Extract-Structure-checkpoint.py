#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import tensorflow as tf
import json
import ast
import os
import csv
import math
from sklearn.utils import shuffle
from math import ceil
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# In[5]:


tqdm.pandas()


# In[6]:


args = pd.Series({
    "root_dir":"/mnt/disks/data/",
    "dataset_path":"/mnt/disks/data/fma/fma_large",
    "embeddings":"music_style",
    "sequence_size": 1280,
    "train_id": "hierarchical_all",
    'sample_size': 1
})


# In[7]:


base_path = "/mnt/disks/data/fma/trains"


job_path = os.path.join(base_path,args.train_id)


tfrecord_path = os.path.join(job_path,"tfrecords")

# In[16]:

base_path = os.path.join(args.root_dir,"fma")

# In[17]:

models_path = os.path.join(args.root_dir,"models")


metadata_path_fma = os.path.join(base_path,"fma_metadata")

# In[18]:

metadata_path = os.path.join(job_path,"metadata.json")


categories_labels_path = os.path.join(job_path,"labels.json")


# In[8]:


def __load_json__(path):
    with open(path, 'r') as f:
        tmp = json.loads(f.read())

    return tmp




# In[9]:


def create_dir(path):
    # checking if the directory demo_folder2 
    # exist or not.
    if not os.path.isdir(path):

        # if the demo_folder2 directory is 
        # not present then create it.
        os.makedirs(path)
    return True



# In[10]:


import shutil
shutil.rmtree(job_path)


# In[11]:


create_dir(job_path)


# ## Load genres file. Contains relationships beetwen genres

# In[12]:


genres_df = pd.read_csv(os.path.join(metadata_path_fma,'genres.csv'))


# In[13]:


genres_df


# In[14]:


genres_df[genres_df['genre_id'] == 495]


# ## Análise do tracks.csv

# In[15]:


tracks = os.path.join(metadata_path_fma,"tracks_valid.csv")


# In[16]:


df_tracks = pd.read_csv(tracks)


# In[131]:


df_tracks["track_genres"] = df_tracks.track_genres.apply(lambda x : ast.literal_eval(x))


# In[132]:


genres_df['genre_id'] = genres_df.genre_id.astype(int)


# In[133]:


genres_df['parent'] = genres_df.parent.astype(int)


# ## Join com tabela de generos

# ### Geração da hierarquia a partir das tracks
# 

# In[134]:


def get_structure(genres_id, df_genres):
    def get_from_df(genre_id, df_genres):
        if genre_id != 0:
            parent_genre = df_genres[df_genres["genre_id"]
                                     == genre_id].parent.values[0]
            return [genre_id, get_from_df(parent_genre, df_genres)]
    for genre_id in genres_id:
        print(get_from_df(genre_id, df_genres))


# In[135]:


def get_structure(genres_id, df_genres):
    def get_from_df(genre_id, df_genres, output=[]):
        if genre_id != 0:
            parent_genre = df_genres[df_genres["genre_id"]
                                     == genre_id].parent.values[0]
            output.append(genre_id)
            get_from_df(parent_genre, df_genres, output=output)
            return output
    output_list = []
    for genre_id in genres_id:
        output_list.append(get_from_df(genre_id, df_genres, output=[]))
    return output_list


# In[136]:


genres_df


# In[146]:


get_structure([27,1032], genres_df)
get_structure([98,1032], genres_df)


# In[138]:


df_tracks.track_genres.iloc[256]


# In[139]:


df_tracks.iloc[158]


# In[141]:


df_tracks["full_genre_id"] = df_tracks.track_genres.progress_apply(lambda x : get_structure(x,genres_df))


# In[67]:


df_tracks = df_tracks[['track_id','full_genre_id']]


# In[17]:


df_tracks.to_csv(os.path.join(metadata_path_fma,'tracks_valid.csv'),index=False)

