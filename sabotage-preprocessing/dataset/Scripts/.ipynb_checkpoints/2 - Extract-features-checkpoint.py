#!/usr/bin/env python
# coding: utf-8

# In[21]:

import os
import csv
import json
import pandas as pd
import numpy as np
import tensorflow as tf

from math import ceil
from joblib import Parallel, delayed

import multiprocessing
from tqdm import tqdm

from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs



# In[11]:



# In[12]:


tqdm.pandas()


# In[13]:


args = pd.Series({
    "root_dir":"/mnt/disks/data/",
    "dataset_path":"/mnt/disks/data/fma/fma_large",
    "embeddings":"music_style"
})


# In[14]:


# In[16]:


base_path = os.path.join(args.root_dir,"fma")


# In[17]:


models_path = os.path.join(args.root_dir,"models")


metadata_path_fma = os.path.join(base_path,"fma_metadata")



# In[15]:


if args.embeddings == "music_style":
    model_path = os.path.join(models_path,args.embeddings,"discogs-effnet-bs64-1.pb")



# In[22]:


df = pd.read_csv(os.path.join(metadata_path_fma,"tracks_valid.csv"))


# In[23]:


model = TensorflowPredictEffnetDiscogs(graphFilename=model_path,output="PartitionedCall:1")


# In[24]:


def create_dir(path):
    # checking if the directory demo_folder2 
    # exist or not.
    if not os.path.isdir(path):

        # if the demo_folder2 directory is 
        # not present then create it.
        os.makedirs(path)
    return True


def extract_feature(file_path,model):
    ### Configuração do model para extrair a representação do aúdio
    # model = TensorflowPredictEffnetDiscogs(graphFilename=model_path)
    audio = MonoLoader(filename=file_path, sampleRate=16000)()
    activations = model(audio)
    return activations


# In[25]:


def find_path(track_id,dataset_path):
    track_id = track_id.zfill(6)
    folder_id = track_id[0:3]
    file_path = os.path.join(dataset_path,folder_id,track_id+'.mp3')
    return file_path
    


# In[36]:


df['file_path'] = df.track_id.apply(lambda x: find_path(str(x),args.dataset_path))



# In[26]:


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


def parse_single_music(data,music):
    # cat1, cat2, cat3, cat4, cat5 = data
    track_id, track_title, valid_genre, file_path = data
    
    #define the dictionary -- the structure -- of our single example
    data = {
        'emb' : _bytes_feature(serialize_array(music)),
        'track_id' : _int64_feature(track_id)
    }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out
# In[45]:


# In[45]:



# In[24]:

def process_df(df,i,count,batch_size,tfrecords_path):
    batch_df = df[i:i+batch_size]
        
    tqdm.pandas()

    X = batch_df.file_path.progress_apply(lambda x: extract_feature(x,model))   

    print("Extraiu as features")


    tfrecords = [parse_single_music(data, x) for data, x in zip(batch_df.values, X)]

    path = os.path.join(tfrecords_path,f"{str(count).zfill(10)}.tfrecord")

    #with tf.python_io.TFRecordWriter(path) as writer:
    with tf.io.TFRecordWriter(path) as writer:
        for tfrecord in tfrecords:
            writer.write(tfrecord.SerializeToString())

    print(f"{count} {len(tfrecords)} {path}")


def generate_tf_records(df,model,filename="train"):
    
    tfrecords_path = os.path.join(args.dataset_path,"tfrecords",filename)
    
    create_dir(tfrecords_path)
    
    
    batch_size = 1024 * 10  # 10k records from each file batch
    count = 0
    total = ceil(len(df) / batch_size)
    
    with Parallel(n_jobs=10, require='sharedmem') as para:
        print("Estamos usando paralelismo!!!")
        para(delayed(process_df)(df,i,count,batch_size,tfrecords_path) for count,i in enumerate(range(0, len(df), batch_size)))
    



# In[ ]:


generate_tf_records(df,model,filename=args.embeddings)


# In[ ]:





# In[ ]:




