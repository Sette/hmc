#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from math import ceil
from sklearn.utils import shuffle

from joblib import Parallel, delayed

import multiprocessing
from tqdm import tqdm

from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
from sklearn.model_selection import train_test_split


# In[2]:



args = pd.Series({
    "root_dir":"/mnt/disks/data/",
    "embeddings":"music_style",
    "train_id": "global_sample_test",
    'sample_size':1
})


# In[3]:





job_path = "/mnt/disks/data/fma/trains"


# In[15]:


train_path = os.path.join(job_path,args.train_id)


# In[16]:


base_path = os.path.join(args.root_dir,"fma")


# In[17]:


models_path = os.path.join(args.root_dir,"models")


metadata_path_fma = os.path.join(base_path,"fma_metadata")


# In[18]:


metadata_file = os.path.join(train_path,"metadata.json")


labels_file = os.path.join(train_path,"labels.json")


# In[4]:



def create_dir(path):
    # checking if the directory demo_folder2 
    # exist or not.
    if not os.path.isdir(path):

        # if the demo_folder2 directory is 
        # not present then create it.
        os.makedirs(path)
    return True


# In[5]:


create_dir(train_path)


# In[6]:



if args.embeddings == "music_style":
    model_path = os.path.join(models_path,args.embeddings,"discogs-effnet-bs64-1.pb")


# In[7]:




def __load_json__(path):
    with open(path, 'r') as f:
        tmp = json.loads(f.read())

    return tmp



df = pd.read_csv(os.path.join(metadata_path_fma,"tracks_genres_id_full.csv"))


# In[8]:


df= df[df.valid == True]


# In[9]:



df = df.sample(frac=args.sample_size)


# ## Gerando labels

# In[ ]:





# In[10]:


def create_labels(arr):
    labels = sorted(list(set(arr)))
    return dict([(label, i) for i, label in enumerate(labels)])   


# In[11]:


df


# In[12]:


labels = {
        'global': create_labels(df.first_genre_id)
}


# In[13]:


labels


# In[14]:


df['first_genre_id_label'] = df.first_genre_id.apply(lambda x:  labels['global'][x])


# In[15]:


df


# In[16]:


with open(labels_file, "w+") as f:
    f.write(json.dumps(labels))


# In[17]:


### Exemplo de extração de features
audio = MonoLoader(filename=df.iloc[1].file_path, sampleRate=16000)()
model = TensorflowPredictEffnetDiscogs(graphFilename=model_path,output="PartitionedCall:1")
activations = model(audio)


# In[18]:


activations = model(audio)


# In[19]:


activations.shape


# In[20]:


groups = df.groupby("first_genre_id_label")


# In[21]:



def __split_data__(group, percentage=0.1):
    if len(group) == 1:
        return group, group

    shuffled = shuffle(group.values)
    finish_test = int(ceil(len(group) * percentage))

    first = pd.DataFrame(shuffled[:finish_test], columns=group.columns)
    second = pd.DataFrame(shuffled[finish_test:], columns=group.columns)

    return first, second


# In[32]:


def __split_data_sample(groups):
    dataset_trainset_path = os.path.join(train_path,"trainset.csv")
    dataset_testset_path = os.path.join(train_path,"testset.csv")
    dataset_validationset_path = os.path.join(train_path,"validationset.csv")
    
    
    X_train,y_train,X_test,y_test,X_val,y_val = (list(),list(),list(),list(),list(),list())
    for code, group in groups:
        
        test, train_to_split  = __split_data__(group, 0.05) # 10%
        validation, train = __split_data__(train_to_split, 0.05) # %10
        #rint(test)
        
        X_train.append(train)
        X_test.append(test)
        X_val.append(validation)
        
    X_train = pd.concat(X_train, sort=False).sample(frac=1).reset_index(drop=True)
    X_train.to_csv(dataset_trainset_path, index=False, quoting=csv.QUOTE_ALL)
    print(dataset_trainset_path)
    
    X_test = pd.concat(X_test, sort=False).sample(frac=1).reset_index(drop=True)
    X_test.to_csv(dataset_testset_path, index=False, quoting=csv.QUOTE_ALL)
    print(dataset_testset_path)

    X_val = pd.concat(X_val, sort=False).sample(frac=1).reset_index(drop=True)
    X_val.to_csv(dataset_validationset_path, index=False, quoting=csv.QUOTE_ALL)
    print(dataset_validationset_path)
    
    return X_train,X_test,X_val


# In[22]:




X_train,X_test,X_validation = __split_data_sample(groups)


# In[23]:



def extract_feature(file_path,model):
    ### Configuração do model para extrair a representação do aúdio
    # model = TensorflowPredictEffnetDiscogs(graphFilename=model_path)
    audio = MonoLoader(filename=file_path, sampleRate=16000)()
    activations = model(audio)
    return activations


# In[44]:


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
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array



def parse_single_music(data,music,labels):
    cat1 = data
    
    label1 = np.array(cat1, np.int64)
    
    
    
    #define the dictionary -- the structure -- of our single example
    data = {
        'emb' : _bytes_feature(serialize_array(music)),
        'label' : _int64_feature(label1)
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

    batch_df = batch_df[['first_genre_id_label']]


    tfrecords = [parse_single_music(data, x,labels) for data, x in zip(batch_df.values, X)]

    path = os.path.join(tfrecords_path,f"{str(count).zfill(10)}.tfrecord")

    #with tf.python_io.TFRecordWriter(path) as writer:
    with tf.io.TFRecordWriter(path) as writer:
        for tfrecord in tfrecords:
            writer.write(tfrecord.SerializeToString())

    print(f"{count} {len(tfrecords)} {path}")


def generate_tf_records(df,labels,model,filename="train"):
    
    tfrecords_path = os.path.join(train_path,"tfrecords",filename)
    
    create_dir(tfrecords_path)
    
    
    batch_size = 1024 * 10  # 10k records from each file batch
    count = 0
    total = ceil(len(df) / batch_size)
    
    with Parallel(n_jobs=6, require='sharedmem') as para:
        print("Estamos usando paralelismo!!!")
        para(delayed(process_df)(df,i,count,batch_size,tfrecords_path) for count,i in enumerate(range(0, len(df), batch_size)))
    
# In[25]:


df.file_path.iloc[2]


# In[26]:


model_path


# In[27]:


model


# In[ ]:



extract_feature(df.file_path.iloc[2],model)


# In[ ]:



dataset_names = ["train","test","val"]

datasets = [X_train,X_test,X_validation]



# In[ ]:


with Parallel(n_jobs=3, require='sharedmem') as para:
    print("Estamos usando paralelismo!!!")
    para(delayed(generate_tf_records)(dataset,labels,model,dataset_name) for (dataset_name,dataset) in zip(dataset_names,datasets))


# In[ ]:



metadata = {
    "train_count":len(X_train),
    "test_count":len(X_test),
    "val_count":len(X_validation),
    "global_size": len(labels['global']),
    "root_dir":args.root_dir,
    "embeddings":args.embeddings,
    "train_id": args.train_id
}


# In[ ]:




# In[ ]:


with open(metadata_file, 'w+') as f:
    f.write(json.dumps(metadata))


# In[ ]:


# In[ ]:





# In[ ]:



