#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import tensorflow as tf
import multiprocessing



def parse_tfr_element(element):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'emb' : tf.io.FixedLenFeature([], tf.string),
        'label' : tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    
    content = tf.io.parse_single_example(element, data)

    label = content['label']
    emb = content['emb']
    

    #get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(emb, out_type=tf.float32)
    return (feature, label)


def get_dataset(filename,batch_size):
    #create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    #pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset.batch(batch_size).prefetch(1)



# In[6]:


args = pd.Series({
    "root_dir":"/mnt/disks/data/",
    "embeddings":"music_style",
    "train_id": "global_sample_test"
})


# In[7]:


job_path = "/mnt/disks/data/fma/trains"


# In[8]:


batch_size = 32
num_iterations = 100

# In[15]:


train_path = os.path.join(job_path,args.train_id)


# In[16]:


base_path = os.path.join(args.root_dir,"fma")


# In[17]:


models_path = os.path.join(args.root_dir,"models")


# In[18]:


metadata_path = os.path.join(base_path,"fma_metadata")


# In[9]:


import numpy as np

def load_dataset(path,dataset='train',batch_size=32):
    tfrecords_path = os.path.join(path,'tfrecords',dataset)
    tfrecords_path = [os.path.join(tfrecords_path,path) for path in os.listdir(tfrecords_path)]
    # print(tfrecords_path)
    # tfrecords_path = tf.data.Dataset.list_files(tfrecords_path)
    dataset = get_dataset(tfrecords_path,batch_size)
    
    return dataset.batch(batch_size).prefetch(1)



def load_dataset_df(path,dataset='train'):
    
    dataset = load_dataset(path,dataset='train')
    
    df = pd.DataFrame(
        dataset.as_numpy_iterator(),
        columns=['Feature', 'Label']
    )
        
    df.Label = df.Label.apply(lambda x: x[0])
    df.dropna(inplace=True)
    
    
    try:
        df.Feature = df.Feature.apply(lambda x: x[0] if x.shape[0] != 0 else None)
    except:
        print(x)
    
    return df
    
    

# In[10]:


train_dataset = load_dataset(train_path,dataset='train',batch_size=batch_size)


# In[11]:


test_dataset = load_dataset(train_path,dataset='test',batch_size=batch_size)


# In[12]:


val_dataset = load_dataset(train_path,dataset='val',batch_size=batch_size)


# In[22]:




import xgboost as xgb
# Treina o modelo em batches
num_batches = 100
for i, (X_batch, y_batch) in enumerate(train_dataset.take(num_batches), 1):
    print(X_batch)
    # dtrain = xgb.DMatrix(X_batch.numpy(), label=y_batch.numpy())
    # booster = xgb.train(params, dtrain, evals=[(dtrain, 'train')], verbose_eval=False)

    if i % 10 == 0:
        print(f'Batch {i} - AUC: {booster.eval(dtrain)["auc"]:.4f}')

# Salva o modelo
booster.save_model('model.xgb')