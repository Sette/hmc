#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import tensorflow as tf
import pandas as pd
import pickle
from tqdm.notebook import tqdm
import logging


import sys
sys.setrecursionlimit(1000000)


# In[2]:


args = pd.Series({
    "root_dir":"/mnt/disks/data/",
    "dataset_path":"/mnt/disks/data/fma/fma_large",
    "embeddings":"music_style",
    "train_id": "hierarchical_partition"
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


def __load_json__(path):
    with open(path, 'r') as f:
        tmp = json.loads(f.read())

    return tmp




# In[5]:


import tensorflow as tf
import multiprocessing



class Dataset:
    def __init__(self, tfrecords_path, epochs, batch_size):
        self.tfrecords_path = tfrecords_path
        self.epochs = epochs
        self.batch_size = batch_size

    def list_files(self):
        return [os.path.join(tfrecords_path,file_path) for file_path in os.listdir(tfrecords_path)]

    def build(self):
        files = self.list_files()

        print("build_tf record: files_count: {} / batch_size: {} / epochs: {}".format(len(files), self.batch_size, self.epochs))

        ds = tf.data.TFRecordDataset(files, num_parallel_reads=multiprocessing.cpu_count())
                      

        return ds
    
   
    @staticmethod
    def __parse__(example):
        parsed = tf.parse_single_example(example, features={
            'emb' : tf.io.FixedLenFeature([], tf.string),
            'track_id' : tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        })
        
        content = tf.io.parse_single_example(element, data)

        label = tf.cast(content['track_id'], tf.int32)
        label_hot = tf.one_hot(label1[0], label1[1])
        
        emb = content['emb']
        #get our 'feature'
        feature = tf.io.parse_tensor(emb, out_type=tf.float32)

        inp = {'emb': feature}
        out = {'global_output': label_hot}

        return inp, out


# In[6]:


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



def parser(serialized_example):
    features_description = {'emb': tf.io.FixedLenFeature([], tf.string)}
    features = tf.io.parse_single_example(serialized_example, features_description)
    features = tf.io.decode_raw(features['emb'], tf.float32)
    return features


def parse_tfr_element(element):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'emb' : tf.io.FixedLenFeature([], tf.string),
        'track_id' : tf.io.FixedLenFeature([], tf.int64),
    }
    
    content = tf.io.parse_single_example(element, data)

    track_id = content['track_id']
    emb = content['emb']
    

    #get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(emb, out_type=tf.float32)
    return (feature, track_id)


def get_dataset(filename):
    #create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    #pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset




# In[7]:


import numpy as np


def load_dataset(path,dataset=args.embeddings):
    tfrecords_path = os.path.join(path,'tfrecords',dataset)
    
    
    tfrecords_path = [os.path.join(tfrecords_path,path) for path in os.listdir(tfrecords_path)]
    dataset = get_dataset(tfrecords_path)
    
    df = pd.DataFrame(
        dataset.as_numpy_iterator(),
        columns=['feature', 'track_id']
    )
        
    df.dropna(inplace=True)
    
    
    try:
        df.feature = df.feature.apply(lambda x: x[0] if x.shape[0] != 0 else None)
    except:
        print(x)
    
    return df
    


# In[8]:


df = load_dataset(args.dataset_path,dataset=args.embeddings)


# In[9]:


df


# In[10]:


df.dropna(inplace=True)


# In[11]:


df


# In[12]:


tracks_df = pd.read_csv(os.path.join(train_path,"tracks.csv"))


# In[13]:


tracks_df


# In[14]:


labels = __load_json__(labels_file)


# In[15]:


tqdm.pandas()


# In[16]:


# tracks_df.loc[:,'labels_1'] = tracks_df.labels_1.astype(str).progress_apply(lambda x: labels['label1'][x])

# tracks_df.loc[:,'labels_2'] = tracks_df.labels_2.astype(str).progress_apply(lambda x: labels['label2'][x])

# tracks_df.loc[:,'labels_3'] = tracks_df.labels_3.astype(str).progress_apply(lambda x: labels['label3'][x])

# tracks_df.loc[:,'labels_4'] = tracks_df.labels_4.astype(str).progress_apply(lambda x: labels['label4'][x])

# tracks_df.loc[:,'labels_5'] = tracks_df.labels_5.astype(str).progress_apply(lambda x: labels['label5'][x])


# In[17]:


tracks_df = tracks_df.merge(df, on='track_id')


# In[18]:


# genres_df = tracks_df.drop_duplicates(subset=['labels_5'])[['labels_1','labels_2','labels_3','labels_4','labels_5']]
genres_df = tracks_df.drop_duplicates(subset=['labels_2'])[['labels_1','labels_2']]


# In[19]:


from sklearn import svm
from sklearn import tree
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled


# In[20]:


# Cria um dicionário que mapeia o ID de cada gênero musical aos IDs de seus subgêneros
genre_dict = {
    ROOT:genres_df.labels_1.unique().tolist()}

def add_node(genre_id,parent_id):
    if pd.notna(parent_id):
        if parent_id not in genre_dict:
            genre_dict[parent_id] = []
        genre_dict[parent_id].append(genre_id)



for i, row in genres_df.iterrows():
    genre_id = row['labels_2']
    parent_id = row['labels_1']
    add_node(genre_id,parent_id)

#     genre_id = row['labels_3']
#     parent_id = row['labels_2']
#     add_node(genre_id,parent_id)
    
#     genre_id = row['labels_4']
#     parent_id = row['labels_3']
#     add_node(genre_id,parent_id)
    
#     genre_id = row['labels_5']
#     parent_id = row['labels_4']
    # add_node(genre_id,parent_id)
    
    
    

# In[13]:


# genre_dict



# In[21]:


base_estimator = make_pipeline(
    TruncatedSVD(n_components=24),
    svm.SVC(
        gamma=0.001,
        kernel="rbf",
        probability=True
    ),
)

tree_estimator = tree.DecisionTreeClassifier()

clf = HierarchicalClassifier(
    base_estimator=base_estimator,
    class_hierarchy=genre_dict,
    progress_wrapper=tqdm,
    feature_extraction="preprocessed"
)


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(
    tracks_df.feature.values.tolist(),
    tracks_df.labels_2.astype(str).values.tolist(),
    test_size=0.2,
    random_state=42,
)


# In[24]:


logging.disable(logging.CRITICAL)


# In[ ]:


model = clf.fit(X_train, y_train)


# In[26]:


filename = os.path.join(train_path,'hsvm.model')
pickle.dump(model, open(filename, 'wb'))

y_pred = clf.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))


print('cabou')



