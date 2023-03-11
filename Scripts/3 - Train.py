#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import pandas as pd


# In[4]:


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
        
        '''''
            Shuffle and reapeat
        '''''
        
        
        ds = ds.shuffle(buffer_size=1024 * 1 * 10)
        ds = ds.repeat(count=self.epochs)
        
        
        
        '''''
            Map and batch
        '''''
        
                      
        ds = ds.map(self.__parse__, num_parallel_calls=None)
        ds = ds.batch(self.batch_size,drop_remainder=False)
        
        
                      
        ds = ds.prefetch(buffer_size=5)
                      

        return ds
    
   
    @staticmethod
    def __parse__(example):
        parsed = tf.parse_single_example(example, features={
            'emb' : tf.io.FixedLenFeature([], tf.string),
            'label' : tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        })
        
        content = tf.io.parse_single_example(element, data)

        label = tf.cast(content['label'], tf.int32)
        label_hot = tf.one_hot(label1[0], label1[1])
        
        emb = content['emb']
        #get our 'feature'
        feature = tf.io.parse_tensor(emb, out_type=tf.float32)

        inp = {'emb': feature}
        out = {'global_output': label_hot}

        return inp, out


# In[5]:



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
        'label' : tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    
    content = tf.io.parse_single_example(element, data)

    label = content['label']
    emb = content['emb']
    

    #get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(emb, out_type=tf.float32)
    return (feature, label)


def get_dataset(filename):
    #create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    #pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset



# In[6]:


args = pd.Series({
    "root_dir":"/mnt/disks/data/",
    "embeddings":"music_style",
    "train_id": "global_sample_test"
})


# In[7]:


job_path = "/mnt/disks/data/fma/trains"


# In[8]:




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


def load_dataset(train_path,dataset='train'):
    tfrecords_path = os.path.join(train_path,'tfrecords',dataset)
    
    
    tfrecords_path = [os.path.join(tfrecords_path,path) for path in os.listdir(tfrecords_path)]
    dataset = get_dataset(tfrecords_path)
    
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


df_train = load_dataset(train_path,dataset='train')


# In[11]:


df_test = load_dataset(train_path,dataset='test')


# In[12]:


df_val = load_dataset(train_path,dataset='val')


# In[22]:


df_train.shape


# In[17]:


df_test.dropna(inplace=True)


# In[18]:


df_test.shape


# In[19]:


df_train.dropna(inplace=True)


# In[20]:


df_val.dropna(inplace=True)


# In[21]:


df_val.shape


# In[23]:


import xgboost as xgb


# In[24]:


xgb_model = xgb.XGBClassifier(random_state=42,eval_metric="auc",n_jobs=20)


# In[26]:


# Show all messages, including ones pertaining to debugging
xgb.set_config(verbosity=2)

# Get current value of global configuration
# This is a dict containing all parameters in the global configuration,
# including 'verbosity'
config = xgb.get_config()
assert config['verbosity'] == 2

# Example of using the context manager xgb.config_context().
# The context manager will restore the previous value of the global
# configuration upon exiting.
assert xgb.get_config()['verbosity'] == 2  # old value restored


# In[ ]:


xgb_model.fit(df_train.Feature.values.tolist(), df_train.Label.values.tolist(), 
        eval_set=[(df_val.Feature.values.tolist(), df_val.Label.values.tolist())])


# In[ ]:


from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report


# In[ ]:


y_pred = xgb_model.predict(df_test.Feature.values.tolist())

mean_squared_error(df_test.Label.values.tolist(), y_pred)


# In[41]:


pd.DataFrame(classification_report(df_test.Label.values.tolist(), y_pred,output_dict=True)).transpose()


# In[ ]:




