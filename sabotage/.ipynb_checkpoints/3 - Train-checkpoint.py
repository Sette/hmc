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


genre_dict

 Parameters
    ----------
    base_estimator : classifier object, function, dict, or None
        A scikit-learn compatible classifier object implementing "fit" and "predict_proba" to be used as the
        base classifier.
        If a callable function is given, it will be called to evaluate which classifier to instantiate for
        current node. The function will be called with the current node and the graph instance.
        Alternatively, a dictionary mapping classes to classifier objects can be given. In this case,
        when building the classifier tree, the dictionary will be consulted and if a key is found matching
        a particular node, the base classifier pointed to in the dict will be used. Since this is most often
        useful for specifying classifiers on only a handlful of objects, a special "DEFAULT" key can be used to
        set the base classifier to use as a "catch all".
        If not provided, a base estimator will be chosen by the framework using various meta-learning
        heuristics (WIP).
    class_hierarchy : networkx.DiGraph object, or dict-of-dicts adjacency representation (see examples)
        A directed graph which represents the target classes and their relations. Must be a tree/DAG (no cycles).
        If not provided, this will be initialized during the `fit` operation into a trivial graph structure linking
        all classes given in `y` to an artificial "ROOT" node.
    prediction_depth : "mlnp", "nmlnp"
        Prediction depth requirements. This corresponds to whether we wish the classifier to always terminate at
        a leaf node (mandatory leaf-node prediction, "mlnp"), or wish to support early termination via some
        stopping criteria (non-mandatory leaf-node prediction, "nmlnp"). When "nmlnp" is specified, the
        stopping_criteria parameter is used to control the behaviour of the classifier.
    algorithm : "lcn", "lcpn"
        The algorithm type to use for building the hierarchical classification, according to the
        taxonomy defined in [1].
        "lcpn" (which is the default) stands for "local classifier per parent node". Under this model,
        a multi-class classifier is trained at each parent node, to distinguish between each child nodes.
        "lcn", which stands for "local classifier per node". Under this model, a binary classifier is trained
        at each node. Under this model, a further distinction is made based on how the training data set is constructed.
        This is controlled by the "training_strategy" parameter.
    training_strategy: "exclusive", "less_exclusive", "inclusive", "less_inclusive",
                       "siblings", "exclusive_siblings", or None.
        This parameter is used when the "algorithm" parameter is to set to "lcn", and dictates how training data
        is constructed for training the binary classifier at each node.
    stopping_criteria: function, float, or None.
        This parameter is used when the "prediction_depth" parameter is set to "nmlnp", and is used to evaluate
        at a given node whether classification should terminate or continue further down the hierarchy.
        When set to a float, the prediction will stop if the reported confidence at current classifier is below
        the provided value.
        When set to a function, the callback function will be called with the current node attributes,
        including its metafeatures, and the current classification results.
        This allows the user to define arbitrary logic that can decide whether classification should stop at
        the current node or continue. The function should return True if classification should continue,
        or False if classification should stop at current node.
    root : integer, string
        The unique identifier for the qualified root node in the class hierarchy. The hierarchical classifier
        assumes that the given class hierarchy graph is a rooted DAG, e.g has a single designated root node
        of in-degree 0. This node is associated with a special identifier which defaults to a framework provided one,
        but can be overridden by user in some cases, e.g if the original taxonomy is already rooted and there"s no need
        for injecting an artifical root node.
    progress_wrapper : progress generator or None
        If value is set, will attempt to use the given generator to display progress updates. This added functionality
        is especially useful within interactive environments (e.g in a testing harness or a Jupyter notebook). Setting
        this value will also enable verbose logging. Common values in tqdm are `tqdm_notebook` or `tqdm`
    feature_extraction : "preprocessed", "raw"
        Determines the feature extraction policy the classifier uses.
        When set to "raw", the classifier will expect the raw training examples are passed in to `.fit()` and `.train()`
        as X. This means that the base_estimator should point to a sklearn Pipeline that includes feature extraction.
        When set to "preprocessed", the classifier will expect X to be a pre-computed feature (sparse) matrix.
    mlb : MultiLabelBinarizer or None
        For multi-label classification, the MultiLabelBinarizer instance that was used for creating the y variable.
    mlb_prediction_threshold : float
        For multi-label prediction tasks (when `mlb` is set to a MultiLabelBinarizer instance), can define a prediction
        score threshold to use for considering a label to be a prediction. Defaults to zero.
    use_decision_function : bool
        Some classifiers (e.g. sklearn.svm.SVC) expose a `.decision_function()` method which would take in the
        feature matrix X and return a set of per-sample scores, corresponding to each label. Setting this to True
        would attempt to use this method when it is exposed by the base classifier.

# In[22]:


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
    base_estimator=tree_estimator,
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


# In[27]:


print('cabou')


# In[ ]:





# In[ ]:




