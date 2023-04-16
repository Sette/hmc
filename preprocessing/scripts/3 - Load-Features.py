
import tensorflow as tf
import json
import os
import pandas as pd
args = pd.Series({
    "root_dir":"/mnt/disks/data/",
    "embeddings":"music_style",
    "train_id": "global_test_batch",
    'sample_size':1.0
})


# In[3]:




job_path = "/mnt/disks/data/fma/trains"


# In[15]:


train_path = os.path.join(job_path,args.train_id)


# In[16]:


base_path = os.path.join(args.root_dir,"fma")


# In[17]:


models_path = os.path.join(args.root_dir,"models")


# In[18]:


metadata_file = os.path.join(train_path,"metadata.json")


labels_path = os.path.join(train_path,"labels.json")


def __load_json__(path):
    with open(path, 'r') as f:
        tmp = json.loads(f.read())

    return tmp


labels = __load_json__(labels_path)

metadata = __load_json__(metadata_file)




def parse_tfr_element(element):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'emb' : tf.io.FixedLenFeature([], tf.string),
        'label' : tf.io.FixedLenFeature([136], tf.int64)
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


dataset = get_dataset("/mnt/disks/data/fma/trains/global_sample_test/tfrecords/test/0000000000.tfrecord")

for sample in dataset.take(1):
    print(sample[0].shape)
    print(sample[1].shape)
    print(sample)