import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from essentia.standard import MonoLoader, TensorflowPredictMusiCNN
from sklearn.model_selection import train_test_split

base_path = "/mnt/disks/data/fma"

model_name = "msd-musicnn-1.pb"

model_path = os.path.join("/home/jupyter/models",model_name)

metadata_path = os.path.join(base_path,"fma_metadata")

df = pd.read_csv(os.path.join(metadata_path,"tracks_genres_id_full.csv"))

### Exemplo de extração de features
audio = MonoLoader(filename=df.iloc[1].file_path, sampleRate=16000)()
model = TensorflowPredictMusiCNN(graphFilename=model_path, output="model/dense/BiasAdd")
activations = model(audio)

def extract_feature(file_path):
    audio = MonoLoader(filename=df.iloc[1].file_path, sampleRate=16000)()
    model = TensorflowPredictMusiCNN(graphFilename=model_path, output="model/dense/BiasAdd")
    activations = model(audio)
    return activations


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

def parse_single_music(music, label):

    #define the dictionary -- the structure -- of our single example
    data = {
        'emb' : _bytes_feature(serialize_array(music)),
        'label' : _int64_feature(label)
    }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out








def write_musics_to_tfr_short(musics, labels, filename:str="musics"):
    filename= filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0

    for index in tqdm(range(len(musics))):

        #get the data we want to write
        current_music = musics[index] 
        current_label = labels[index]

        current_music = extract_feature(current_music)
        
        out = parse_single_music(music=current_music, label=current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count


X_train, X_test, y_train, y_test = train_test_split(df.file_path.tolist(), df.first_genre_id_label.tolist(), test_size=0.30, random_state=42)

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

count = write_musics_to_tfr_short(X_validation, y_validation, filename="fma_validation")

print(count)


count = write_musics_to_tfr_short(X_test, y_test, filename="fma_test")

print(count)

count = write_musics_to_tfr_short(X_train, y_train, filename="fma_train")

print(count)