import tensorflow as tf

import pandas as pd
import os

def load_dataset(dataset):
        
    df = pd.DataFrame(
        dataset.as_numpy_iterator(),
        columns=['feature','genres']
    )

    df.dropna(inplace=True)

    return df

    


class Dataset:
    def __init__(self,files,epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        self.files = files

    def build(self,df=False):
        
        files = [os.path.join(self.files,file) for file in os.listdir(self.files)]
        
        dataset = tf.data.TFRecordDataset(files)
        
        # dataset = dataset.shuffle(buffer_size=1024 * 50 * 10)

        #pass every single feature through our mapping function
        dataset = dataset.map(
            self.__parse__
        )
        
        if df==True:
            return load_dataset(dataset)
        
        return dataset
    
    
    

    @staticmethod
    def __parse__(element):
        
        data = {
            'label1': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            'label2': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            'label3': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            'label4': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            'label5': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            'emb': tf.io.FixedLenFeature([], tf.string),
            'track_id' : tf.io.FixedLenFeature([], tf.int64),
        }
        
        content = tf.io.parse_single_example(element, data)

        track_id = content['track_id']
        emb = tf.io.parse_tensor(content['emb'], out_type=tf.float32)
        

        label1 = content['label1']
        label2 = content['label2']
        label3 = content['label3']
        label4 = content['label4']
        label5 = content['label5']
        
        inp = {"emb":emb }

        labels = {'first_level_output': label1,
               'second_level_output': label2,
               'third_level_output': label3,
               'fourth_level_output': label4,
               'fifth_level_output':label5}

        return inp, labels
