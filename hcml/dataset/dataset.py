import tensorflow as tf
import multiprocessing
import pandas as pd
import os


BUFFER_SIZE = 10

def load_dataset(dataset):
    df = pd.DataFrame(
        dataset.as_numpy_iterator(),
        columns=['level1_output','level2_output',\
                 'level3_output','level4_output','features']
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

        ds = tf.data.TFRecordDataset(files)
        
        '''''
            Shuffle and reapeat
        '''''
        
        ds = ds.shuffle(buffer_size=1024 * 50 * 10)
        ds = ds.repeat(count=self.epochs)
        
        
        '''''
            Map and batch
        '''''
        
                      
        ds = ds.map(self.__parse__, num_parallel_calls=None)

        if df==True:
            return load_dataset(ds)
        
        ds = ds.batch(self.batch_size,drop_remainder=False)
        
        
                      
        ds = ds.prefetch(buffer_size=5)
        
       
        
        return ds
    

    @staticmethod
    def __parse__(element):
        
        data = {
            'label1': tf.io.FixedLenFeature([], tf.int64),
            'label2': tf.io.FixedLenFeature([], tf.int64),
            'label3': tf.io.FixedLenFeature([], tf.int64),
            'label4': tf.io.FixedLenFeature([], tf.int64),
            'features': tf.io.FixedLenFeature([1280], tf.float32),
            'track_id' : tf.io.FixedLenFeature([], tf.int64),
        }
        
        content = tf.io.parse_single_example(element, data)

        track_id = content['track_id']
        label1 = tf.cast(content['label1'], tf.int32)

        label2 = tf.cast(content['label2'], tf.int32)

        label3 = tf.cast(content['label3'], tf.int32)

        label4 = tf.cast(content['label4'], tf.int32)

        
        inp = {"features":content['features'] }


        labels = {'level1_output': label1,
               'level2_output': label2,
               'level3_output': label3,
               'level4_output': label4,
        }

        return inp, labels
