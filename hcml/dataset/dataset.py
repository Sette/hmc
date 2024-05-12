import tensorflow as tf
import multiprocessing
import pandas as pd
import os


BUFFER_SIZE = 10



class Dataset:
    def __init__(self,files,epochs, batch_size,depth):
        self.epochs = epochs
        self.batch_size = batch_size
        self.files = files
        self.depth = depth

    def load_dataframe(self, dataset):
        columns = ['features','track_id']
        for level in range(1, self.depth+1):
            columns.append(f'level{level}_output')
        
        df = pd.DataFrame(
            dataset.as_numpy_iterator(),
            columns=columns
        )
    
        df.dropna(inplace=True)
    
        return df

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
            return self.load_dataframe(ds)
        
        ds = ds.batch(self.batch_size,drop_remainder=False)
        
        
                      
        ds = ds.prefetch(buffer_size=5)
        
        
        return ds

    def __parse__(self, element):
        data = {}
        for level in range(1, self.depth+1):
            data[f'label{level}'] = tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        
        data.update({
            'features': tf.io.FixedLenFeature([1280], tf.float32),
            'track_id' : tf.io.FixedLenFeature([], tf.int64),
        })
        
        content = tf.io.parse_single_example(element, data)

        labels = {}
        for level in range(1, self.depth+1):
            local_label = content[f'label{level}']
            labels.update({f'level{level}_output':local_label})

        inp = {"features":content['features'] }

        return inp, labels
