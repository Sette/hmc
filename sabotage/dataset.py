import tensorflow as tf
import multiprocessing
import pandas as pd
import os


BUFFER_SIZE_3 = 10

def load_dataset(dataset):
    df = pd.DataFrame(
        dataset.as_numpy_iterator(),
        columns=['feature','genres_globais',]
    )

    df.dropna(inplace=True)

    return df



class Dataset:
    def __init__(self,files,epochs, batch_size,num_nodes_per_level,num_classes_per_node):
        self.epochs = epochs
        self.batch_size = batch_size
        self.files = files
        self.num_classes_per_node = num_classes_per_node
        self.num_nodes_per_level = num_nodes_per_level

    def build(self,df=False):
        
        files = [os.path.join(self.files,file) for file in os.listdir(self.files)]

        
        ds = tf.data.TFRecordDataset(files,num_parallel_reads=multiprocessing.cpu_count())
        
        '''''
            Shuffle and reapeat
        '''''
        
        ds = ds.shuffle(buffer_size=1024 * 50 * BUFFER_SIZE_3)
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
    

    def __parse__(self,element):
        ### Estrutura dos tfrecords
        data = {
            'label1': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'label2': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'label3': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'label4': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'label5': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'features': tf.io.FixedLenFeature([1280], tf.float32),
            'track_id' : tf.io.FixedLenFeature([], tf.int64),
        }
        
        
        content = tf.io.parse_single_example(element, data)

        track_id = content['track_id']
        
        
        label1 = tf.cast(content['label1'], tf.int32)
        label1_hot = tf.one_hot(label1[0], label1[1])

        label2 = tf.cast(content['label2'], tf.int32)
        label2_hot = tf.one_hot(label2[0], label2[1])

        label3 = tf.cast(content['label3'], tf.int32)
        label3_hot = tf.one_hot(label3[0], label3[1])

        label4 = tf.cast(content['label4'], tf.int32)
        label4_hot = tf.one_hot(label4[0], label4[1])

        label5 = tf.cast(content['label5'], tf.int32)
        label5_hot = tf.one_hot(label5[0], label5[1])

        
        features = content['features']


        global_labels_list = {'level1': label1,
               'level2': label2,
               'level3': label3,
               'level4': label4,
               'level5': label5
        }
        

        # Labels locais
        local_labels_list = [
            tf.sparse.to_dense(label1_hot),
            tf.sparse.to_dense(label2_hot),
            tf.sparse.to_dense(label3_hot),
            tf.sparse.to_dense(label4_hot),
            tf.sparse.to_dense(label5_hot)
        ]

        return features,global_labels_list,local_labels_list
