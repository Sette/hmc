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
