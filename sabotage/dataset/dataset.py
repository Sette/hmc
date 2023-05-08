import tensorflow as tf
import multiprocessing


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
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



def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'emb': tf.io.FixedLenFeature([], tf.string),
        'track_id': tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data)

    track_id = content['track_id']
    emb = content['emb']

    # get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(emb, out_type=tf.float32)
    return (feature, track_id)


def get_dataset(filename):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset


def load_dataset(path, dataset=args.embeddings):
    tfrecords_path = os.path.join(path, 'tfrecords', dataset)

    tfrecords_path = [os.path.join(tfrecords_path, path) for path in os.listdir(tfrecords_path)]
    dataset = get_dataset(tfrecords_path)

    df = pd.DataFrame(
        dataset.as_numpy_iterator(),
        columns=['feature', 'track_id']
    )

    df.dropna(inplace=True)

    try:
        df.feature = df.feature.apply(lambda x: x[0] if x.shape[0] != 0 else None)
    except:
        pass

    return df


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
