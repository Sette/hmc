import tensorflow as tf

class Dataset:
    def __init__(self,files,epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        self.files = files

    def build(self):
        ds = tf.data.Dataset.list_files(self.files)
        ds = tf.data.TFRecordDataset(ds, num_parallel_reads=8, buffer_size=1024 * 1024 * 256)

        '''''
            Shuffle and reapeat
        '''''

        ds = ds.shuffle(buffer_size=1024 * 50 * 10)
        ds = ds.repeat(count=self.epochs)

        '''''
            Map and batch
        '''''

        ds = ds.map(self.__parse__, num_parallel_calls=None)
        ds = ds.batch(self.batch_size, drop_remainder=False)

        ds = ds.prefetch(buffer_size=5)

        return ds

    @staticmethod
    def __parse__(element):
        parsed = tf.io.parse_single_example(element, features={
            'label1': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'label2': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'label3': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'label4': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'label5': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'emb' : tf.io.FixedLenFeature([], tf.string),
            'track_id' : tf.io.FixedLenFeature([], tf.int64),
        })

        track_id = parsed['track_id']

        emb = parsed['emb']
        inp = tf.io.parse_tensor(emb, out_type=tf.float32)

        label1 = tf.cast(parsed['label1'], tf.int32)
        label1_hot = tf.one_hot(label1[0], label1[1])

        label2 = tf.cast(parsed['label2'], tf.int32)
        label2_hot = tf.one_hot(label2[0], label2[1])

        label3 = tf.cast(parsed['label3'], tf.int32)
        label3_hot = tf.one_hot(label3[0], label3[1])

        label4 = tf.cast(parsed['label4'], tf.int32)
        label4_hot = tf.one_hot(label4[0], label4[1])

        label5 = tf.cast(parsed['label5'], tf.int32)
        label5_hot = tf.one_hot(label5[0], label5[1])

        out = {'first_level_output': label1_hot,
               'second_level_output': label2_hot,
               'third_level_output': label3_hot,
               'fourth_level_output': label4_hot,
               'fifth_level_output':label5_hot}

        return inp, out
