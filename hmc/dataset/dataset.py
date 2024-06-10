

BUFFER_SIZE = 10


class Dataset:
    def __init__(self,files,epochs, batch_size, levels_size):
        self.epochs = epochs
        self.batch_size = batch_size
        self.files = files
        self.depth = len(levels_size)
        self.levels_size = levels_size

    def load_dataframe(self, dataset):
        columns = ['features','labels']
        
        
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
        
        #ds = ds.shuffle(buffer_size=1024 * 50 * 10)
        #ds = ds.repeat(count=self.epochs)
        
        
        '''''
            Map and batch
        '''''
        
                      
        ds = ds.map(self.__parse__, num_parallel_calls=None)

        if df==True:
            return self.load_dataframe(ds)
        
        ds = ds.batch(self.batch_size,drop_remainder=False)
        
        
                      
        ds = ds.prefetch(buffer_size=5)
        
        
        return ds

    def convert_to_binary(self, label, num_classes):
        # Filtrar índices negativos
        valid_indices = tf.boolean_mask(label, label >= 0)
        
        # Inicializar vetor binário
        binary_label = tf.zeros(num_classes, dtype=tf.float32)
        
        # Atualizar vetor binário com base nos rótulos válidos
        indices = tf.expand_dims(valid_indices, 1)
        updates = tf.ones_like(valid_indices, dtype=tf.float32)
        binary_label = tf.tensor_scatter_nd_update(binary_label, indices, updates)

        return binary_label

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
            binary_label = self.convert_to_binary(local_label, self.levels_size[f'level{level}'])
            labels.update({f'level{level}': binary_label})
        

        inp = {"features":content['features'] }

        return inp, labels
