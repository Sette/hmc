import os
import pandas as pd
from datetime import datetime as dt

from sabotage.arguments import  build
from sabotage.train import run

import tensorflow as tf


# Set python level verbosity
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

# Set C++ Graph Execution level verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.compat.v1.logging.DEBUG)

base_path = "/mnt/disks/data/fma/trains"
id = "hierarchical_all"


train_path = os.path.join(base_path,id)
tfrecords_path =os.path.join(train_path,'tfrecords')
metadata_path = os.path.join(train_path,"metadata.json")
labels_path = os.path.join(train_path,"labels.json")


args = pd.Series({
    "batch_size":32,
    "epochs":10,
    "dropout":0.1,
    'patience':1,
    'max_queue_size':64,
    "labels_path": labels_path,
    "metadata_path": metadata_path,
    "trainset_pattern": os.path.join(tfrecords_path,'train'),
    "testset_pattern": os.path.join(tfrecords_path,'test'),
    "valset_pattern": os.path.join(tfrecords_path,'val')
})


if __name__ == '__main__':
    time_start = dt.utcnow()
    print("[{}] Experiment started at {}".format(id, time_start.strftime("%H:%M:%S")))
    print(".......................................")
    print(args)
    run(args)
    time_end = dt.utcnow()
    time_elapsed = time_end - time_start
    print(".......................................")
    print("[{}] Experiment finished at {} / elapsed time {}s".format(id, time_end.strftime("%H:%M:%S"), time_elapsed.total_seconds()))