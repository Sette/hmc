import json
import os


from keras import Model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from multiprocessing import cpu_count

from sabotage.model import build_model
# from sabotage.model.callback import ValidateCallback, BackupAndRestoreCheckpoint
from sabotage.dataset import Dataset

from keras.callbacks import EarlyStopping

print("========================= Tensorflow =========================")
print("GPU is available: {}".format(tf.test.is_gpu_available()))
print("==============================================================")


def run(args: object):
    print(args)

    with open(args.metadata_path, 'r') as f:
        metadata = json.loads(f.read())
        print(metadata)

    with open(args.labels_path, 'r') as f:
        labels = json.loads(f.read())

    levels_size = {'level1': labels['label1_count'] ,
                   'level2': labels['label2_count'] ,
                   'level3': labels['label3_count'] ,
                   'level4': labels['label4_count'] ,
                   'level5': labels['label5_count'] }

    params: dict = {
        'levels_size': levels_size,
        'sequence_size': metadata['sequence_size'],
        'dropout': args.dropout
    }

    print(params)
    model = build_model(**params)
    ds_train = Dataset(args.trainset_pattern, args.epochs, args.batch_size).build(df=False)
    ds_validation = Dataset(args.valset_pattern, args.epochs, args.batch_size).build(df=False)
    callbacks = [EarlyStopping(monitor='loss', patience=args.patience, verbose=1)]
    model.fit(ds_train,
              validation_data=ds_validation,
              steps_per_epoch=metadata['trainset_count'] // args.batch_size,
              validation_steps=metadata['validationset_count'] // args.batch_size,
              epochs=args.epochs,
              max_queue_size=args.max_queue_size,
              workers=cpu_count(),
              use_multiprocessing=True,
              callbacks=callbacks)
