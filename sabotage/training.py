import json
import pickle

from multiprocessing import cpu_count

from sabotage.model.model import build_model
# from sabotage.model.callback import ValidateCallback, BackupAndRestoreCheckpoint
from sabotage.dataset.dataset import Dataset

import tensorflow as tf
from tensorflow.python.lib.io.file_io import FileIO
from keras.callbacks import EarlyStopping

print("========================= Tensorflow =========================")
print("GPU is available: {}".format(tf.test.is_gpu_available()))
print("==============================================================")


def run(args):
    print(args)

    if tf.test.is_gpu_available():

        with open(args.metadata_path, 'r') as f:
            metadata = json.loads(f.read())
            print(metadata)

        with open(args.labels_path, 'r') as f:
            labels = json.loads(f.read())

        with open(args.testset_path, 'rb') as f:
            x_test, y_test, texts_test = pickle.loads(f.read())

        params = {
            'vocab_size': metadata['vocab_size'],
            'first_level_output_size': labels['labels1_count'],
            'second_level_output_size': labels['labels2_count'],
            'third_level_output_size': labels['labels3_count'],
            'fourth_level_output_size': labels['labels4_count'],
            'lstm_size': args.lstm_size,
            'dropout': args.dropout,
            'embed_size': args.embed_size
        }

        print("======================== Model Params ========================")
        print(params)
        print("==============================================================")

        model = build_model(**params)

        ds_train = Dataset(args.bucket_name, args.trainset_pattern, args.epochs, args.batch_size).build()
        ds_validation = Dataset(args.bucket_name, args.validationset_pattern, args.epochs, args.batch_size).build()

        # backup_and_restore_callback = BackupAndRestoreCheckpoint(model, args.train_path)

        # callbacks = [backup_and_restore_callback,
        #              ValidateCallback(model, labels, x_test, y_test, args.train_path),
        #              EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=args.patience, verbose=1, mode='min')]

        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=args.patience, verbose=1, mode='min')]

        # next_epoch = backup_and_restore_callback.restore()

        model.fit(ds_train,
                  validation_data=ds_validation,
                  steps_per_epoch=metadata['trainset_count'] // args.batch_size,
                  validation_steps=metadata['validationset_count'] // args.batch_size,
                  epochs=args.epochs,
                  # initial_epoch=next_epoch,
                  max_queue_size=args.max_queue_size,
                  workers=cpu_count(),
                  use_multiprocessing=True,
                  callbacks=callbacks)

    else:

        with open(args.metadata_path, 'r') as f:
            metadata = json.loads(f.read())
            print(metadata)

        with open(args.labels_path, 'r') as f:
            labels = json.loads(f.read())

        with open(args.testset_path, 'rb') as f:
            x_test, y_test, texts_test = pickle.loads(f.read())

        params = {}

        model = build_model(**params)

        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=args.patience, verbose=1, mode='min')]

        ds_train = Dataset(args.bucket_name, args.trainset_pattern, args.epochs, args.batch_size).build()
        ds_validation = Dataset(args.bucket_name, args.validationset_pattern, args.epochs, args.batch_size).build()

        model.fit(ds_train,
                  validation_data=ds_validation,
                  steps_per_epoch=metadata['trainset_count'] // args.batch_size,
                  validation_steps=metadata['validationset_count'] // args.batch_size,
                  epochs=args.epochs,
                  max_queue_size=args.max_queue_size,
                  workers=cpu_count(),
                  use_multiprocessing=True,
                  callbacks=callbacks)
