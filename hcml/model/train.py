import json
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from .model import build_model
# from sabotage.model.callback import ValidateCallback, BackupAndRestoreCheckpoint

print("========================= Tensorflow =========================")
print("GPUs availables: {}".format(len(tf.config.list_physical_devices('GPU'))))
print("==============================================================")


def run(args: object):
    print(args)

    with open(args.metadata_path, 'r') as f:
        metadata = json.loads(f.read())
        print(metadata)

    with open(args.labels_path, 'r') as f:
        labels = json.loads(f.read())

    levels_size = {'level1': labels['label_1_count'] ,
                   'level2': labels['label_2_count'] ,
                   'level3': labels['label_3_count'] ,
                   'level4': labels['label_4_count'] }

    params: dict = {
        'levels_size': levels_size,
        'sequence_size': metadata['sequence_size'],
        'dropout': args.dropout
    }

    print(params)
    model = build_model(**params)
    
    tf.keras.utils.plot_model(
        model,
        to_file="model.png",
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
        layer_range=None,
        show_layer_activations=False,
        show_trainable=False,
    )
    
    
#     ds_train = Dataset(args.trainset_pattern, args.epochs, args.batch_size).build(df=False)
#     ds_validation = Dataset(args.valset_pattern, args.epochs, args.batch_size).build(df=False)
#     callbacks = [EarlyStopping(monitor='loss', patience=args.patience, verbose=1)]
#     model.fit(ds_train,
#               validation_data=ds_validation,
#               steps_per_epoch=metadata['trainset_count'] // args.batch_size,
#               validation_steps=metadata['validationset_count'] // args.batch_size,
#               epochs=args.epochs,
#               max_queue_size=args.max_queue_size,
#               workers=cpu_count(),
#               use_multiprocessing=True,
#               callbacks=callbacks)
