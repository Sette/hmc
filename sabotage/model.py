import tensorflow as tf
from keras import layers

MUSIC_SHAPE = (200,200)

def _load_weights(model, weights_path):
    print(model.summary())

    if weights_path and tf.io.gfile.exists(f"{weights_path}.index"):
        print(f"load weights for model {model.name}. weights_path={weights_path}")
        model.load_weights(weights_path)

    return model


def image_weights(weights_path):
    if weights_path and tf.io.gfile.exists(f"{weights_path}.index"):
        return None

    return 'imagenet'


class OneHotLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.one_hot(tf.math.argmax(x, axis=1), x.shape[1], dtype=x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape



def build_cnn(x):
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)

    return x

def build_classification(inputs, size, name):
    x = layers.Concatenate(axis=1)(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(size, activation='softmax', name=name)(x)

    return x

def build_model(level_size, weights_path=None):
    print("build model presal poc")
    music = tf.keras.layers.Input(shape=MUSIC_SHAPE, dtype=tf.float32, name="music")

    x = build_cnn(music)

    output = []
    for level_size ,level_name in level_size:
        out = build_classification([x], level_size, level_name)
        x = [x, out]
        output.append(out)

    model = tf.keras.models.Model([music], [output], name="ModelPresalv1")

    _load_weights(model, weights_path)

    return model