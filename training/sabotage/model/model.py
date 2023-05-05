import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2

from b2w.black_magic.visao.model.attention import Position_Embedding, TransformerBlock, Attention
from b2w.black_magic.visao.model.config import IMAGE_SHAPE

from tensorflow.keras.layers import GlobalAveragePooling2D


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


def build_classification(inputs, size, name):
    x = tf.keras.layers.Concatenate(axis=1)(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(size, activation='softmax', name=name)(x)

    return x


def build_model_dept(dept_size, seto_size, fami_size, subf_size, sequence_size, weights_path=None):
    print("build model dept")
    text = tf.keras.layers.Input(shape=(sequence_size, 128), dtype=tf.float32, name="name")
    image = tf.keras.layers.Input(shape=IMAGE_SHAPE, dtype=tf.float32, name="image")

    xname = Position_Embedding()(text)
    xname = TransformerBlock(128, 8, 32)(xname)
    xname = tf.keras.layers.Flatten()(xname)

    ximage = ResNet50V2(include_top=False, weights=image_weights(weights_path),
                        input_shape=IMAGE_SHAPE, pooling='avg')(image)

    dept = build_classification([xname, ximage], dept_size, "departamento")

    model = tf.keras.models.Model([text, image], [dept], name="ModelDepartamento")

    _load_weights(model, weights_path)

    return model


def build_model_outros(dept_size, seto_size, fami_size, subf_size, sequence_size, weights_path=None):
    print("build model outros")
    text = tf.keras.layers.Input(shape=(sequence_size, 128), dtype=tf.float32, name="name")
    image = tf.keras.layers.Input(shape=IMAGE_SHAPE, dtype=tf.float32, name="image")
    dept = tf.keras.layers.Input(shape=dept_size, dtype=tf.float32, name="dept")

    xname = Position_Embedding()(text)
    xname = TransformerBlock(128, 8, 32)(xname)
    xname = tf.keras.layers.Flatten()(xname)

    ximage = ResNet50V2(include_top=False, weights=image_weights(weights_path),
                        input_shape=IMAGE_SHAPE, pooling='avg')(image)

    seto = build_classification([xname, ximage, dept], seto_size, "setor")
    fami = build_classification([xname, ximage, dept, seto], fami_size, "familia")
    subf = build_classification([xname, ximage, dept, seto, fami], subf_size, "subfamilia")

    model = tf.keras.models.Model([dept, text, image], [seto, fami, subf], name="ModelOutros")

    _load_weights(model, weights_path)

    return model


def build_model_livro(dept_size, seto_size, fami_size, subf_size, sequence_size, weights_path=None):
    print("build model livro")
    text = tf.keras.layers.Input(shape=(sequence_size, 128), dtype=tf.float32, name="name")
    image = tf.keras.layers.Input(shape=IMAGE_SHAPE, dtype=tf.float32, name="image")

    xname = Position_Embedding()(text)
    xname = TransformerBlock(128, 8, 32)(xname)
    xname = tf.keras.layers.Flatten()(xname)

    ximage = ResNet50V2(include_top=False, weights=image_weights(weights_path),
                        input_shape=IMAGE_SHAPE, pooling='avg')(image)

    dept = build_classification([xname, ximage], dept_size, "departamento")
    seto = build_classification([xname, ximage, dept], seto_size, "setor")
    fami = build_classification([xname, ximage, dept, seto], fami_size, "familia")
    subf = build_classification([xname, ximage, dept, seto, fami], subf_size, "subfamilia")

    model = tf.keras.models.Model([text, image], [dept, seto, fami, subf], name="ModelLivros")

    _load_weights(model, weights_path)

    return model


def build_model_presal_poc(dept_size, seto_size, fami_size, subf_size, sequence_size, word_emb_size, weights_path=None):
    print("build model presal poc")
    text = tf.keras.layers.Input(shape=(sequence_size, word_emb_size), dtype=tf.float32, name="name")
    image = tf.keras.layers.Input(shape=IMAGE_SHAPE, dtype=tf.float32, name="image")

    xname = Position_Embedding()(text)
    xname = TransformerBlock(word_emb_size, 8, 32)(xname)
    xname = tf.keras.layers.Flatten()(xname)

    ximage = ResNet50V2(include_top=False, weights=image_weights(weights_path),
                        input_shape=IMAGE_SHAPE, pooling='avg')(image)

    dept = build_classification([xname, ximage], dept_size, "departamento")
    seto = build_classification([xname, ximage, dept], seto_size, "setor")
    fami = build_classification([xname, ximage, dept, seto], fami_size, "familia")
    subf = build_classification([xname, ximage, dept, seto, fami], subf_size, "subfamilia")

    model = tf.keras.models.Model([text, image], [dept, seto, fami, subf], name="ModelPresal")

    _load_weights(model, weights_path)

    return model


def build_model_presal_only_images_poc(dept_size, seto_size, fami_size, subf_size, sequence_size, word_emb_size,
                                       weights_path=None):
    print("build model presal poc")
    # text = tf.keras.layers.Input(shape=(sequence_size, word_emb_size), dtype=tf.float32, name="name")
    image = tf.keras.layers.Input(shape=IMAGE_SHAPE, dtype=tf.float32, name="image")

    # xname = Position_Embedding()(text)
    # xname = TransformerBlock(word_emb_size, 8, 32)(xname)
    # xname = tf.keras.layers.Flatten()(xname)

    ximage = ResNet50V2(include_top=False, weights=image_weights(weights_path),
                        input_shape=IMAGE_SHAPE, pooling='avg')(image)

    dept = build_classification([ximage], dept_size, "departamento")
    seto = build_classification([ximage, dept], seto_size, "setor")
    fami = build_classification([ximage, dept, seto], fami_size, "familia")
    subf = build_classification([ximage, dept, seto, fami], subf_size, "subfamilia")

    model = tf.keras.models.Model([image], [dept, seto, fami, subf], name="ModelImagesPresal")

    _load_weights(model, weights_path)

    return model


def build_model_presal_v1(dept_size, seto_size, fami_size, subf_size, sequence_size, word_emb_size, weights_path=None):
    print("build model presal poc")
    text = tf.keras.layers.Input(shape=(sequence_size, word_emb_size), dtype=tf.float32, name="name")
    image = tf.keras.layers.Input(shape=IMAGE_SHAPE, dtype=tf.float32, name="image")

    xname = Attention()(text)

    inception_resnet_v2 = ResNet50V2(include_top=False, weights=image_weights(weights_path), input_shape=IMAGE_SHAPE)
    for layer in inception_resnet_v2.layers[:-16]:
        layer.trainable = False
    cnn_model = inception_resnet_v2(image)

    ximage = GlobalAveragePooling2D()(cnn_model)

    dept = build_classification([xname, ximage], dept_size, "departamento")
    seto = build_classification([xname, ximage, dept], seto_size, "setor")
    fami = build_classification([xname, ximage, dept, seto], fami_size, "familia")
    subf = build_classification([xname, ximage, dept, seto, fami], subf_size, "subfamilia")

    model = tf.keras.models.Model([text, image], [dept, seto, fami, subf], name="ModelPresalv1")

    _load_weights(model, weights_path)

    return model