#!/usr/bin/env python

import tensorflow as tf
from tensorflow.compat.v1.keras.layers import Embedding, LSTM, CuDNNLSTM, Dropout, Dense, Input, Activation, Bidirectional
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import concatenate
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.layers import Layer


class OutputNormalization(Layer):
    def call(self, x):
        return tf.one_hot(tf.math.argmax(x, axis=1), x.shape[1], dtype=x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


def build_model_presal_v1(dept_size, seto_size, fami_size, subf_size, sequence_size, word_emb_size ,weights_path=None):
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
    
    


def build_model(
        first_level_output_size,
        second_level_output_size,
        third_level_output_size,
        fourth_level_output_size,
        dropout=0.1,
        embed_size=32):

    inp = Input(shape=(maxlen,), dtype='int32', name="features")

    ##
    # Create a word embedding vocab
    emb = Embedding(vocab_size + 1, embed_size, trainable=True)(inp)

    if tf.test.is_gpu_available():
        ##
        # Create a bidirecional LSTM Layer
        lstm = Bidirectional(CuDNNLSTM(lstm_size, return_sequences=True))(emb)
        lstm = Bidirectional(CuDNNLSTM(lstm_size))(lstm)
    else:
        ##
        # Create a bidirecional LSTM Layer
        lstm = Bidirectional(LSTM(lstm_size, return_sequences=True))(emb)
        lstm = Bidirectional(LSTM(lstm_size))(lstm)

    ##
    # First Classification Level
    first_level_dense1 = Dense(lstm_size * 2, activation='relu')(lstm)
    first_level_dropout1 = Dropout(dropout)(first_level_dense1)
    first_level_dense2 = Dense(lstm_size, activation='relu')(first_level_dropout1)
    first_level_dropout2 = Dropout(dropout)(first_level_dense2)
    first_level_output = Dense(first_level_output_size, activation='softmax', name="first_level_output")(
        first_level_dropout2)

    ##
    # Create second level input
    second_level_input = concatenate([OutputNormalization()(first_level_output), lstm], axis=1)

    ##
    # Second Classification Level
    second_level_input_size = (lstm_size + first_level_output_size)
    second_level_dense1 = Dense(second_level_input_size * 2, activation='relu')(second_level_input)
    second_level_dropout1 = Dropout(dropout)(second_level_dense1)
    second_level_dense2 = Dense(second_level_input_size, activation='relu')(second_level_dropout1)
    second_level_dropout2 = Dropout(dropout)(second_level_dense2)
    second_level_output = Dense(second_level_output_size, activation='softmax', name="second_level_output")(
        second_level_dropout2)

    ##
    # Create third level input
    third_level_input = concatenate([OutputNormalization()(second_level_output), lstm], axis=1)

    ##
    # Third Classification Level
    third_level_input_size = (lstm_size + second_level_output_size)
    third_level_dense1 = Dense(third_level_input_size * 2, activation='relu')(third_level_input)
    third_level_dropout1 = Dropout(dropout)(third_level_dense1)
    third_level_dense2 = Dense(third_level_input_size, activation='relu')(third_level_dropout1)
    third_level_dropout2 = Dropout(dropout)(third_level_dense2)
    third_level_output = Dense(third_level_output_size, activation='softmax', name="third_level_output")(
        third_level_dropout2)

    ##
    # Create a fourth level input
    fourth_level_input = concatenate([OutputNormalization()(third_level_output), lstm], axis=1)

    ##
    # Fourth Classification Level
    fourth_level_input_size = (lstm_size + third_level_output_size)
    fourth_level_dense1 = Dense(fourth_level_input_size * 2, activation='relu')(fourth_level_input)
    fourth_level_dropout1 = Dropout(dropout)(fourth_level_dense1)
    fourth_level_dense2 = Dense(fourth_level_input_size, activation='relu')(fourth_level_dropout1)
    fourth_level_dropout2 = Dropout(dropout)(fourth_level_dense2)
    fourth_level_output = Dense(fourth_level_output_size, activation='softmax', name="fourth_level_output")(
        fourth_level_dropout2)

    # Build model
    model = Model(inputs=inp, outputs=[
        first_level_output,
        second_level_output,
        third_level_output,
        fourth_level_output
    ])

    # Define Optimizer and loss metrics
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model