import tensorflow as tf
from keras.layers import Layer, Input, Dense, Dropout, Concatenate, BatchNormalization, Flatten, Reshape
from keras.optimizers import Adam
from keras import backend as K
import numpy as np

class OutputNormalization(Layer):
    def call(self, x):
        return tf.one_hot(tf.math.argmax(x, axis=1), x.shape[1], dtype=x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

def build_classification(x, size, dropout, input_shape=1024, name='default'):
    x: object = Dense(input_shape, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(int(input_shape/2), activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(int(input_shape/4), activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(size, activation='sigmoid', name=name)(x)

    return x


def build_model(levels_size: dict, sequence_size: int = 1280, dropout: float = 0.1) -> tf.keras.models.Model:
    """

    :rtype: tf.keras.models.Model
    """
    input_shape = (sequence_size,)
    music = Input(shape=input_shape, dtype=tf.float32, name="features")
    fcn_size = 1024

    outputs = []
    for level, size in levels_size.items():
        if level != 'level1':
            # Aplicar OutputNormalization na saída anterior
            #output_normalized = OutputNormalization()(prev_output)
            #print(output_normalized.shape)
            current_input = Concatenate(axis=1)([prev_output, music])
            current_output = build_classification(current_input, size, dropout, input_shape=sequence_size*2, name=level)
        else:
            current_input = music
            current_output = build_classification(current_input, size, dropout, input_shape=fcn_size, name=level)

        
        print(level)
        # Convert the tensor to a NumPy array
        outputs.append(current_output)
        # Atualizar a saída anterior para a próxima iteração
        prev_output = current_output
        


    model = tf.keras.models.Model(inputs=music, outputs=outputs, name="Essentia")

    #     _load_weights(model, weights_path)
    
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']*4, run_eagerly=True)

    return model