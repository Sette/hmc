import tensorflow as tf
from keras.layers import Layer, Input, Dense, Dropout, Concatenate, BatchNormalization, Flatten, Reshape
from keras.optimizers import Adam
from keras import backend as K
import numpy as np

class OutputNormalization(tf.keras.layers.Layer):
    def call(self, x):
        # Obtemos a classe com a maior probabilidade
        one_hot_encoded = tf.one_hot(tf.math.argmax(x, axis=1), x.shape[1], dtype=x.dtype)
        return one_hot_encoded

    def compute_output_shape(self, input_shape):
        return input_shape

def build_classification(x, size, dropout, input_shape=1024, name='default'):
    x: object = Dense(input_shape, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(int(input_shape/2), activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(int(input_shape/4), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(size, activation='softmax', name=name)(x)

    return x


def build_model(levels_size: dict, sequence_size: int = 1280, dropout: float = 0.6) -> tf.keras.models.Model:
    """

    :rtype: tf.keras.models.Model
    """
    input_shape = (sequence_size,)
    music = Input(shape=input_shape, dtype=tf.float32, name="features")
    fcn_size = 1024

    outputs = []
    for level, size in enumerate(start=1, iterable=levels_size):
        level_name = f'level{level}'
        if level != 1:
            # Aplicar OutputNormalization na saída anterior
            output_normalized = BatchNormalization()(OutputNormalization()(current_output))
            current_input = Concatenate(axis=1)([output_normalized, music])
            current_output = build_classification(current_input, size, dropout, input_shape=fcn_size, name=level_name)
        else:
            current_input = music
            current_output = build_classification(current_input, size, dropout, input_shape=fcn_size, name=level_name)

        
        
        # Convert the tensor to a NumPy array
        outputs.append(current_output)
        print(level)
        print(current_output.shape)
        # Atualizar a saída anterior para a próxima iteração
        current_input = current_output

        


    model = tf.keras.models.Model(inputs=music, outputs=outputs, name="Essentia")

    #     _load_weights(model, weights_path)
    
    #optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']*4, run_eagerly=True)

    return model