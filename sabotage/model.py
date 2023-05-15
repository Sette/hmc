import tensorflow as tf
from keras import layers
from keras.layers import Dense, Dropout, Concatenate
from keras.optimizers import Adam

MUSIC_SHAPE = (200,200)

class OutputNormalization(layers.Layer):
    def call(self, x):
        return tf.one_hot(tf.math.argmax(x, axis=1), x.shape[1], dtype=x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


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



def build_cnn(input):
    x = layers.Conv1D(32, 3, activation='relu')(input)
    x = layers.MaxPooling1D()(x)
    # x = layers.Flatten()(x)
    
    
    x = layers.Conv1D(64, 3, activation='relu', padding="valid")(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Conv1D(128, 3, activation='relu', padding="valid")(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Flatten()(x)

    return x

# def build_classification(inputs, size, name,dropout=0.1):
#     x = layers.Concatenate(axis=1)(inputs)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Dense(1024, activation="relu")(x)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Dense(size, activation='softmax', name=name)(x)
#     return x

def build_model(levels_size, sequence_size, cnn_size=128, dropout=0.1, weights_path=None):
    
    
    music = tf.keras.layers.Input(shape=(512, sequence_size,), dtype=tf.float32, name="emb")
    
    
    cnn = build_cnn(music)
    # outputs = {}
    # input = x
    # for level_name,level_size in levels_size.items():
    #     out = build_classification(input, level_size, level_name,dropout)
    #     outputs[level_name] = out
    #     input = [out, x]

    ##
    # First Classification Level
    first_level_dense1 = Dense(cnn_size * 2, activation='relu')(cnn)
    first_level_dropout1 = Dropout(dropout)(first_level_dense1)
    first_level_dense2 = Dense(cnn_size, activation='relu')(first_level_dropout1)
    first_level_dropout2 = Dropout(dropout)(first_level_dense2)
    first_level_output = Dense(levels_size['first_level_output_size'], activation='softmax', name="first_level_output")(
        first_level_dropout2)

    ##
    # Create second level input
    second_level_input = Concatenate(axis=1)([OutputNormalization()(first_level_output), cnn])

    ##
    # Second Classification Level
    second_level_input_size = (cnn_size + levels_size['first_level_output_size'])
    second_level_dense1 = Dense(second_level_input_size * 2, activation='relu')(second_level_input)
    second_level_dropout1 = Dropout(dropout)(second_level_dense1)
    second_level_dense2 = Dense(second_level_input_size, activation='relu')(second_level_dropout1)
    second_level_dropout2 = Dropout(dropout)(second_level_dense2)
    second_level_output = Dense(levels_size['second_level_output_size'], activation='softmax', name="second_level_output")(
        second_level_dropout2)

    ##
    # Create third level input
    third_level_input = Concatenate(axis=1)([OutputNormalization()(second_level_output), cnn])

    ##
    # Third Classification Level
    third_level_input_size = (cnn_size + levels_size['second_level_output_size'])
    third_level_dense1 = Dense(third_level_input_size * 2, activation='relu')(third_level_input)
    third_level_dropout1 = Dropout(dropout)(third_level_dense1)
    third_level_dense2 = Dense(third_level_input_size, activation='relu')(third_level_dropout1)
    third_level_dropout2 = Dropout(dropout)(third_level_dense2)
    third_level_output = Dense(levels_size['third_level_output_size'], activation='softmax', name="third_level_output")(
        third_level_dropout2)

    ##
    # Create a fourth level input
    fourth_level_input = Concatenate(axis=1)([OutputNormalization()(third_level_output), cnn])

    ##
    # Fourth Classification Level
    fourth_level_input_size = (cnn_size + levels_size['third_level_output_size'])
    fourth_level_dense1 = Dense(fourth_level_input_size * 2, activation='relu')(fourth_level_input)
    fourth_level_dropout1 = Dropout(dropout)(fourth_level_dense1)
    fourth_level_dense2 = Dense(fourth_level_input_size, activation='relu')(fourth_level_dropout1)
    fourth_level_dropout2 = Dropout(dropout)(fourth_level_dense2)
    fourth_level_output = Dense(levels_size['fourth_level_output_size'], activation='softmax', name="fourth_level_output")(
        fourth_level_dropout2)

    # Create a fourth level input
    fifth_level_input = Concatenate(axis=1)([OutputNormalization()(fourth_level_output), cnn])

    ##
    # Fifth Classification Level
    fifth_level_input_size = (cnn_size + levels_size['fourth_level_output_size'])
    fifth_level_dense1 = Dense(fourth_level_input_size * 2, activation='relu')(fifth_level_input)
    fifth_level_dropout1 = Dropout(dropout)(fifth_level_dense1)
    fifth_level_dense2 = Dense(fifth_level_input_size, activation='relu')(fifth_level_dropout1)
    fifth_level_dropout2 = Dropout(dropout)(fifth_level_dense2)
    fifth_level_output = Dense(levels_size['fifth_level_output_size'], activation='softmax',
                                name="fifth_level_output")(
        fifth_level_dropout2)

    model = tf.keras.models.Model([music], [
        first_level_output,
        second_level_output,
        third_level_output,
        fourth_level_output,
        fifth_level_output,
    ], name="Model-Essentia")
    
    
    _load_weights(model, weights_path)
    
    
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model