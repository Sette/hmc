import tensorflow as tf
from keras import layers
from keras.layers import Dense, Dropout, Concatenate, Bidirectional, LSTM, Input, Reshape, BatchNormalization, Flatten, Normalization
from keras.optimizers import Adam


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



class OneHotLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.one_hot(tf.math.argmax(x, axis=1), x.shape[1], dtype=x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

def build_bilstm(emb,lstm_size):
    # Create a bidirecional LSTM Layer
#     lstm = Bidirectional(CuDNNLSTM(lstm_size, return_sequences=True))(emb)
#     lstm = Bidirectional(CuDNNLSTM(lstm_size))(lstm)
    # Definir o shape da entrada (uma dimensão)
    input_shape = (1280, 1)  # None para permitir sequências de comprimento variável
    lstm = LSTM(64, input_shape=input_shape)(emb)

    return lstm


def build_cnn(feature,input_shape):
    x = layers.Conv1D(32, 3, activation='relu',input_shape=input_shape)(feature)
    # x = OutputNormalization()(x)
    # x = layers.MaxPooling1D(pool_size=1278, strides=1)(x)
    
    
    # x = layers.Conv1D(64, 3, activation='relu', padding="valid")(x)
    # x = layers.MaxPooling1D()(x)
    # x = layers.Conv1D(128, 3, activation='relu', padding="valid")(x)
    # x = layers.MaxPooling1D()(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)

    return x

def build_classification(inputs, size, name):
    x = tf.keras.layers.Concatenate(axis=1)(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(size-1, activation='softmax', name=name)(x)

    return x



def build_model(levels_size, sequence_size, batch_size=32, cnn_size=128,lstm_size=256, dropout=0.1, weights_path=None):

    input_shape = (1280,1)
    
    music = Input(shape=input_shape, dtype=tf.float32, name="features")

    x = Normalization(input_shape=[1280,1], axis=None)(music)

    x = layers.Conv1D(128, 3, activation='relu', padding="valid")(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Conv1D(64, 3, activation='relu', padding="valid")(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Conv1D(32, 3, activation='relu', padding="valid")(x)
    x = layers.MaxPooling1D()(x)

    
    x = layers.Flatten()(x)

    
    ## First Classification Level
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    first = Dense(levels_size['level1_size']+1, activation='softmax', name="level1")(x)
    

    ##
#     # Create second level input
#     second_level_input = Concatenate(axis=1)([OutputNormalization()(first_level_output), cnn])

#     ##
#     # Second Classification Level
#     second_level_input_size = (cnn_size + levels_size['first_level_output_size'])
#     second_level_dense1 = Dense(second_level_input_size * 2, activation='relu')(second_level_input)
#     second_level_dropout1 = Dropout(dropout)(second_level_dense1)
#     second_level_dense2 = Dense(second_level_input_size, activation='relu')(second_level_dropout1)
#     second_level_dropout2 = Dropout(dropout)(second_level_dense2)
#     second_level_output = Dense(levels_size['second_level_output_size'], activation='softmax', name="second_level_output")(
#         second_level_dropout2)

#     ##
#     # Create third level input
#     third_level_input = Concatenate(axis=1)([OutputNormalization()(second_level_output), cnn])

#     ##
#     # Third Classification Level
#     third_level_input_size = (cnn_size + levels_size['second_level_output_size'])
#     third_level_dense1 = Dense(third_level_input_size * 2, activation='relu')(third_level_input)
#     third_level_dropout1 = Dropout(dropout)(third_level_dense1)
#     third_level_dense2 = Dense(third_level_input_size, activation='relu')(third_level_dropout1)
#     third_level_dropout2 = Dropout(dropout)(third_level_dense2)
#     third_level_output = Dense(levels_size['third_level_output_size'], activation='softmax', name="third_level_output")(
#         third_level_dropout2)

#     ##
#     # Create a fourth level input
#     fourth_level_input = Concatenate(axis=1)([OutputNormalization()(third_level_output), cnn])

#     ##
#     # Fourth Classification Level
#     fourth_level_input_size = (cnn_size + levels_size['third_level_output_size'])
#     fourth_level_dense1 = Dense(fourth_level_input_size * 2, activation='relu')(fourth_level_input)
#     fourth_level_dropout1 = Dropout(dropout)(fourth_level_dense1)
#     fourth_level_dense2 = Dense(fourth_level_input_size, activation='relu')(fourth_level_dropout1)
#     fourth_level_dropout2 = Dropout(dropout)(fourth_level_dense2)
#     fourth_level_output = Dense(levels_size['fourth_level_output_size'], activation='softmax', name="fourth_level_output")(
#         fourth_level_dropout2)

#     # Create a fourth level input
#     fifth_level_input = Concatenate(axis=1)([OutputNormalization()(fourth_level_output), cnn])

#     ##
#     # Fifth Classification Level
#     fifth_level_input_size = (cnn_size + levels_size['fourth_level_output_size'])
#     fifth_level_dense1 = Dense(fourth_level_input_size * 2, activation='relu')(fifth_level_input)
#     fifth_level_dropout1 = Dropout(dropout)(fifth_level_dense1)
#     fifth_level_dense2 = Dense(fifth_level_input_size, activation='relu')(fifth_level_dropout1)
#     fifth_level_dropout2 = Dropout(dropout)(fifth_level_dense2)
#     fifth_level_output = Dense(levels_size['fifth_level_output_size'], activation='softmax',
#                                 name="fifth_level_output")(
#         fifth_level_dropout2)

    model = tf.keras.models.Model([music], [
        first,
        # second_level_output,
        # third_level_output,
        # fourth_level_output,
        # fifth_level_output,
    ], name="Essentia")
    
    
#     _load_weights(model, weights_path)
    
    
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model