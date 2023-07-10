import tensorflow as tf
from keras import layers
from keras.layers import Dense, Dropout, Concatenate, Input, Normalization
from keras.optimizers import Adam


class OutputNormalization(layers.Layer):
    def call(self, x, **kwargs):
        return tf.one_hot(tf.math.argmax(x, axis=1), x.shape[1], dtype=x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


def _load_weights(model, weights_path):
    print(model.summary())

    if weights_path and tf.io.gfile.exists(f"{weights_path}.index"):
        print(f"load weights for model {model.name}. weights_path={weights_path}")
        model.load_weights(weights_path)

    return model


def build_cnn(feature, input_shape):
    x: object = Normalization(input_shape=[input_shape, 1], axis=None)(feature)
    x = layers.Conv1D(128, 3, activation='relu', padding="valid")(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Conv1D(64, 3, activation='relu', padding="valid")(x)
    x = layers.MaxPooling1D()(x)
    # x = layers.Conv1D(32, 3, activation='relu', padding="valid")(x)
    # x = layers.MaxPooling1D()(x)
    x = layers.Flatten()(x)

    return x


def build_classification(x, levels_size, dropout, input_shape=1024, name='default'):
    x: object = Dense(input_shape, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(input_shape/2, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(input_shape/4, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(levels_size[name], activation='softmax', name=name+'_output')(x)

    return x


def build_model(levels_size: dict, sequence_size: int = 1280, dropout: float = 0.1) -> tf.keras.models.Model:
    """

    :rtype: tf.keras.models.Model
    """
    input_shape = (sequence_size, 1)
    music = Input(shape=input_shape, dtype=tf.float32, name="features")
    fcn_size = 1024

    x: object = build_cnn(music, input_shape)

    first = build_classification(x, levels_size, dropout, input_shape=fcn_size, name='level1')
    second_input = Concatenate(axis=1)([OutputNormalization()(first), x])
    second = build_classification(second_input, levels_size, dropout, input_shape=fcn_size+levels_size['level1'], name='level2')
    third_input = Concatenate(axis=1)([OutputNormalization()(second), x])
    third = build_classification(third_input, levels_size, dropout, input_shape=fcn_size+levels_size['level2'], name='level3')
    fourth_input = Concatenate(axis=1)([OutputNormalization()(third), x])
    fourth = build_classification(fourth_input, levels_size, dropout, input_shape=fcn_size+levels_size['level3'], name='level4')
    fifth_input = Concatenate(axis=1)([OutputNormalization()(fourth), x])
    fifth = build_classification(fifth_input, levels_size, dropout, input_shape=fcn_size+levels_size['level4'], name='level5')

    model = tf.keras.models.Model([music], [
        first,
        second,
        third,
        fourth,
        fifth,
    ], name="Essentia")

    #     _load_weights(model, weights_path)

    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model



# Definição do modelo
def build_hierarchical_model(num_nodes_per_level: list, num_classes_per_node: list, sequence_size: int = 1280, dropout: float = 0.1) -> tf.keras.models.Model:
    """

    :rtype: tf.keras.models.Model
    """
    input_shape = (sequence_size, 1)
    music = Input(shape=input_shape, dtype=tf.float32, name="features")
    fcn_size = 1024


    # # Camadas convolucionais compartilhadas
    # inputs = tf.keras.Input(shape=input_shape)
    # conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
    # conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(conv1)
    # # ...

    x: object = build_cnn(music, input_shape)
    
    
    
    # Camadas para cada nível da hierarquia
    level_outputs = []
    prev_level_output = x

    for i in range(len(num_nodes_per_level)):
        level_node_outputs = []
        for j in range(len(num_nodes_per_level[i])):
            # node_conv = layers.Conv1D(128, 3, activation='relu', padding="valid")(prev_level_output)
            # node_flatten = layers.Flatten()(node_conv)
            node_dense = layers.Dense(256, activation="relu")(prev_level_output)
            node_output = layers.Dense(num_nodes_per_level[i][j], activation="softmax")(node_dense)
            level_node_outputs.append(node_output)
    
        prev_level_output = layers.concatenate(level_node_outputs)
        level_outputs.extend(level_node_outputs)

    # Concatenação das saídas de todos os níveis
    merged_output = layers.concatenate(level_outputs)

    # Lista de funções de perda por nó
    loss_functions = []
    for i in range(len(num_nodes_per_level)):
        for j in range(len(num_nodes_per_level[i])):
            loss = tf.keras.losses.CategoricalCrossentropy()
            loss_functions.append(loss)

    # Função de perda total
    def hierarchical_loss(y_true, y_pred):
        losses = [loss_fn(y_true[:, i], y_pred[:, i]) for i, loss_fn in enumerate(loss_functions)]
        weighted_losses = [loss * weight for loss, weight in zip(losses, loss_weights)]
        return tf.reduce_sum(weighted_losses)

    # Modelo final
    model = tf.keras.Model(inputs=music, outputs=merged_output)
    model.compile(optimizer="adam", loss=hierarchical_loss, metrics=["accuracy"])

    return model

# Chamada do modelo:
# # Defina o número de nós por nível, o número de classes por nó e os pesos das perdas
# num_nodes_per_level = [3, 2, 4]  # Exemplo: 3 nós no nível 1, 2 nós no nível 2, 4 nós no nível 3
# num_classes_per_node = [[5, 4, 6], [3, 2], [4, 5, 3, 2]]  # Exemplo: classes para cada nó em cada nível
# loss_weights = [1.0, 0.5, 0.3]  # Pesos das perdas em cada nível
#
# # Construa o modelo hierárquico
# model = hierarchical_model(num_nodes_per_level, num_classes_per_node, loss_weights)





