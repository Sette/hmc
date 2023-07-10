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


def build_level_classifier(x, levels_size, dropout, input_shape=1024, name='default'):
    x: object = Dense(input_shape, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(input_shape/2, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(input_shape/4, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(levels_size[name], activation='softmax', name=name+'_output')(x)

    return x


def build_node_classifier(x, num_classes, node, level, dropout, input_shape=256):
    x: object = Dense(input_shape, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(num_classes, activation='softmax', name=f'{level}-{node}-local')(x)
    

    return x




# Definição do modelo
def build_hierarchical_model(num_nodes_per_level: list, num_classes_per_node: list, sequence_size: int = 1280, dropout: float = 0.1) -> tf.keras.models.Model:
    """

    :rtype: tf.keras.models.Model
    """
    input_shape = (sequence_size, 1)
    music = Input(shape=input_shape, dtype=tf.float32, name="features")
    fcn_size = 1024

    x: object = build_cnn(music, input_shape)
    
    
    # Camadas para cada nível da hierarquia
    level_outputs = []
    prev_level_output = x
    for level in range(len(num_nodes_per_level)):
        node_outputs = []
        num_nodes = len(num_nodes_per_level[level])
        num_classes = len(num_classes_per_node[level])
        
              
        for node in range(num_nodes):
            node_output = build_node_classifier(prev_level_output, num_classes, node, level, dropout, input_shape=256)
            node_outputs.append(node_output)

        prev_level_output = layers.concatenate(node_outputs,name=f'layer{level}')
        level_outputs.extend(node_outputs)

    # Concatenação das saídas de todos os níveis
    merged_output = layers.concatenate(level_outputs,name=f'global')

    # Lista de funções de perda por nó
    loss_functions = []
    for i in range(len(num_nodes_per_level)):
        for j in range(len(num_classes_per_node[i])):
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





