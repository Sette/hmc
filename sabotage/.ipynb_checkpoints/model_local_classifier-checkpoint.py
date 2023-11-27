import tensorflow as tf
from tensorflow.keras import layers

# Definição do modelo
def hierarchical_model(num_nodes_per_level, num_classes_per_node, loss_weights):
    input_shape = (input_dim,)  # Defina a dimensão de entrada adequada

    # Camadas convolucionais compartilhadas
    inputs = tf.keras.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
    conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(conv1)
    # ...

    # Camadas para cada nível da hierarquia
    level_outputs = []
    prev_level_output = conv2

    for i in range(len(num_nodes_per_level)):
        level_node_outputs = []
        for j in range(num_nodes_per_level[i]):
            node_conv = layers.Conv2D(128, kernel_size=(3, 3), activation="relu")(prev_level_output)
            node_flatten = layers.Flatten()(node_conv)
            node_dense = layers.Dense(256, activation="relu")(node_flatten)
            node_output = layers.Dense(num_classes_per_node[i][j], activation="softmax")(node_dense)
            level_node_outputs.append(node_output)

        prev_level_output = layers.concatenate(level_node_outputs)
        level_outputs.extend(level_node_outputs)

    # Concatenação das saídas de todos os níveis
    merged_output = layers.concatenate(level_outputs)

    # Lista de funções de perda por nó
    loss_functions = []
    for i in range(len(num_nodes_per_level)):
        for j in range(num_nodes_per_level[i]):
            loss = tf.keras.losses.CategoricalCrossentropy()
            loss_functions.append(loss)

    # Função de perda total
    def hierarchical_loss(y_true, y_pred):
        losses = [loss_fn(y_true[:, i], y_pred[:, i]) for i, loss_fn in enumerate(loss_functions)]
        weighted_losses = [loss * weight for loss, weight in zip(losses, loss_weights)]
        return tf.reduce_sum(weighted_losses)

    # Modelo final
    model = tf.keras.Model(inputs=inputs, outputs=merged_output)
    model.compile(optimizer="adam", loss=hierarchical_loss, metrics=["accuracy"])

    return model

# Defina o número de nós por nível, o número de classes por nó e os pesos das perdas
num_nodes_per_level = [3, 2, 4]  # Exemplo: 3 nós no nível 1, 2 nós no nível 2, 4 nós no nível 3
num_classes_per_node = [[5, 4, 6], [3, 2], [4, 5, 3, 2]]  # Exemplo: classes para cada nó em cada nível
loss_weights = [1.0, 0.5, 0.3]  # Pesos das perdas em cada nível

# Construa o modelo hierárquico
model = hierarchical_model(num_nodes_per_level, num_classes_per_node, loss_weights)

# Treinamento do modelo
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Avaliação do
