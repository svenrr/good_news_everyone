import tensorflow as tf

from tensorflow.keras import layers

def model1(config, vectorize_layer, regularizer):

    model = tf.keras.Sequential([
        vectorize_layer,
        layers.Embedding(config.vocab_size + 1, config.embedding_dim),
        layers.Dropout(config.dropout),
        layers.GlobalAveragePooling1D(),
        #layers.Flatten(),
        layers.Dropout(config.dropout),
        layers.Dense(config.tmp, 
                     activation='relu',
                     kernel_regularizer=regularizer),
        layers.Dropout(config.dropout),
        layers.Dense(7,
                     activation='softmax',
                     kernel_regularizer=regularizer)
    ])

    if config.print_model:
        model.summary()

    return model


def rnn1(config, vectorize_layer):

    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(
            input_dim=len(vectorize_layer.get_vocabulary()),
            output_dim=config.embedding_dim,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        layers.Bidirectional(tf.keras.layers.LSTM(config.tmp)),
        layers.Dropout(config.dropout),
        layers.Dense(64, activation='relu'),
        layers.Dropout(config.dropout),
        layers.Dense(7, activation='softmax')
    ])


    if config.print_model:
        model.summary()

    return model