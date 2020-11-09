import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
#import tensorflow_datasets as tfidf
from time import time

def nnlm_en_dim50(train_examples, config):
    
    print('Load model from hub')
    start = time()
    hub_layer = hub.KerasLayer(
        'tf_models/pretrained_models/tf2-preview_nnlm-en-dim128_1',
        output_shape=[128], input_shape=[],
        dtype=tf.string, trainable=False)
    print('Model loaded in {:.2f}s'.format(time() - start))
    #train_set, _ = tfidf.as_numpy(train_examples)
    hub_layer(train_examples[:3])

    model = tf.keras.Sequential()
    model.add(hub_layer)
    layers.Dropout(config.dropout)
    layers.GlobalAveragePooling1D()
    model.add(layers.Dropout(config.dropout))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(config.dropout))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(config.dropout))
    model.add(layers.Dense(7, activation='softmax'))

    if config.print_model:
        model.summary()

    return model