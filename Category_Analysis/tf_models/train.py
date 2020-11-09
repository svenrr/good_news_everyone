# utils
from time import time
import numpy as np
import matplotlib.pyplot as plt
import os

# Arguments
import argparse
from configs.config_utils import get_config

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses

# enable warnings in tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed


# Text processing
from text_processing import preprocess_text,\
                            load_ds_from_files,\
                            load_ds_from_numpy

# Models
from models.model1 import model1, rnn1
from models.transfer_models import nnlm_en_dim50


def train(train_data=None, train_labels=None,
          test_data=None, test_labels=None,
          config='default.yml',
          dropout=None, embedding_dim=None, vocab_size=None,
          sequence_length=None, regularization=None, tmp=64):

    # get total path, in case function is called from somewhere else
    total_path = os.path.dirname(os.path.realpath(__file__)) + '/'

    config = get_config(total_path + 'configs/' + config)
    config.total_path = total_path


    # Changes in config due to python input
    if dropout != None: config.dropout = dropout
    if embedding_dim != None: config.embedding_dim = embedding_dim
    if vocab_size != None: config.vocab_size = vocab_size
    if sequence_length != None: config.sequence_length = sequence_length
    if regularization != None: config.regularization = regularization
    if tmp != None: config.tmp = tmp


    ### Print general information ###
    print('Tensorflow version {}'.format(tf.__version__))


    ### Load Datasets ###
    print('Load Datasets')
    if config.load_from_file:
        train_ds, val_ds, test_ds = load_ds_from_files(config)
    else:
        train_ds, val_ds, test_ds = load_ds_from_numpy(
            train_data, train_labels,
            test_data, test_labels, config)



    ### Preprocess datasets ###
    if config.preprocess:
        vectorize_layer = preprocess_text(train_ds, config)


    ### Autotune ###
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


    ### Regularization ###
    if config.regularizer == 'l1':
        regularizer = tf.keras.regularizers.l1(config.regularization)
    elif config.regularizer == 'l2':
        regularizer = tf.keras.regularizers.l1(config.regularization)


    ### Create model ###
    if config.model == 'model1':
        model = model1(config, vectorize_layer, regularizer)
    elif config.model == 'rnn1':
        model = rnn1(config, vectorize_layer)
    elif config.model == 'nnlm_en_dim50':
        model = nnlm_en_dim50(train_data, config)


    ### Model compilation ###
    # Loss
    if config.loss == 'binary':
        loss = losses.BinaryCrossentropy(from_logits=True)
    elif config.loss == 'categorical':
        loss = losses.CategoricalCrossentropy()
    else:
        print('Chosen loss is not valid!!!\nUse "binary" or "categorical"')
        exit()

    # Metric
    if config.metric == 'binary_accuracy':
        metric = tf.metrics.BinaryAccuracy(threshold=0.0)
    elif config.metric == 'categorical_accuracy':
        metric = tf.metrics.CategoricalAccuracy()
    else:
        print('Chosen metric is not valid!!!\nUse "binary_accuracy" or "categorical_accuracy"')
        exit()

    model.compile(loss=loss,
              optimizer='adam',
              metrics=metric)


    ### Model fit ###
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        verbose=config.print_fit)


    loss, accuracy = model.evaluate(
        test_ds,
        verbose=config.print_fit)

    if config.print_test:
        print("Loss: ", loss)
        print("Accuracy: ", accuracy)


    ### Plot training ###
    history_dict = history.history
    history_dict.keys()

    acc = history_dict[config.metric]
    val_acc = history_dict['val_' + config.metric]
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    if config.print_plots:

        epochs = range(1, len(acc) + 1)

        _, ax = plt.subplots(1,2, figsize=(15,6))
        # "bo" is for "blue dot"
        ax[0].plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        ax[0].plot(epochs, val_loss, 'b', label='Validation loss')
        ax[0].set_title('Training and validation loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        ax[1].plot(epochs, acc, 'bo', label='Training acc')
        ax[1].plot(epochs, val_acc, 'b', label='Validation acc')
        ax[1].set_title('Training and validation accuracy')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend(loc='lower right')

        plt.show()


    #examples = [
    #"The movie was great!",
    #"The movie was okay.",
    #"The movie was terrible..."
    #]

    #model.predict(examples)


    # Save model
    if config.save_model:
        model.save_weights('./checkpoints/my_checkpoint')


    # Return final accuracies of train, val, and test
    return acc[-1], val_acc[-1], accuracy