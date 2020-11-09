import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import string
import re
import numpy as np
from sklearn.utils import shuffle

def preprocess_text(train_ds, config):

    def custom_standardization(input_data):

        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')


    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=config.vocab_size,
        output_mode='int',
        output_sequence_length=config.sequence_length)


    # Make a text-only dataset (without labels), then call adapt
    train_text = train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    return vectorize_layer


def load_ds_from_numpy(train_data_full, train_labels_full,
                       test_data, test_labels, config):

    # Convert labels to numbers
    d = dict()
    count = 0
    for i in train_labels_full.unique():
        d[i] = count
        count += 1
    train_labels_full = [d[i] for i in train_labels_full]
    test_labels = [d[i] for i in test_labels]
    depth = len(d)

    # Split into train, val set
    train_size = int((1-config.train_val_split)*train_data_full.shape[0])

    # Shuffle data
    train_data_full, train_labels_full = shuffle(train_data_full, train_labels_full, random_state=42)

    train_data = train_data_full[:train_size]
    train_labels = train_labels_full[:train_size]
    val_data = train_data_full[train_size:]
    val_labels = train_labels_full[train_size:]

    # Turn labels to one hot encodings
    train_labels = tf.one_hot(train_labels, depth)
    val_labels = tf.one_hot(val_labels, depth)
    test_labels = tf.one_hot(test_labels, depth)

    # Turn numpy arrays to Tensors
    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_ds = train_ds.batch(config.batch_size)

    val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_ds = val_ds.batch(config.batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    test_ds = test_ds.batch(config.batch_size)

    # Set class names
    train_ds.class_names = d.keys()
    val_ds.class_names = d.keys()
    test_ds.class_names = d.keys()

    return train_ds, val_ds, test_ds

def load_ds_from_files(config):

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        config.train_ds_dir, 
        batch_size=config.batch_size, 
        validation_split=0.2, 
        subset='training', 
        seed=config.seed)

    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        config.train_ds_dir, 
        batch_size=config.batch_size, 
        validation_split=0.2, 
        subset='validation', 
        seed=config.seed)

    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        config.test_ds_dir, 
        batch_size=config.batch_size)


    ### Get label names
    if config.print_samples:
        for text_batch, label_batch in raw_train_ds.take(1):
            for i in range(3):
                print("Review", text_batch.numpy()[i])
                print("Label", label_batch.numpy()[i])

    count = 0
    for i in raw_train_ds.class_names:
        print("Label {} corresponds to".format(count), i)
        count += 1

    return raw_train_ds, raw_val_ds, raw_test_ds