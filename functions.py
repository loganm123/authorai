#import necessary modules
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras as keras
import gutenbergpy.textget
from knockknock import discord_sender
from webhook_url import *

def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)

#function to train the model
@discord_sender(webhook_url=webhook_url)
def train_model(model, dataset, epochs):
    history = model.fit(dataset, epochs=epochs)
    return history

def create_model(max_id):
    #create and compile our model
    model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
                        #dropout=0.2, recurrent_dropout=0.2),
                        dropout=0.2),
        keras.layers.GRU(128, return_sequences=True,
                        #dropout=0.2, recurrent_dropout=0.2),
                        dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                        activation="softmax"))
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model

def create_training_set(encoded):
    train_size = tokenizer.document_count * 90 // 100
    dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
    n_steps = 100
    window_length = n_steps + 1 # target = input shifted 1 character ahead
    dataset = dataset.window(window_length, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    np.random.seed(42)
    tf.random.set_seed(42)
    batch_size = 32
    dataset = dataset.shuffle(10000).batch(batch_size)
    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    max_id = len(tokenizer.word_index)
    dataset = dataset.map(
        lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
    dataset = dataset.prefetch(1)
    return dataset, max_id


#define function for preparing the book for us to train our model on
def get_clean_tokenize_encode(id):
    raw_book = gutenbergpy.textget.get_text_by_id(id)
    clean_book = gutenbergpy.textget.strip_headers(raw_book)
    book = str(clean_book, 'UTF-8')
    global tokenizer 
    tokenizer = keras.preprocessing.text.Tokenizer(char_level = True)
    tokenizer.fit_on_texts(book)
    global encoded
    [encoded]= np.array(tokenizer.texts_to_sequences([book]))-1
    return encoded

def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text