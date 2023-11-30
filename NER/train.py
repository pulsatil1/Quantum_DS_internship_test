from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import config
import utils


df = pd.read_csv(config.dataset_path)

sentences = df.groupby("Sentence")["Word"].apply(list).values
labels = df.groupby("Sentence")["Label"].apply(list).values

train_data, val_data, train_labels, val_labels = train_test_split(
    sentences, labels, test_size=0.2, random_state=config.RANDOM_STATE)

y_train = utils.pad_labels(train_labels)
y_val = utils.pad_labels(val_labels)

vocab_size = df["Word"].nunique()

# Fit Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=config.oov)
tokenizer.fit_on_texts(train_data)
tokenizer.fit_on_texts(val_data)

# Converting words into token sequences
train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(val_data)

X_train = utils.pad_data(train_sequences)
X_val = utils.pad_data(test_sequences)

embedding_dim = 300
embedding_dim = 300
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(
        config.MAX_LEN, embedding_dim, input_length=config.MAX_LEN),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=100, return_sequences=True)),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=100, return_sequences=True)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              metrics=['accuracy'])

# Checkpointing
save_model = tf.keras.callbacks.ModelCheckpoint(filepath=config.weights_path,
                                                monitor='val_loss',
                                                save_weights_only=True,
                                                save_best_only=True,
                                                verbose=1
                                                )

history = model.fit(X_train, y_train, epochs=70, batch_size=32,
                    validation_data=(X_val, y_val), callbacks=[save_model], verbose=1)

model.save(config.model_path)

with open(config.tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
