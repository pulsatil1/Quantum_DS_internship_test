import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
import numpy as np
import pickle
import nltk
import config
import utils
nltk.download('punkt')


# sentence = 'The Rocky Mountains is an extensive mountain range located in western North America.'
sentence = input("Enter the sentence: ")
model = keras.models.load_model(config.model_path)
model.load_weights(config.weights_path)

with open(config.tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

words = list([nltk.word_tokenize(sentence)])
sequences_text = tokenizer.texts_to_sequences(words)
padded_text = utils.pad_data(sequences_text)

pred = model.predict(padded_text)

text = np.squeeze(words)
result = np.squeeze(pred)
indexes = np.argwhere(result > 0.5)
if indexes.shape[0] > 0:
    print("Found mountains:")
    for idx in indexes:
        print(text[idx[0]])
else:
    print("Mountains not found")
