import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import config


def pad_labels(y):
    tag = list()
    for i in y:
        tag.append(np.array(i + [0] * (config.MAX_LEN-len(i))))

    return np.array(tag)


def pad_data(X):
    # Padded sequences to make them same length
    pad_sequence = pad_sequences(
        X, padding=config.padding, maxlen=config.MAX_LEN, truncating='post')

    return pad_sequence
