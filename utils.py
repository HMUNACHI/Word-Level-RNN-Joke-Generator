import re
import numpy as np
import cPickle as pickle
from os.path import join


def clean_punctuation(joke):
    tokens = re.findall(r"[\w']+|[.,!?;]+", joke)
    cleaned = []
    for token in tokens:
        if '?' in token:
            cleaned.append('?')
        elif '!' in token:
            cleaned.append('!')
        elif '..' in token:
            cleaned.append('...')
        else:
            cleaned.append(token)
    if '.' not in cleaned[-1] and '?' not in cleaned[-1] and '!' not in cleaned[-1]:
        cleaned.append('.')
    return " ".join(cleaned)


def generate_overlapping_encoded_sequences(jokes, maxlen, step):
    sentences = []
    next_words = [] # holds the targets
    for joke in jokes:
        for i in range(0, len(joke) - maxlen, step):
            sentences.append(joke[i: i + maxlen])
            next_words.append(joke[i + maxlen])
    return sentences, next_words


def load_short_jokes_corpus(data_path=['data', 'short-jokes.pkl'], limit=None):
    file_path = reduce(join, data_path)
    with open(file_path, 'rb') as f:
        if limit is None:
            jokes = pickle.load(f)
        else:
            jokes = pickle.load(f)[:limit]

        corpus = " ".join(jokes)
        return map(lambda j: j.lower(), jokes), corpus
