import numpy as np
import keras
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.optimizers import RMSprop
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
import random
import sys
from os.path import join

from utils import clean_punctuation, generate_overlapping_encoded_sequences, load_short_jokes_corpus

def load_and_clean_jokes():
    jokes, text = load_short_jokes_corpus(limit=None)
    jokes = map(clean_punctuation, jokes)
    text = " ".join(jokes)
    return jokes, text

jokes, text = load_and_clean_jokes()
tokenizer = Tokenizer(filters='"#$%&()*+,-/:;<=>@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(jokes)
index_to_word = dict( (index, word) for word, index in tokenizer.word_index.items())

def random_seed(integer_encoded_docs):
    rand_joke = random.choice(integer_encoded_docs)
    start_sequence = random.randint(0, len(rand_joke)-k)
    return filter(lambda x: x not in '.?!', decode([rand_joke[start_sequence: start_sequence + k]]))


def integer_encode(string_of_words):
    return tokenizer.texts_to_sequences([string_of_words])


def decode(integer_encoded, index_to_word=index_to_word):
    return " ".join(map(lambda key: index_to_word[key], integer_encoded[0]))


# returns tuple (highest_n_indices, highest_n_values) from model's prediction
def sample_n(preds, n=5):
    highest_indices = np.argpartition(preds, -n)[-n:]
    highest_values = map(lambda i: preds[i], highest_indices)
    return highest_indices, highest_values


# OUT: (seq, score)
def successors(integer_encoded_seed, prev_score, maxlen, k=5, index_to_word=index_to_word, readable=False):
    padded_seq = pad_sequences([integer_encoded_seed], maxlen=maxlen, padding='post')
    preds = model.predict(padded_seq, verbose=0)[0]
    highest_indices, highest_probs = sample_n(preds, n=k)
    scores = [prev_score * p for p in highest_probs]
    if readable:
        for index, score in zip(highest_indices, scores):
            yield index_to_word[index], score
    else:
        for index, score in zip(highest_indices, scores):
            yield index, score


def ends_with_stop_word(encoding):
     return decode([encoding]).split(' ')[-1] in ('.', '!', '?')


from Queue import PriorityQueue
def beam_search(seed, k=5):
    encoded_seed = integer_encode(seed)[0]
    pq = PriorityQueue()
    pq.put((-1.0, encoded_seed, 0))
    while pq:
        prev_score, encoding, depth = pq.get()
        # NOTE: COMMENT THIS IF YOU WOULD NOT LIKE TO SEE INTERMEDIATE RESULTS
        if depth >= 7:
            print decode([encoding])
        if depth >= 7 and ends_with_stop_word(encoding):
            return decode([encoding])
        for next_encoded_word, score in successors(encoding, prev_score, maxlen, k=k):
            # don't allow punctuation after punctuation
            if ends_with_stop_word(encoding) and ends_with_stop_word([next_encoded_word]):
                continue
            pq.put((score, encoding + [next_encoded_word], depth + 1))




if __name__ == '__main__':

    vocab_size = len(tokenizer.word_index) + 1

    # print('Vocab Size', vocab_size)

    # Encoding -- preprocessing for embedding layer
    integer_encoded_docs = tokenizer.texts_to_sequences(jokes)
    split_encoded_docs, next_words = generate_overlapping_encoded_sequences(integer_encoded_docs, 11, 3)
    padded_docs = pad_sequences(split_encoded_docs, padding='post')
    max_words, maxlen = 5, padded_docs.shape[1]

    # need to get next word for each of these
    next_words = np.asarray(next_words)
    # print("Number of Sequences:", len(padded_docs))

    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=padded_docs.shape[1], mask_zero=True))
    model.add(Bidirectional(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(2048, kernel_regularizer=l2(0.001), activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    model.load_weights(reduce(join,['models', 'finalWeights']))


    while True:
        seed = raw_input('Input a Seed: ')
        print 'Processing...'
        print 'Will display intermediate results so that you can see how this works!'
        print 'Output: {beam}'.format(beam=beam_search(clean_punctuation(seed)))

    # --- TO TRAIN A NEW MODEL ---
    # print 'Vectorization.'
    # y = np.zeros((len(padded_docs), vocab_size), dtype=np.bool)
    # for i, padded_doc in enumerate(padded_docs):
    #     y[i, next_words[i]] = 1
    #
    #
    # for i in range(0, 50):
    #     print 'Epoch: {i}'.format(i=i)
    #     model.fit(padded_docs, y, batch_size=2048, nb_epoch=1)
    #     if i > 4:
    #         model.save_weights('finalWeights'.format(i=i), overwrite=True)
    #     print '\n'
    #     for i in range(5):
    #         seed = random_seed(integer_encoded_docs)
    #         print 'Seed: {seed}'.format(seed=seed)
    #         print 'Generated: {out}'.format(out=beam_search(seed))
