import gc
import operator

import gensim.downloader as api
import numpy as np
import pandas as pd

from src import config
from src.new_data_cleaning import clean


def build_vocab(X):

    tweets = X.apply(lambda s: s.split()).values
    vocab = {}

    for tweet in tweets:
        for word in tweet:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1

    return vocab


def check_embeddings_coverage(X, embeddings):
    vocab = build_vocab(X)

    covered = {}
    oov = {}
    n_covered = 0
    n_oov = 0

    for word in vocab:
        try:
            covered[word] = embeddings[word]
            n_covered += vocab[word]
        except:
            oov[word] = vocab[word]
            n_oov += vocab[word]

    vocab_coverage = len(covered) / len(vocab)
    text_coverage = (n_covered / (n_covered + n_oov))

    sorted_oov = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    return sorted_oov, vocab_coverage, text_coverage


if __name__ == "__main__":
    print("Load data...")
    train_df = pd.read_csv(config.ORIGINAL_TRAIN_DATA)
    test_df = pd.read_csv(config.ORIGINAL_TEST_DATA)

    train_df[config.CLEANED_TEXT] = train_df['text'].apply(lambda s: clean(s))
    test_df[config.CLEANED_TEXT] = test_df['text'].apply(lambda s: clean(s))

    train_df.to_csv(config.MODIFIED_TRAIN, index=False)
    test_df.to_csv(config.MODIFIED_TRAIN, index=False)

    train_df = pd.read_csv(config.MODIFIED_TRAIN)
    test_df = pd.read_csv(config.MODIFIED_TEST)

    glove_embeddings = np.load(config.GLOVE_EMBEDDINGS, allow_pickle=True)

    vectors = api.load(config.PRETRAINED_WORD2VEC)
