import re
import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from spellchecker import SpellChecker

from src import config

nltk.download("stopwords")


def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            word = corrected_text.append(spell.correction(word))
            if word is not None:
                corrected_text.append(word)
        else:
            if word is not None:
                corrected_text.append(word)
    corrected_text = [w for w in corrected_text if w is not None]
    return " ".join(corrected_text)


spell = SpellChecker()


def clean_tweet(tweet):
    # make text lower case
    tweet = tweet.lower()
    # remove stock market tickers like $GE
    tweet = re.sub(r"\$\w*", "", str(tweet))
    # remove old style retweet text "RT"
    tweet = re.sub(r"^RT[\s]+", "", str(tweet))
    # remove hyperlinks
    tweet = re.sub(r"https?:\/\/.*[\r\n]*", "", str(tweet))
    # remove html
    tweet = re.sub(r"<.*?>", "", str(tweet))
    # remove emojis
    tweet = re.sub(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        "",
        str(tweet),
    )
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r"#", "", str(tweet))
    # remove punctuation
    punct = set(string.punctuation)
    tweet = "".join(ch for ch in tweet if ch not in punct)
    # remove stopwords
    stop_words = set(stopwords.words("english"))
    tweet = " ".join(word for word in tweet.split() if word not in stop_words)
    # spelling correction
    tweet = correct_spellings(str(tweet))
    return tweet


def create_meta_features(df):
    # word_count
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    # unique_word_count
    df["unique_word_count"] = df["text"].apply(lambda x: len(set(str(x).split())))
    # stop_word_count
    df["stop_word_count"] = df["text"].apply(
        lambda x: len([w for w in str(x).lower().split() if w in stopwords.words("english")])
    )
    # url_count
    df["url_count"] = df["text"].apply(
        lambda x: len([w for w in str(x).lower().split() if "http" in w or "https" in w])
    )
    # mean_word_length
    df["mean_word_length"] = df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    # char_count
    df["char_count"] = df["text"].apply(lambda x: len(str(x)))
    # punctuation_count
    df["punctuation_count"] = df["text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    # hashtag_count
    df["hashtag_count"] = df["text"].apply(lambda x: len([c for c in str(x) if c == "#"]))
    # mention_count
    df["mention_count"] = df["text"].apply(lambda x: len([c for c in str(x) if c == "@"]))

    return df


def add_keyword_to_text(df):
    df[config.TEXT_WITH_KEYWORD] = df["keyword"].fillna("") + " " + df["text"]
    return df


def remove_repetition_subtext(df, col):
    # Remove character repetition (1...num times) from subtext list in the df[col]
    subtext = ["!", "#", "$", "(", ")", "*", "+", "-", ".", ":", ";", "=", "?", "@", "[", "]", "^", "|", "_", "{", "}"]
    num_chr_col_before = df[col].str.len().sum()
    for i in range(len(df)):
        text = df.loc[i, col]
        for c in subtext:
            while text.rfind(c + c) > -1:
                text = text.replace(c + c, c)
        df.loc[i, col] = text

    num_chr_col_after = df[col].str.len().sum()
    print(num_chr_col_before - num_chr_col_after, "characters removed")

    return df
