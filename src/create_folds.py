import pandas as pd
import tensorflow as tf
from sklearn import model_selection

import config as config
from data_cleaning import relabel_target
from preprocessing import clean_tweet

if __name__ == "__main__":
    # read data
    print("Load data...")
    train_df = pd.read_csv(config.ORIGINAL_TRAIN_DATA)
    test_df = pd.read_csv(config.ORIGINAL_TEST_DATA)

    # ralable some tweets
    train_df = relabel_target(train_df)

    # clean tweets
    train_df[config.CLEANED_TEXT] = train_df[config.TEXT].apply(clean_tweet)
    test_df[config.CLEANED_TEXT] = test_df[config.TEXT].apply(clean_tweet)
    train_df[config.CLEANED_TEXT] = train_df[config.CLEANED_TEXT].astype(str)
    test_df[config.CLEANED_TEXT] = test_df[config.CLEANED_TEXT].astype(str)

    # create folds
    train_df["k_fold"] = -1
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(n_splits=config.N_FOLDS)
    for f, (t_, v_) in enumerate(kf.split(X=train_df, y=train_df.target.values)):
        train_df.loc[v_, "k_fold"] = f

    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(train_df[config.CLEANED_TEXT])

    print("Saving modified data...")
    train_df.to_csv(config.MODIFIED_TRAIN)
    test_df.to_csv(config.MODIFIED_TEST)
