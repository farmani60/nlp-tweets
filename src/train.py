import gensim.downloader as api
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn import metrics, model_selection

from src import config
from src.data_cleaning import relabel_target
from src.dataset import TweetDataset
from src.engine import evaluate, train
from src.lstm import LSTM
from src.preprocessing import clean_tweet


def load_embedding_matrix(corpus,  gensim_pretrained_emb):
    print("Loading embedding vectors...")
    vectors = api.load(gensim_pretrained_emb)
    embedding_weights = np.zeros((config.VOCAB_SIZE, config.EMBED_SIZE))
    for word, i in corpus:
        if word in vectors:
            embedding_weights[i] = vectors[word]
    return embedding_weights


def create_folds():
    print("Load data...")
    train_df = pd.read_csv(config.ORIGINAL_TRAIN_DATA)
    test_df = pd.read_csv(config.ORIGINAL_TEST_DATA)

    # relabel some tweets
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

    return train_df, test_df


def run(df, fold):
    # fetch training dataframe
    train_df = df[df.k_fold != fold].reset_index(drop=True)

    # fetch validation dataframe
    valid_df = df[df.k_fold == fold].reset_index(drop=True)

    print("Fitting tokenizer...")
    # we use tf.keras for tokenization
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(train_df[config.CLEANED_TEXT])

    # model_path = f"{config.MODEL_DIR}/PRETRAIN_WORD2VEC_{model}/"
    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)

    # with open(f'{model_path}tokenizer.pkl', 'wb') as handle:
    #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # convert training and valid data to sequences
    x_train = tokenizer.texts_to_sequences(train_df[config.CLEANED_TEXT].values)
    x_test = tokenizer.texts_to_sequences(valid_df[config.CLEANED_TEXT].values)

    # zero pad the training sequences given the maximum length
    # this padding is done on left hand side
    # if sequence is > MAX_LEN, it is truncated on left hand side too
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=config.MAX_LEN
    )
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=config.MAX_LEN
    )

    # initialize dataset class for training
    train_dataset = TweetDataset(
        tweets=x_train,
        targets=train_df.target.values
    )

    # create torch dataloader for training
    # torch dataloader loads the data using dataset
    # class in batches specified by batch size
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=2
    )

    # initialize dataset class for validation
    valid_dataset = TweetDataset(
        tweets=x_test,
        targets=valid_df.target.values
    )

    # create torch dataloader for validation
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        num_workers=1
    )

    # load embedding vectors
    embedding_matrix = load_embedding_matrix(tokenizer.word_index.items(), config.PRETRAINED_WORD2VEC)

    # create torch device for using gpu
    # device = torch.device("cuda")

    # fetch LSTM model
    model = LSTM(embedding_matrix)

    # send model to device
    # model.to(device)

    # initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("Training Model...")
    # set the accuracy to zero
    best_accuracy = 0.
    # set early stopping counter to zero
    early_stopping_counter = 0
    # train and validate for all epochs
    for epoch in range(config.N_EPOCHS):
        # train one epoch
        train(train_data_loader, model, optimizer, device=None)
        # validate
        predictions, targets = evaluate(valid_data_loader, model, device=None)
        predictions = torch.tensor(predictions)
        targets = np.array(targets)
        # this threshold should be done after using sigmoid
        prediction_classes = np.array(1 * (torch.sigmoid(predictions) >= config.THRESHOLD))
        # calculate accuracy
        accuracy = metrics.accuracy_score(targets, prediction_classes)
        print(
            f"FOLD: {fold}, EPOCH: {epoch}, Accuracy Score = {accuracy}"
        )
        # simple early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1

        if early_stopping_counter > config.MAX_EARLY_STOPPING_COUNTER:
            break


if __name__ == "__main__":

    train_df, test_df = create_folds()

    for fold in range(config.N_FOLDS):
        run(df=train_df, fold=fold)
