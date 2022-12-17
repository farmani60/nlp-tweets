# data
DATA_DIR = "../data/"
ORIGINAL_TRAIN_DATA = DATA_DIR + "train.csv"
ORIGINAL_TEST_DATA = DATA_DIR + "test.csv"
MODIFIED_TRAIN = DATA_DIR + "modified_train.csv"
MODIFIED_TEST = DATA_DIR + "modified_test.csv"
SUBMISSION = DATA_DIR + "sample_submission.csv"
MODEL_DIR = "../models/"

# target
TARGET = "target"
RELABELED_TARGET = "config.RELABELED_TARGET"

# features
ID = "id"
TEXT = "text"
KEYWORD = "keyword"
LOCATION = "location"
FOLD = "kfold"
TOKENS = "tokens"

# created features
ALL_TEXT = "all_text"
CLEANED_TEXT = "cleaned_text"

# Pretrained Word2Vec
PRETRAINED_WORD2VEC = "word2vec-google-news-300"
EMBED_SIZE = 300

# training
MAX_LEN = 202
VOCAB_SIZE = 172901
N_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
N_FOLDS = 5

# validation
THRESHOLD = 0.5

# stopping
MAX_EARLY_STOPPING_COUNTER = 2
