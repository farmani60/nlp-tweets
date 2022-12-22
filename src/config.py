# data
DATA_DIR = "../data/"
EMBEDDINGS_DIR = "../embeddings/"
ORIGINAL_TRAIN_DATA = DATA_DIR + "train.csv"
ORIGINAL_TEST_DATA = DATA_DIR + "test.csv"
MODIFIED_TRAIN = DATA_DIR + "modified_train.csv"
MODIFIED_TEST = DATA_DIR + "modified_test.csv"
SUBMISSION = DATA_DIR + "sample_submission.csv"
MODEL_DIR = "../models/"

# target
TARGET = "target"
RELABELED_TARGET = "relabeled_target"

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
TEXT_WITH_KEYWORD = "text_with_keyword"

# Pretrained vectors
# can find pretrained embeddings:
# import gensim.downloader as api
# api.info()
PRETRAINED_WORD2VEC = "word2vec-google-news-300"
PRETRAINED_GLOVE = "glove-wiki-gigaword-300"
GLOVE_EMBEDDINGS = EMBEDDINGS_DIR + "glove.840B.300d.pkl"
EMBED_SIZE = 300

# training
MAX_LEN = 128
VOCAB_SIZE = 172901
N_EPOCHS = 10
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 8
LEARNING_RATE = 0.001
N_FOLDS = 5

# validation
THRESHOLD = 0.5

# stopping
MAX_EARLY_STOPPING_COUNTER = 2
