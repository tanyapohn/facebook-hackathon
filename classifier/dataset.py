import pandas as pd
import spacy
from torchtext import data


def parse_label(label: str) -> int:
    """
    Get the actual labels from label string
    :param label: labels of the form '__label__2'
    :return label (int) : integer value corresponding to label string
    """
    return int(label.strip()[-1])


def get_pandas_df(filename: str) -> pd.DataFrame:
    """
    Load the data into Pandas.DataFrame object
    This will be used to convert data to torchtext object

    """
    with open(filename, 'r') as datafile:
        data = [line.strip().split(',', maxsplit=1) for line in datafile]
        data_text = list(map(lambda x: x[1], data))
        data_label = list(map(lambda x: parse_label(x[0]), data))

    full_df = pd.DataFrame({"text": data_text, "label": data_label})
    return full_df


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}

    def load_data(self, train_file, test_file=None, val_file=None):
        """
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        :param train_file: path to training file
        :param test_file: path to test file
        :param val_file: path to validation file
        """

        NLP = spacy.load('en_core_web_sm')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

        # Creating Field for data
        text = data.Field(sequential=True, tokenize=tokenizer, lower=True,
                          fix_length=self.config.max_sen_len)
        label = data.Field(sequential=False, use_vocab=False)
        data_fields = [("text", text), ("label", label)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, data_fields) for i in
                          train_df.values.tolist()]
        train_data = data.Dataset(train_examples, data_fields)

        test_df = get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, data_fields) for i in
                         test_df.values.tolist()]
        test_data = data.Dataset(test_examples, data_fields)

        # If validation file exists, load it.
        # Otherwise get validation data from training data
        if val_file:
            val_df = get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, data_fields) for i in
                            val_df.values.tolist()]
            val_data = data.Dataset(val_examples, data_fields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)

        text.build_vocab(train_data)
        self.vocab = text.vocab

        self.train_iterator = data.BucketIterator(
            train_data,
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} test examples".format(len(test_data)))
        print("Loaded {} validation examples".format(len(val_data)))
