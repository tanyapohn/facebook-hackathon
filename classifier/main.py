import os

import torch
import torch.optim as optim
from torch import nn

from classifier.config import Config
from classifier.dataset import Dataset
from classifier.model import Transformer
from classifier.utils import evaluate_model

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PARENT_DIR, 'data')


def main():
    config = Config()
    train_file = os.path.join(DATA_DIR, 'fake_cleaned_train.csv')
    test_file = os.path.join(DATA_DIR, 'fake_cleaned_test.csv')
    val_file = os.path.join(DATA_DIR, 'fake_cleaned_val.csv')
    dataset = Dataset(config)
    dataset.load_data(train_file, test_file, val_file)

    model = Transformer(config, len(dataset.vocab))
    if torch.cuda.is_available():
        model.cuda()

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    model.add_optimizer(optimizer)
    model.add_loss_op(criterion)

    train_losses = []
    val_accuracies = []

    for epoch in range(config.max_epochs):
        print("Epoch: {}".format(epoch))
        train_loss, val_accuracy = model.run_epoch(
            dataset.train_iterator,
            dataset.val_iterator, epoch,
        )
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)

    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Test Accuracy: {:.4f}'.format(test_acc))


if __name__ == '__main__':
    main()
