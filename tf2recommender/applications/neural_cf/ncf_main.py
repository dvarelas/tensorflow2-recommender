import pandas as pd
import tensorflow as tf

from tf2recommender.models.ncf import NeuralCF
from tf2recommender.utils.indexers import ColumnIndexer


def main(config):
    training_args = config['training_args']
    col_names = config['col_names']

    # read in the train set
    train = pd.read_csv(
        'tf2recommender/data/u1_base.csv',
        sep='\t',
        header=None,
        usecols=[0, 1, 2],
        names=[col_names.user_col, col_names.item_col, col_names.rating_col])

    # read in the test set
    test = pd.read_csv(
        'tf2recommender/data/u1_test.csv',
        sep='\t',
        header=None,
        usecols=[0, 1, 2],
        names=[col_names.user_col, col_names.item_col, col_names.rating_col])

    # drop users and items that do not exist in the training set
    test = test[test['item'].isin(train['item'].values)]

    # instantiate the column index for both user and items
    indexer = ColumnIndexer(train, ['user', 'item'])

    # index the train set
    train = indexer.transform(train)
    # index the test set
    test = indexer.transform(test)

    # get the number of distinct users and items
    number_of_users = len(set(train[col_names.user_col].values))
    number_of_items = len(set(train[col_names.item_col].values))

    # create user item rating tuples
    train_users_items_ratings = ((
        train[col_names.user_col + '_indexed'].values,
        train[col_names.item_col + '_indexed'].values),
                                 train[col_names.rating_col].values)

    test_users_items_ratings = ((test[col_names.user_col + '_indexed'].values,
                                 test[col_names.item_col + '_indexed'].values),
                                test[col_names.rating_col].values)

    # instantiate the tf datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_users_items_ratings)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_users_items_ratings)

    train_batches = train_dataset.shuffle(1000).batch(training_args.batch_size)
    test_batches = test_dataset.batch(training_args.batch_size)

    ncf = NeuralCF(number_of_users, number_of_items, training_args.user_dim,
                   training_args.item_dim, training_args.hidden1_dim,
                   training_args.hidden2_dim)

    model = ncf.model()

    print(model.summary())

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    model.fit(train_batches, epochs=training_args.num_epochs)

    print('\n# Evaluate')
    print(model.evaluate(test_batches))
