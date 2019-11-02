from tf2recommender.config.local import ColumnNames


class MfTrainingArguments(object):
    """
    Training arguments
    """

    def __init__(self, user_dim, item_dim, batch_size, num_epochs):
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs


config = {
    'training_args':
    MfTrainingArguments(user_dim=10, item_dim=5, batch_size=16, num_epochs=5),
    'col_names':
    ColumnNames(user_col='user', item_col='item', rating_col='rating')
}
