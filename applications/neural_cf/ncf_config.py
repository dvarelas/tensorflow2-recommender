from config.local import ColumnNames


class NcfTrainingArguments(object):
    def __init__(self, user_dim, item_dim, batch_size, num_epochs, hidden1_dim,
                 hidden2_dim):
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim


config = {
    'training_args':
    NcfTrainingArguments(
        user_dim=10,
        item_dim=6,
        batch_size=16,
        num_epochs=5,
        hidden1_dim=8,
        hidden2_dim=2),
    'col_names':
    ColumnNames(user_col='user', item_col='item', rating_col='rating')
}
