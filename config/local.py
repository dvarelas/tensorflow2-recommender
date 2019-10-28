class TrainingArguments(object):
    """
    Training arguments
    """
    def __init__(self, user_dim, item_dim, batch_size, num_epochs):
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs


class ColumnNames(object):
    """
    Column names
    """
    def __init__(self, user_col, item_col, rating_col):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
