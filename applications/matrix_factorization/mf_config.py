from config.local import TrainingArguments, ColumnNames

config = {
        'training_args': TrainingArguments(
            user_dim=10,
            item_dim=5,
            batch_size=16,
            num_epochs=5),
        'col_names': ColumnNames(
            user_col='user',
            item_col='item',
            rating_col='rating')}
