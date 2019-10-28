class ColumnIndexer(object):
    """
    This class in used in order to index
    columns like user id, item id in a pandas dataframe

    """

    def __init__(self, df, col_names):
        self.df = df
        self.col_names = col_names
        self.distinct_items = self.get_distinct_items()
        self.indexers, self.reverse_indexers = self.generate_indexers()

    def get_distinct_items(self):
        """
        Get all the distinct item in the column

        :return:
        """
        distinct_items = {
            col_name: set(self.df[col_name].values)
            for col_name in self.col_names
        }
        return distinct_items

    def generate_indexers(self):
        """
        Builds the indexers and the reverse indexers for all
        the columns defined when instantiating the class

        :return:
        """
        indexers = {
            col: {k: v
                  for v, k in enumerate(distinct_item)}
            for col, distinct_item in self.distinct_items.items()
        }
        reverse_indexers = {
            col: {v: k
                  for k, v in indexer.items()}
            for col, indexer in indexers.items()
        }
        return indexers, reverse_indexers

    def transform(self, dataset):
        """
        Transforms the original dataset
        by adding indexed versions of the columns

        :param dataset: Original pandas dataframe
        :return:
        """
        for col in self.col_names:
            dataset[col + '_indexed'] = dataset[col]\
                .map(self.indexers[col])
        return dataset
