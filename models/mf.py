import tensorflow as tf


class MatrixFactorization(object):
    """
    This class implements matrix factorization using the tf2 api
    """

    def __init__(self, n_users, n_items, user_dim, item_dim):
        self.n_users = n_users
        self.n_items = n_items
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.input_user, self.input_item, self.input_rating, self.user_embeddings, self.item_embeddings = self.fit(
        )

    @staticmethod
    def inputs_init():
        """
        Initialises the necessary inputs
        :return:
        """
        input_user = tf.keras.Input((1, ))
        input_item = tf.keras.Input((1, ))
        input_rating = tf.keras.Input((1, ))
        return input_user, input_item, input_rating

    def embeddings_layers_init(self):
        """
        Initialises the embeddings layers
        :return:
        """

        user_embeddings = tf.keras.layers.Embedding(
            self.n_users, self.user_dim, input_length=1)

        item_embeddings = tf.keras.layers.Embedding(
            self.n_items, self.item_dim, input_length=1)

        return user_embeddings, item_embeddings

    def fit(self):
        """
        Initialises inputs and weights
        :return:
        """
        input_user, input_item, input_rating = self.inputs_init()
        user_embeddings, item_embeddings = self.embeddings_layers_init()

        return input_user, input_item, input_rating, user_embeddings, item_embeddings

    def predict(self):
        """
        Generate predictions by defining the architecture
        :return:
        """
        input_item_vector = self.item_embeddings(self.input_item)
        input_user_vector = self.user_embeddings(self.input_user)
        input_item_vector_reshaped = tf.keras.layers.Reshape(
            (self.item_dim, 1))(input_item_vector)
        input_user_vector_reshaped = tf.keras.layers.Reshape(
            (self.user_dim, 1))(input_user_vector)

        dot_product = tf.keras.layers.dot(
            [input_item_vector_reshaped, input_user_vector_reshaped], axes=2)
        predicted_rating = tf.keras.layers.Dense(
            1, activation='linear')(dot_product)
        return predicted_rating

    def model(self):
        """
        Define the keras model
        :return:
        """
        model = tf.keras.Model(
            inputs=[self.input_user, self.input_item], outputs=self.predict())
        return model
