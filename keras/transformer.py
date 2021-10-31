import tensorflow as tf
from tensorflow.python.framework.test_ops import attr
from tensorflow.python.keras.layers import Dense
from tensorflow.python.layers.base import Layer


# The positional encoding vector is added to the embedding vector.
# Words will be closer to each other based on the similarity of their
# meaning and their position in the sentence, in the d-dimensional space.

class PositionalEncoding(Layer):
    def __init__(self, position, dim):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.get_positional_encoding(position, dim)

    def get_position_angle(self, position, i, dim):
        angle = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(dim, tf.float32))
        return position * angle

    def get_positional_encoding(self, position, dim):
        angles = self.get_position_angle(position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                                         i=tf.range(dim, dtype=tf.float32)[tf.newaxis, :],
                                         d_model=dim)
        # calculate sin function of every even i
        sin = tf.math.sin(angles[:, 0::2])
        # calculate sin function of every even i
        cos = tf.math.cos(angles[:, 1::2])

        positional_encoding = tf.concat([sin, cos], axis=1)
        positional_encoding = positional_encoding[tf.newaxis, ...]
        return tf.cast(positional_encoding, tf.float32)

    def call(self, input):
        return input + self.positional_encoding[:, :tf.shape(input)[1], :]


# Query, keys, and values are split into multiple heads because it allows the model to attend
# to information at different positions from different representational spaces.
# Attention output for each head is then concatenated and put through a final dense layer.

# MultiHeadAttention consists of linear layers split into heads,
# SCALED dot product attention, concatenating heads, and
# final linear layer

# inputs are a dictionary of query, keys and values (because of functional api requires inputs as one argument)

class MultiHeadAttention(Layer):
    def __init__(self, dim, heads_num, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.heads_num = heads_num
        self.dim = dim

        assert dim % heads_num == 0
        self.depth = self.dim // self.heads_num

        self.query = Dense(units=dim)
        self.keys = Dense(units=dim)
        self.values = Dense(units=dim)
        self.final_dense = Dense(units=dim)

    def split_to_heads(self, input, batch_size):
        input = tf.reshape(input, shape=(batch_size, -1, self.heads_num, self.depth))
        return tf.transpose(input, perm=[0, 2, 1, 3])

    def call(self, input):
        query, keys, values, mask = input['query'], input['keys'], input['values'], input['mask']
        batch_size = tf.shape(query)[0]

        # linear layers (because of functional api requires inputs as one argument)
        query = self.query(query)
        keys = self.keys(keys)
        values = self.values(values)

        # split into heads
        query = self.split_to_heads(query, batch_size)
        keys = self.split_to_heads(keys, batch_size)
        values = self.split_to_heads(values, batch_size)

        attention = scaled_dot_product(query, keys, values, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concatenated_attention = tf.reshape(attention, (batch_size, -1, self.dim))

        output = self.final_dense(concatenated_attention)

        return output