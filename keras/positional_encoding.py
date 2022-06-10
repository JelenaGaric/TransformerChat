import tensorflow as tf
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

