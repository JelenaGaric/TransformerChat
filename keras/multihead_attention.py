import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.layers.base import Layer


# Query, keys, and values are split into multiple heads because it allows the model to attend
# to information at different positions from different representational spaces.
# Attention output for each head is then concatenated and put through a final dense layer.

# MultiHeadAttention consists of:
# linear layers split into heads
# SCALED dot product attention
# concatenating heads
# final linear layer

# inputs are a dictionary of query, keys and values (because of functional api requires inputs as one argument)

class MultiHeadAttention(Layer):
    def __init__(self, model_dim, heads_num, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.heads_num = heads_num
        self.model_dim = model_dim

        assert model_dim % heads_num == 0
        self.depth = self.model_dim // self.heads_num

        self.query = Dense(units=model_dim)
        self.keys = Dense(units=model_dim)
        self.values = Dense(units=model_dim)
        self.final_dense = Dense(units=model_dim)

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
        concatenated_attention = tf.reshape(attention, (batch_size, -1, self.model_dim))

        output = self.final_dense(concatenated_attention)

        return output


# Attention(Q,K,V) = softmax_k(QK_t/root(d_k))V
def scaled_dot_product(query, key, value, mask):
    QK_t = tf.matmul(query, key, transpose_b=True)
    d_k = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = QK_t/tf.math.sqrt(d_k)
    # for paddings
    if mask:
        logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    return tf.matmul(attention_weights, value)