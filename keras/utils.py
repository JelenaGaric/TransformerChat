import tensorflow as tf

def scaled_dot_product(query, keys, values, mask):
    # dot product of query vector and transposed keys vector
    multiply_qk = tf.matmul(query, keys, transpose_b=True)

    # don’t want the dot product to be huge if we choose a huge vector length,
    # so we divide by the square root of the vector length
    d_k = tf.cast(tf.shape(keys)[-1], tf.float32)
    scales = multiply_qk / tf.matmul.sqrt(d_k)

    # zero on the padding tokens with the mask
    if mask is not None:
        scales += (mask * -1e9)

    # normalise by softmax
    attention_weights = tf.nn.softmax(scales, axis=-1)

    # the output is multiplication of weights and values
    return tf.matmul(attention_weights, values)