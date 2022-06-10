import tensorflow as tf
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Dropout, Dense
from keras.layers import LayerNormalization, Embedding

from multihead_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding


# encoder layer consists of 2 sublayers:
# multi-head attention and padding mask
# 2 dense layers followed by dropout

# multi-head attention:
# does multiplication of query and keys weights
# and after normalisation of it multiplies that with corresponding value weight

def encoder_layer(units, model_dim, heads_num, dropout_rate, name="encoder_layer"):
    # inputs
    input = Input(shape=(None, model_dim), name="input")
    padding_mask = Input(shape=(1, 1, None), name="padding_mask")

    # attention
    multi_head_attention = MultiHeadAttention(model_dim, heads_num, name="multi_head_attention")({
        'query': input,
        'keys': input,
        'values': input,
        'mask': padding_mask,
    })
    multi_head_attention = Dropout(rate=dropout_rate)(multi_head_attention)
    multi_head_attention = LayerNormalization(epsilon=1e-6)(input + multi_head_attention)

    # output
    output = Dense(units=units, activation='relu')(multi_head_attention)
    output = Dense(units=model_dim)(output)
    output = Dropout(rate=dropout_rate)(output)
    output = LayerNormalization(epsilon=1e-6)(multi_head_attention + output)

    return Model(inputs=[input, padding_mask], outputs=output, name=name)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)


# encoder consists of
# input embedding
# positional encoding
# N of encoder layers

def encoder(vocab_size, layers_num, units, model_dim, heads_num, dropout_rate, name="encoder"):
    input = Input(shape=(None,), name="input")
    padding_mask = Input(shape=(1, 1, None), name="padding_mask")
    embeddings = Embedding(vocab_size, model_dim)(input)
    # scaling
    embeddings *= tf.math.sqrt(tf.cast(model_dim, tf.float32))
    embeddings = PositionalEncoding(vocab_size, model_dim)(embeddings)

    output = Dropout(rate=dropout_rate)(embeddings)
    for i in range(layers_num):
        output = encoder_layer(
            units=units,
            model_dim=model_dim,
            heads_num=heads_num,
            dropout=dropout_rate,
            name=f"encoder_layer_{i}"
            )([output, padding_mask])

    return Model(inputs=[input, padding_mask], outputs=output, name=name)
