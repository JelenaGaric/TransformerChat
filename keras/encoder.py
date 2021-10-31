import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dropout, LayerNormalization, Dense
from transformer import MultiHeadAttention

# encoder layer consists of: multi-head attention and padding mask
# and 2 dense layers followed by dropout

# multi-head attention:
# does multiplication of query and keys weights
# and after normalisation of it
# multiplies that with corresponding value weight

def encoder_layer(units, dim, heads_num, dropout, name="encoder_layer"):
    # inputs
    input = tf.keras.Input(shape=(None, dim), name="input")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # attention
    multi_head_attention = MultiHeadAttention(dim, heads_num, name="multi_head_attention")({
        'query': input,
        'keys': input,
        'values': input,
        'mask': padding_mask,
    })
    multi_head_attention = Dropout(rate=dropout)(multi_head_attention)
    multi_head_attention = LayerNormalization(epsilon=1e-6)(input + multi_head_attention)

    # output
    output = Dense(units=units, activation='relu')(multi_head_attention)
    output = Dense(units=dim)(output)
    output = Dropout(rate=dropout)(output)
    output = LayerNormalization(epsilon=1e-6)(multi_head_attention + output)

    return Model(inputs=[input, padding_mask], outputs=output, name=name)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)
