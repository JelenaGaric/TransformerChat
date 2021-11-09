from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import LayerNormalization, Dropout, Dense
from tensorflow.python.keras.models import Model

from transformer import MultiHeadAttention

# Decoder layer consists of 3 sublayers:
# Masked multi-head attention with look-ahead and padding mask
# Masked multi-head attention with padding mask and 2 dense layers followed by dropout
# Value and key  of sublayer 2 receive the encoder output as inputs
# and its query receives the output from the masked multi-head attention (sublayer 1)

def decoder_layer(units, dim, heads_num, dropout, name="decoder_layer"):
    input = Input(shape=(None, dim), name="input")
    encoder_output = Input(shape=(None, dim), name="encoder_output")
    look_ahead = Input(shape=(1,None,None), name="look_ahead")
    padding_mask = Input(shape=(1,1,None), name="padding_mask")

    # mask to “zeroes out” parts of the scores Tensor that correspond to future words that aren't supposed to be seen
    # This masking, combined with fact that the output embeddings are offset by one position, ensures that the
    # predictions for position i can depend only on the known outputs at positions less than i
    masked_attention = MultiHeadAttention(dim, heads_num, name="masked_attention")(
        input={
            'query': input,
            'keys': input,
            'values': input,
            'mask': look_ahead
        })
    masked_attention = LayerNormalization(epsilon=1e-6)(masked_attention+input)
    multi_head_attention = MultiHeadAttention(dim, heads_num, name="multi_head_attention")(
        input={
            'query': masked_attention,
            'keys': encoder_output,
            'values': encoder_output,
            'mask': padding_mask
        })
    multi_head_attention = Dropout(rate=dropout)(multi_head_attention)
    multi_head_attention = LayerNormalization(epsilon=1e-6)(multi_head_attention+masked_attention)

    output = Dense(units=units, activation='relu')(multi_head_attention)
    output = Dense(units=dim)(output)
    output = Dropout(rate=dropout)(output)
    output = LayerNormalization(epsilon=1e-6)(output + multi_head_attention)

    return Model(inputs=[input, encoder_output, look_ahead, padding_mask],
                 outputs=output, name=name)
