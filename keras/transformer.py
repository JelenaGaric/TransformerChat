import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dropout, Dense
from keras.layers import LayerNormalization, Embedding, Lambda
from tensorflow.python.keras.models import Model

from encoder import encoder
from decoder import decoder

# transformer consists of:
# encoder
# decoder
# final linear layer

# look_ahead_mask is used to mask out future tokens in a sequence
# their length differs so Lambda layers are used

class Transformer():
    def __int__(
            self,
            vocab_size,
            layers_num,
            units,
            model_dim,
            heads_num,
            dropout_rate,
            padding_mask,
            look_ahead_mask,
            name="transformer"
    ):
        self.vocab_size = vocab_size
        self.layers_num = layers_num
        self.units = units
        self.model_dim = model_dim
        self.heads_num = heads_num
        self.dropout_rate = dropout_rate
        self.name = name
        self.padding_mask = padding_mask
        self.look_ahead_mask = look_ahead_mask

    def create_model(self):
        input = Input(shape=(None,), name="input")
        decoder_input = Input(shape=(None,), name="decoder_input")
        encoder_padding_mask_layer = Lambda(self.padding_mask, output_shape=(1, 1, None),
                                            name="encoder_padding_mask")(input)
        # mask the future tokens for decoder inputs at the 1st attention block
        look_ahead_mask_layer = Lambda(self.look_ahead_mask, output_shape=(1, None, None),
                                       name="look_ahead_mask")(decoder_input)
        # mask encoder output for 2nd attention block
        decoder_padding_mask = Lambda(self.padding_mask, output_shape=(1, 1, None),
                                      name="decoder_padding_mask")(input)
        encoder_output = encoder(
            vocab_size=self.vocab_size,
            layers_num=self.layers_num,
            units=self.units,
            model_dim=self.model_dim,
            heads_num=self.heads_num,
            dropout_rate=self.dropout_rate
        )(inputs=[input, encoder_padding_mask_layer])

        decoder_output = decoder(
            vocab_size=self.vocab_size,
            layers_num=self.layers_num,
            units=self.units,
            model_dim=self.model_dim,
            heads_num=self.heads_num,
            dropout_rate=self.dropout_rate
        )(inputs=[decoder_input, encoder_output, look_ahead_mask_layer, decoder_padding_mask])

        output = Dense(units=self.vocab_size, name="output")(decoder_output)

        return Model(inputs=[input, decoder_input], outputs=output, name=self.name)