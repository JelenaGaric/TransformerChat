import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

from transformer import Transformer

LAYERS_NUM = 2
MODEL_DIM = 256
HEADS_NUM = 8
UNITS = 512
DROPOUT_RATE = 0.1

model = Transformer(
    vocab_size=VOCAB_SIZE,
    layers_num=LAYERS_NUM,
    units=UNITS,
    model_dim=MODEL_DIM,
    heads_num=HEADS_NUM,
    dropout_rate=DROPOUT_RATE
)

def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    loss = SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)

def accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    accuracy = SparseCategoricalAccuracy()(y_true, y_pred)
    return accuracy