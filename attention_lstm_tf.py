"""
attention_lstm_tf.py
LSTM with custom attention layer (TensorFlow Keras).
Returns both prediction and attention weights.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, LSTM, Dense
from tensorflow.keras.models import Model

class Attention(Layer):
    """
    Simple additive attention over time dimension.
    Input shape: (batch, timesteps, features)
    Output: context vector (batch, features), attention weights (batch, timesteps, 1)
    """
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        # weight for projecting features -> scalar score
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='att_weight')
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True,
                                 name='att_bias')
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time, features)
        e = tf.tanh(tf.matmul(inputs, self.W) + self.b)  # (batch, time, 1)
        weights = tf.nn.softmax(e, axis=1)  # (batch, time, 1)
        context = tf.reduce_sum(weights * inputs, axis=1)  # (batch, features)
        return context, weights

def build_attention_lstm(input_shape, units=64):
    """
    Build an LSTM model that returns predictions and attention weights.
    Keras will train using the first output (the prediction).
    """
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(units, return_sequences=True)(inputs)
    context, attn_weights = Attention()(lstm_out)
    out = Dense(1)(context)
    model = Model(inputs=inputs, outputs=[out, attn_weights])
    model.compile(optimizer='adam', loss='mse')
    return model
