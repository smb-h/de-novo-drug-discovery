import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Dropout, Input
from tensorflow.keras.models import Sequential

np.set_printoptions(precision=3, suppress=True)
tfd = tfp.distributions


def prior_mean_field(kernel_size, bias_size, dtype=None):  # prior Function
    n = kernel_size + bias_size
    return lambda t: tfd.Independent(
        tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=tf.ones(n)), reinterpreted_batch_ndims=1
    )


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):  # Posterior Function
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :n], scale=1e-5 + 0.01 * tf.nn.softplus(c + t[..., n:])),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


class CustomizedLayer_polarizer(keras.layers.Layer):
    def __init__(self, units=32):
        super(CustomizedLayer_polarizer, self).__init__()

    def call(self, inputs):
        G_thesis = inputs
        G_antithesis = 1 - inputs

        return [G_thesis, G_antithesis]


class CustomizedLayer_attention(keras.layers.Layer):
    def __init__(self, units=32):
        super(CustomizedLayer_attention, self).__init__()

    def call(self, inputs):
        # G_LSTM= inputs[:,:60]
        # G_Attention= inputs[:,60:]
        # res= tf.math.add(G_LSTM, G_Attention)
        elem_prod = inputs[:, :, :60] + inputs[:, :, 60:]
        # res = k.sum(elem_prod, axis=-1, keepdims=True)
        return elem_prod


def Model():  # Prepair Model
    input_len = 35  # should be "X.shape[0]" value
    hidden_units = [70, 70, 70]
    batch_size = 50
    counter_L = 0
    look_back = 1
    model = Sequential()
    InData_Ex1 = Input(shape=([1, 70]), name="Input_Ex1")
    InData_Ex2 = Input(shape=([1, 70]), name="polarizer")
    InData_Ex3 = Input(shape=([1, 70]), name="Input_EX3")
    EX_lstm1 = layers.LSTM(60, return_sequences=True)(InData_Ex1)
    EX_lstm2 = layers.LSTM(60, return_sequences=True)(InData_Ex2)
    GateIn = layers.Dense(units=60, activation="sigmoid")(InData_Ex3)
    Gate_pp = layers.Dense(units=60, activation="sigmoid")(GateIn)
    Gate_pp = layers.Dense(units=60, activation="sigmoid")(Gate_pp)
    CFPG = CustomizedLayer_polarizer(units=60)(Gate_pp)
    # GatesODD = layers.Dense(units=60, activation='sigmoid')(GatesIn)
    MultiplictionEven_In = layers.Concatenate(axis=-1)([EX_lstm1, CFPG[0]])
    MultiplictionEven_In = CustomizedLayer_attention()(MultiplictionEven_In)
    EX_lstm1 = layers.LSTM(60, return_sequences=True)(MultiplictionEven_In)
    MultiplictionEven = layers.Dense(units=35, activation="sigmoid")(EX_lstm1)
    MultiplictionODD_In = layers.Concatenate(axis=-1)([EX_lstm2, CFPG[1]])
    MultiplictionODD_In = CustomizedLayer_attention()(MultiplictionODD_In)
    EX_lstm2 = layers.LSTM(60, return_sequences=True)(MultiplictionODD_In)
    MultiplictionODD = layers.Dense(units=35, activation="sigmoid")(EX_lstm2)
    # features = layers.Concatenate([InData_Ex1, InData_Ex2, InData_Ex3, InData_Ex4])
    InData = layers.Concatenate(axis=-1)([MultiplictionEven, MultiplictionODD])
    InData = BatchNormalization()(InData)
    features = InData
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior_mean_field,
            make_posterior_fn=posterior_mean_field,
            kl_weight=1 / input_len,
            activation="relu",
        )(features)
    features = layers.Dense(units=70, activation="sigmoid")(features)
    model = keras.Model(inputs=[InData_Ex1, InData_Ex2, InData_Ex3], outputs=features)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
