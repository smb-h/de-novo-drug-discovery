import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, model_from_json

from .model import BaseModel
from .utils import posterior_mean_field, prior_mean_field


class CustomizedLayer_Polarizer(keras.layers.Layer):
    def __init__(self, units=32):
        self.units = units
        super(CustomizedLayer_Polarizer, self).__init__()

    def get_config(self):
        config = super().get_config().copy()
        config.update({"units": self.units})
        return config

    def call(self, inputs):
        G_thesis = inputs
        G_antithesis = 1 - inputs

        return [G_thesis, G_antithesis]


class CustomizedLayer_Attention(keras.layers.Layer):
    def __init__(self, units=32):
        self.units = units
        super(CustomizedLayer_Attention, self).__init__()

    def get_config(self):
        config = super().get_config().copy()
        config.update({"units": self.units})
        return config

    def call(self, inputs):
        # G_LSTM= inputs[:,:60]
        # G_Attention= inputs[:,60:]
        # res= tf.math.add(G_LSTM, G_Attention)
        elem_prod = inputs[:, :, :60] + inputs[:, :, 60:]
        # res = k.sum(elem_prod, axis=-1, keepdims=True)
        return elem_prod


class CustomizedLayer_fusion(keras.layers.Layer):
    def __init__(self, units=32):
        self.units = units
        super(CustomizedLayer_fusion, self).__init__()

    def get_config(self):
        config = super().get_config().copy()
        config.update({"units": self.units})
        return config

    def call(self, inputs):
        elem_prod = (inputs[:, :, :35] + inputs[:, :, 35:])/2
        return elem_prod


# Model
class Model(BaseModel):
    # init
    def __init__(self, config, session="train", logger=None) -> None:
        super().__init__(config, session, logger)
        self.name = "BPMoe_S"

    # build
    def build(self):
        hidden_units = [2, 2, 2]
        model = Sequential()

        InData_Ex1 = layers.Input(
            shape=([self.config.get("input_shape")[1], self.config.get("input_shape")[2]]),
            name="Input_Ex1",
        )
        InData_Ex2 = layers.Input(
            shape=([self.config.get("input_shape")[1], self.config.get("input_shape")[2]]),
            name="polarizer",
        )
        InData_Ex3 = layers.Input(
            shape=([self.config.get("input_shape")[1], self.config.get("input_shape")[2]]),
            name="Input_EX3",
        )

        EX_lstm1 = layers.LSTM(60, return_sequences=True)(InData_Ex1)
        EX_lstm2 = layers.LSTM(60, return_sequences=True)(InData_Ex2)
        GateIn = layers.Dense(units=60, activation="sigmoid")(InData_Ex3)
        Gate_pp = layers.Dense(units=60, activation="sigmoid")(GateIn)
        Gate_pp = layers.Dense(units=60, activation="sigmoid")(Gate_pp)
        CFPG = CustomizedLayer_Polarizer(units=60)(Gate_pp)
        # GatesODD = layers.Dense(units=60, activation='sigmoid')(GatesIn)
        MultiplictionEven_In = layers.Concatenate(axis=-1)([EX_lstm1, CFPG[0]])
        MultiplictionEven_In = CustomizedLayer_Attention()(MultiplictionEven_In)
        EX_lstm1 = layers.LSTM(60, return_sequences=True)(MultiplictionEven_In)
        MultiplictionEven = layers.Dense(units=60, activation="sigmoid")(EX_lstm1)
        MultiplictionEven = layers.Dense(units=40, activation="sigmoid")(MultiplictionEven)
        MultiplictionEven = layers.Dense(units=35, activation="sigmoid")(MultiplictionEven)
        MultiplictionODD_In = layers.Concatenate(axis=-1)([EX_lstm2, CFPG[1]])
        MultiplictionODD_In = CustomizedLayer_Attention()(MultiplictionODD_In)
        EX_lstm2 = layers.LSTM(60, return_sequences=True)(MultiplictionODD_In)
        MultiplictionODD = layers.Dense(units=60, activation="sigmoid")(EX_lstm2)
        MultiplictionODD = layers.Dense(units=40, activation="sigmoid")(MultiplictionODD)
        MultiplictionODD = layers.Dense(units=35, activation="sigmoid")(MultiplictionODD)
        # features = layers.Concatenate([InData_Ex1, InData_Ex2, InData_Ex3, InData_Ex4])
        InData = layers.Concatenate(axis=-1)([MultiplictionEven, MultiplictionODD])
        InData = CustomizedLayer_fusion()(InData)
        InData = layers.BatchNormalization()(InData)
        features = InData
        for units in hidden_units:
            features = tfp.layers.DenseVariational(
                units=units,
                make_prior_fn=prior_mean_field,
                make_posterior_fn=posterior_mean_field,
                kl_weight=1 / self.config.get("data_len"),
                activation="relu",
            )(features)
        features = layers.Dense(
            units=self.config.get("input_shape")[2], activation="sigmoid", name="features"
        )(features)
        model = keras.Model(inputs=[InData_Ex1, InData_Ex2, InData_Ex3], outputs=features)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model
