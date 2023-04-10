import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

np.set_printoptions(precision=3, suppress=True)
tfd = tfp.distributions


# prior Function
def prior_mean_field(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return lambda t: tfd.Independent(
        tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=tf.ones(n)), reinterpreted_batch_ndims=1
    )


# Posterior Function
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
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
