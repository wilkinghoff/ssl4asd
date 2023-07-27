from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_probability as tfp

class StatExLayer(layers.Layer):
    def __init__(self, prob, **kwargs):
        super(StatExLayer, self).__init__(**kwargs)
        self.prob = prob

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, training=None):
        # mixup data
        X1 = inputs[0]
        X1_rev = tf.reverse(inputs[0], axis=[0])

        # mixup labels
        y = tf.concat([inputs[1], tf.zeros_like(inputs[1]), tf.zeros_like(inputs[1])], axis=1)
        y_ex = tf.concat([tf.zeros_like(inputs[1]), 0.5 * inputs[1], 0.5 * tf.reverse(inputs[1], axis=[0])], axis=1)  # best?

        # statistics exchange data
        X_tex = (X1 - tf.math.reduce_mean(X1, axis=2, keepdims=True)) / (tf.math.reduce_std(X1, axis=2, keepdims=True) + 1e-16) * tf.math.reduce_std(X1_rev, axis=2, keepdims=True) + tf.math.reduce_mean(X1_rev, axis=2, keepdims=True)
        X_fex = (X1 - tf.math.reduce_mean(X1, axis=1, keepdims=True)) / (tf.math.reduce_std(X1, axis=1, keepdims=True) + 1e-16) * tf.math.reduce_std(X1_rev, axis=1, keepdims=True) + tf.math.reduce_mean(X1_rev, axis=1, keepdims=True)

        # randomly decide on which statistics exchange axis to use
        dec = tf.dtypes.cast(tf.random.uniform(shape=[tf.shape(inputs[0])[0]]) < 1, tf.dtypes.float32)
        dec1 = tf.reshape(dec, [-1] + [1] * (len(inputs[0].shape) - 1))
        X_ex = dec1 * X_fex + (1 - dec1) * X_tex
        #dec2 = tf.reshape(dec, [-1] + [1] * (len(y_new.shape) - 1))
        #y_ex = dec2 * y_fex + (1 - dec2) * y_tex

        # apply mixup or not
        dec = tf.dtypes.cast(tf.random.uniform(shape=[tf.shape(inputs[0])[0]]) < self.prob, tf.dtypes.float32)
        dec1 = tf.reshape(dec, [-1] + [1] * (len(inputs[0].shape) - 1))
        out1 = dec1 * X1 + (1 - dec1) * X_ex
        dec3 = tf.reshape(dec, [-1] + [1] * (len(y.shape) - 1))
        out3 = dec3 * y + (1 - dec3) * y_ex
        #dec4 = tf.reshape(dec, [-1] + [1] * (len(y.shape) - 1))
        #out4 = dec4 * y + (1 - dec4) * y_rev2
        outputs = [out1, out3]

        # pick output corresponding to training phase
        return K.in_train_phase(outputs, [inputs[0], y], training=training)

    def get_config(self):
        config = {
            'prob': self.prob,
        }
        base_config = super(StatExLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

