from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_probability as tfp

class MixupLayer(layers.Layer):
    def __init__(self, prob, alpha=1, **kwargs):
        super(MixupLayer, self).__init__(**kwargs)
        self.prob = prob
        self.alpha = alpha

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, training=None):
        # get mixup weights
        if self.alpha == 1:
            #dist = tfp.distributions.Beta(0.5, 0.5)
            #l = dist.sample([tf.shape(inputs[0])[0]])
            l = tf.random.uniform(shape=[tf.shape(inputs[0])[0]])
        X_l = tf.reshape(l, [-1]+[1]*(len(inputs[0].shape)-1))
        y_l = tf.reshape(l, [-1]+[1]*(len(inputs[1].shape)-1))

        # mixup data
        X1 = inputs[0]
        X2 = tf.reverse(inputs[0], axis=[0])
        X = X1 * X_l + X2 * (1 - X_l)

        # mixup labels
        y1 = inputs[1]
        y2 = tf.reverse(inputs[1], axis=[0])
        y = y1 * y_l + y2 * (1 - y_l)
        #y = tf.math.maximum(y1 * y_l, y2 * (1 - y_l))

        # apply mixup or not
        dec = tf.dtypes.cast(tf.random.uniform(shape=[tf.shape(inputs[0])[0]]) < self.prob, tf.dtypes.float32)
        dec1 = tf.reshape(dec, [-1] + [1] * (len(inputs[0].shape) - 1))
        out1 = dec1 * X + (1 - dec1) * inputs[0]
        dec2 = tf.reshape(dec, [-1] + [1] * (len(inputs[1].shape) - 1))
        out2 = dec2 * y + (1 - dec2) * inputs[1]
        outputs = [out1, out2]

        # pick output corresponding to training phase
        return K.in_train_phase(outputs, inputs, training=training)

    def get_config(self):
        config = {
            'prob': self.prob,
            'alpha': self.alpha
        }
        base_config = super(MixupLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

