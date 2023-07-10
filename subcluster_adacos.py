import math
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K


class SCAdaCos(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, n_subclusters=1, trainable=False, regularizer=None, **kwargs):
        super(SCAdaCos, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.n_subclusters = n_subclusters
        self.s_init = math.sqrt(2) * math.log(n_classes*n_subclusters - 1)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.trainable = trainable

    def build(self, input_shape):
        super(SCAdaCos, self).build(input_shape[0])
        self.W = self.add_weight(name='W_AdaCos' + str(self.n_classes) + '_' + str(self.n_subclusters),
                                 shape=(input_shape[0][-1], self.n_classes*self.n_subclusters),
                                 initializer='glorot_uniform',
                                 trainable=self.trainable,
                                 regularizer=self.regularizer)
        self.s = self.add_weight(name='s' + str(self.n_classes) + '_' + str(self.n_subclusters),
                                 shape=(),
                                  initializer=tf.keras.initializers.Constant(self.s_init),
                                  trainable=False,
                                  aggregation=tf.VariableAggregation.MEAN)

    def call(self, inputs, training=None):
        x, y1, y2 = inputs
        y1_orig = y1
        y1 = tf.repeat(y1, repeats=self.n_subclusters, axis=-1)
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W  # same as cos theta
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))

        if training:
            max_s_logits = tf.reduce_max(self.s * logits)
            B_avg = tf.exp(self.s*logits-max_s_logits)
            #B_avg = tf.where(y1 < 1, tf.exp(self.s * logits-max_s_logits), tf.zeros_like(logits)-max_s_logits)
            B_avg = tf.reduce_mean(tf.reduce_sum(B_avg, axis=1))
            theta_class = tf.reduce_sum(y1 * theta, axis=1) * tf.math.count_nonzero(y1_orig, axis=1, dtype=tf.dtypes.float32)  # take mix-upped angle of mix-upped classes
            theta_med = tfp.stats.percentile(theta_class, q=50)  # computes median
            self.s.assign(
                (max_s_logits + tf.math.log(B_avg)) /
                tf.math.cos(tf.minimum(math.pi / 4, theta_med)) + K.epsilon())
        logits *= self.s
        out = tf.keras.activations.softmax(logits)
        out = tf.reshape(out, (-1, self.n_classes, self.n_subclusters))
        out = tf.math.reduce_sum(out, axis=2)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

    def get_config(self):
        config = {
            'n_classes': self.n_classes,
            'regularizer': self.regularizer,
            'n_subclusters': self.n_subclusters,
            'trainable': self.trainable
        }
        base_config = super(SCAdaCos, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))