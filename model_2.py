import tensorflow as tf
import numpy as np
from tensorflow import (
    shape,
    eye,
    sqrt,
    matmul,
    reduce_sum,
    exp,
    constant,
    cast,
    transpose,
    tensordot,
    reduce_max,
    reduce_mean,
    random,
    linalg,
    math,
    float32,
    float32,
)
from keras import Model, layers, regularizers, initializers, activations
from tensorflow_probability import distributions

from absl import flags
import tensorflow.keras.backend as K
FLAGS = flags.FLAGS



class MODEL(tf.keras.Model):
    def __init__(self, is_training, n_features, n_classes):
        super().__init__()

        # Set random seed for reproducibility
        random.set_seed(19950420)

        self.r_dim = n_classes
        self.n_feature = n_features
        self.n_classes = n_classes
        self.is_training = is_training

        # Define the input layers (as part of the model's architecture)
        self.input_X = tf.keras.Input(shape=(None, n_features), dtype=tf.float32, name='input_X')
        self.input_Y = tf.keras.Input(shape=(None, n_classes), dtype=tf.float32, name='input_Y')

        
        # Define layers using Keras instead of slim
        
        
        self.fc1 = layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(FLAGS.weight_decay),
        )

        self.fc2 = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(FLAGS.weight_decay),
        )

        self.fc3 = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(FLAGS.weight_decay),
        )
        
        self.normalized_mu_layer = layers.Dense(
            self.r_dim,
            activation=None,
            kernel_regularizer=regularizers.l2(FLAGS.weight_decay),
        )

        # Initialize the square root of the residual covariance matrix
        initializer = initializers.RandomUniform(
            minval=-np.sqrt(6.0 / (self.r_dim + FLAGS.z_dim)),
            maxval=np.sqrt(6.0 / (self.r_dim + FLAGS.z_dim)),
        )

        self.r_sqrt_sigma = self.add_weight(
            shape=(self.r_dim, FLAGS.z_dim),
            initializer=initializer,
            dtype=float32,
            trainable=True,
        )

        # Constants
        self.eps1 = constant(1e-6, dtype=float32)
        self.eps2 = constant(1e-6 * 2.0 ** (-100), dtype=float32)
        self.eps3 = 1e-30

        # Define distribution for sampling
        self.norm = random.normal

    def build(self, input_shape):
        # This method is called when the model is built
        # We can use it to validate input shapes
        if input_shape[0][1] != self.n_feature:
            raise ValueError(
                f"Expected input_X with {self.n_feature} features, got {input_shape[0][1]}"
            )
        if input_shape[1][1] != self.r_dim:
            raise ValueError(
                f"Expected input_Y with {self.r_dim} classes, got {input_shape[1][1]}"
            )
        super(Model, self).build(input_shape)

    def call(self, inputs, is_training=None):
        input_X, input_Y = inputs

        # Convert inputs to float32 if they're not already
        input_X = cast(input_X, float32)
        input_Y = cast(input_Y, float32)

        # x = self.input_layer(input_X)

        # print(x)

        # Run through the feature extractor layers
        fc_1 = self.fc1(input_X)
        fc_2 = self.fc2(fc_1)
        feature = self.fc3(fc_2)

        # Compute sigma (covariance matrix)
        sigma = matmul(self.r_sqrt_sigma, transpose(self.r_sqrt_sigma))
        covariance = sigma + eye(self.r_dim, dtype=float32)
        cov_diag = linalg.diag_part(covariance)

        # Compute normalized_mu and r_mu
        normalized_mu = self.normalized_mu_layer(feature)
        r_mu = normalized_mu * sqrt(cov_diag)

        # Determine number of samples
        n_sample = (
            FLAGS.n_train_sample if is_training else FLAGS.n_test_sample
        )

        # Generate noise samples
        noise = random.normal(
            shape=[n_sample, shape(r_mu)[0], FLAGS.z_dim], dtype=float32
        )

        # Compute samples
        B = transpose(self.r_sqrt_sigma)
        sample_r = tensordot(noise, B, axes=1) + r_mu  # tensor: n_sample*n_batch*r_dim

        # Compute probability
        norm_dist = distributions.Normal(0.0, 1.0)  # TF 2.x: Use random.normal
        E = (
            cast(norm_dist.cdf(cast(sample_r, float32)), float32) * (1 - self.eps1)
            + self.eps1 * 0.5
        )

        # Compute sample negative log likelihood
        sample_nll = -((math.log(E) * input_Y + math.log(1 - E) * (1 - input_Y)))
        logprob = -reduce_sum(sample_nll, axis=2)

        # Compute loss avoiding float overflow
        maxlogprob = reduce_max(logprob, axis=0)
        Eprob = reduce_mean(exp(logprob - maxlogprob), axis=0)
        nll_loss = reduce_mean(-math.log(Eprob) - maxlogprob)

        # Compute individual probabilities
        norm_dist = layers.Lambda(
            lambda x: cast(activations.sigmoid(x * 1.70169), float32)
        )  # Approximation of normal CDF

        indiv_prob = norm_dist(normalized_mu) * (1 - self.eps1) + 0.5 * self.eps1

        # Compute cross entropy
        cross_entropy = math.log(indiv_prob) * input_Y + math.log(1 - indiv_prob) * (
            1 - input_Y
        )
        marginal_loss = -reduce_mean(reduce_sum(cross_entropy, axis=1))

        # Compute regularization loss
        l2_loss = reduce_sum(self.losses)

        # Compute total loss
        total_loss = l2_loss + nll_loss

        # Store intermediate values for later use
        self.nll_loss = nll_loss
        self.marginal_loss = marginal_loss
        self.l2_loss = l2_loss
        self.total_loss = total_loss
        self.indiv_prob = indiv_prob
        self.E = E
        self.sample_r = sample_r
        self.r_mu = r_mu
        self.covariance = covariance

        # Return predicted probabilities
        return indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss

       