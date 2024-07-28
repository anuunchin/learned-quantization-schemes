import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal

"""
All of the below implementations set the scaling factor for b (self.scale_b) as a scalar. 
The initializers for scale_w and scale_b can be changes without issues.
Proceed with caution if you're about to change other things...
"""

class DefaultDense(tf.keras.layers.Layer):
    """
    This is a custom layer that mimics a standard dense (fully connected) layer
    with the addition of scale_w and scale_b attributes. In this default implementation, the scale_w
    and scale_b do not affect the layer's output, serving as placeholders for potential scaling logic.
    """
    def __init__(self, units, activation=None):
        super(DefaultDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
        self.scale_w = self.add_weight(shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
        self.scale_b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs): 
        output = tf.matmul(inputs, self.w) + self.b
       
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_scale_w(self):
        return None
    
    def get_scale_b(self):
        return None


class RowWiseQuantized(tf.keras.layers.Layer):
    """
    This is a custom layer that implements a dense (fully connected) layer with
    learned quantization of weights and biases without the straight through estimator.
    """    
    def __init__(self, units, activation=None):
        super(RowWiseQuantized, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        """
        Example shapes:
        self.w:         (784, 128)
        self.b:         (128, )
        self.scale_w:   (784, 1) applied row-wise
        self.scale_b:   (1, 1)
        """
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
        self.scale_w = self.add_weight(shape=(input_shape[-1], 1), initializer=RandomNormal(mean=0.0, stddev=0.05), trainable=True)
        self.scale_b = self.add_weight(shape=(1,1), initializer=RandomNormal(mean=0.0, stddev=0.05), trainable=True)

    def call(self, inputs): 
        w_quantized_nonrounded = self.w / self.scale_w

        w_quantized_rounded = tf.stop_gradient(tf.floor(w_quantized_nonrounded))
        w_quantized_scaled_back = w_quantized_rounded * self.scale_w

        b_quantized_nonrounded = self.b / self.scale_b

        b_quantized_rounded = tf.stop_gradient(tf.floor(b_quantized_nonrounded))
        b_quantized_scaled_back = b_quantized_rounded * self.scale_b

        output = tf.matmul(inputs, w_quantized_scaled_back) + b_quantized_scaled_back
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_scale_w(self):
        return self.scale_w
    
    def get_scale_b(self):
        return self.scale_b


class RowWiseQuantizedSTE(tf.keras.layers.Layer):
    """
    This is a custom layer that implements a dense (fully connected) layer with
    learned quantization of weights and biases using a form of the straight-through estimator.
    """
    def __init__(self, units, activation=None):
        super(RowWiseQuantizedSTE, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        """
        Example shapes:
        self.w:         (784, 128)
        self.b:         (128, )
        self.scale_w:   (784, 1) applied row-wise
        self.scale_b:   (1, 1)
        """
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
        self.scale_w = self.add_weight(shape=(input_shape[-1], 1), initializer=RandomNormal(mean=0.0, stddev=0.05), trainable=True)
        self.scale_b = self.add_weight(shape=(1,1), initializer=RandomNormal(mean=0.0, stddev=0.05), trainable=True)

    def call(self, inputs): 
        # Straight through estimator
        w_quantized_nonrounded = self.w / self.scale_w
        w_quantized_rounded = tf.stop_gradient(tf.floor(w_quantized_nonrounded)) + w_quantized_nonrounded - tf.stop_gradient(w_quantized_nonrounded)
        w_quantized_scaled_back = w_quantized_rounded * self.scale_w

        # Straight through estimator
        b_quantized_nonrounded = self.b / self.scale_b
        b_quantized_rounded = tf.stop_gradient(tf.floor(b_quantized_nonrounded)) + b_quantized_nonrounded - tf.stop_gradient(b_quantized_nonrounded)
        b_quantized_scaled_back = b_quantized_rounded * self.scale_b

        output = tf.matmul(inputs, w_quantized_scaled_back) + b_quantized_scaled_back

        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_scale_w(self):
        return self.scale_w
    
    def get_scale_b(self):
        return self.scale_b


class ColumnWiseQuantized(tf.keras.layers.Layer):
    """
    This is a custom layer that implements a dense (fully connected) layer with
    learned quantization of weights and biases without the straight through estimator.
    """    
    def __init__(self, units, activation=None):
        super(ColumnWiseQuantized, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        """
        Example shapes:
        self.w:         (784, 128)
        self.b:         (128, )
        self.scale_w:   (1, 128) applied column-wise
        self.scale_b:   (1, 1)
        """
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
        self.scale_w = self.add_weight(shape=(1,self.units), initializer=RandomNormal(mean=0.0, stddev=0.00000000001), trainable=True)
        self.scale_b = self.add_weight(shape=(1,1), initializer=RandomNormal(mean=0.0, stddev=0.00000000001), trainable=True)

    def call(self, inputs): 
        w_quantized_nonrounded = self.w / self.scale_w

        w_quantized_rounded = tf.stop_gradient(tf.floor(w_quantized_nonrounded))
        w_quantized_scaled_back = w_quantized_rounded * self.scale_w

        b_quantized_nonrounded = self.b / self.scale_b

        b_quantized_rounded = tf.stop_gradient(tf.floor(b_quantized_nonrounded))
        b_quantized_scaled_back = b_quantized_rounded * self.scale_b

        output = tf.matmul(inputs, w_quantized_scaled_back) + b_quantized_scaled_back
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_scale_w(self):
        return self.scale_w
    
    def get_scale_b(self):
        return self.scale_b


class ColumnWiseQuantizedSTE(tf.keras.layers.Layer):
    """
    This is a custom layer that implements a dense (fully connected) layer with
    learned quantization of weights and biases using a form of the straight-through estimator.
    """
    def __init__(self, units, activation=None):
        super(ColumnWiseQuantizedSTE, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        """
        Example shapes:
        self.w:         (784, 128)
        self.b:         (128, )
        self.scale_w:   (1, 128) applied column-wise
        self.scale_b:   (1, 1)
        """
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
        self.scale_w = self.add_weight(shape=(1,self.units), initializer=RandomNormal(mean=0.0, stddev=0.05), trainable=True)
        self.scale_b = self.add_weight(shape=(1,1), initializer=RandomNormal(mean=0.0, stddev=0.05), trainable=True)

    def call(self, inputs): 
        # Straight through estimator
        w_quantized_nonrounded = self.w / self.scale_w
        w_quantized_rounded = tf.stop_gradient(tf.floor(w_quantized_nonrounded)) + w_quantized_nonrounded - tf.stop_gradient(w_quantized_nonrounded)
        w_quantized_scaled_back = w_quantized_rounded * self.scale_w

        # Straight through estimator
        b_quantized_nonrounded = self.b / self.scale_b
        b_quantized_rounded = tf.stop_gradient(tf.floor(b_quantized_nonrounded)) + b_quantized_nonrounded - tf.stop_gradient(b_quantized_nonrounded)
        b_quantized_scaled_back = b_quantized_rounded * self.scale_b

        output = tf.matmul(inputs, w_quantized_scaled_back) + b_quantized_scaled_back

        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_scale_w(self):
        return self.scale_w
    
    def get_scale_b(self):
        return self.scale_b
