import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
import numpy as np

eps_float32 = np.finfo(np.float32).eps


class MinValueConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value):
        self.min_value = min_value

    def __call__(self, w):
        return tf.maximum(w, self.min_value)

    def get_config(self):
        return {'min_value': self.min_value}


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


# assume you can access the layer
@tf.custom_gradient
def custom_op(inputs, w, b, scale_w, scale_b):
    w_quantized_nonrounded = w / scale_w

    w_quantized_rounded = tf.stop_gradient(tf.floor(w_quantized_nonrounded))

    w_quantized_scaled_back = w_quantized_rounded * scale_w

    b_quantized_nonrounded = b / scale_b

    b_quantized_rounded = tf.stop_gradient(tf.floor(b_quantized_nonrounded))
    b_quantized_scaled_back = b_quantized_rounded * scale_b

    output = tf.matmul(inputs, w_quantized_scaled_back) + b_quantized_scaled_back

    def custom_grad(dy):
        #dy is the gradient of the loss with respect to the output of this custom operation
        # Last layer
        # dy has shape              (32, 10) where 32 is the batch size
        # tf.transpose(w) has shape (10, 128) 
        # grad has shape            (32, 128) 
        grad = dy @ tf.transpose(w) 
        # inputs has shape          (32, 128)
        # dw has shape              (128, 10)
        dw = tf.transpose(inputs) @ dy 
        # db has shape              (10,)
        db = tf.reduce_sum(dy, axis=0)

        # grad_scale_w has shape    (128, 1)
        grad_scale_w = tf.zeros_like(scale_w)
        grad_scale_b = tf.zeros_like(scale_b)
        
        grad_vars = [dw, db, grad_scale_w, grad_scale_b] 

        return grad, grad_vars

    return output, custom_grad


class RowWiseQuantized(tf.keras.layers.Layer):
    """
    This is a custom layer that implements a dense (fully connected) layer with
    learned quantization of weights and biases without the straight through estimator.
    """    
    def __init__(self, units, activation=None):
        super(RowWiseQuantized, self).__init__()
        self.units = units

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
      #  self.scale_w = self.add_weight(shape=(input_shape[-1], 1), initializer=RandomNormal(mean=0.0, stddev=0.0000001), trainable=True)
      #  self.scale_b = self.add_weight(shape=(1,1), initializer=RandomNormal(mean=0.0, stddev=0.0000001), trainable=True)
        self.scale_b = self.add_weight(shape=(1,1), initializer=tf.keras.initializers.Constant(eps_float32*100), trainable=True, constraint = MinValueConstraint(eps_float32))
        self.scale_w = self.add_weight(shape=(input_shape[-1], 1), initializer=tf.keras.initializers.Constant(eps_float32*100), trainable=True, constraint = MinValueConstraint(eps_float32))

    def call(self, inputs): 
        return custom_op(inputs, self.w, self.b, self.scale_w, self.scale_b)

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
        self.scale_b = self.add_weight(shape=(1,1), initializer=tf.keras.initializers.Constant(eps_float32*100), trainable=True, constraint = MinValueConstraint(eps_float32))
        self.scale_w = self.add_weight(shape=(input_shape[-1], 1), initializer=tf.keras.initializers.Constant(eps_float32*100), trainable=True, constraint = MinValueConstraint(eps_float32))

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
        self.scale_w = self.add_weight(shape=(1,self.units), initializer=tf.keras.initializers.Constant(eps_float32*100), trainable=True, constraint = MinValueConstraint(eps_float32))
        self.scale_b = self.add_weight(shape=(1,1), initializer=tf.keras.initializers.Constant(eps_float32*100), trainable=True, constraint = MinValueConstraint(eps_float32))

    def call(self, inputs): 
        return custom_op(inputs, self.w, self.b, self.scale_w, self.scale_b)

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


class QuantizedByAScalar(tf.keras.layers.Layer):
    """
    UNDER DEVELOPMENT
    """    
    def __init__(self, units, activation=None):
        super(QuantizedByAScalar, self).__init__()
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
        self.scalar = self.add_weight(shape=(1,1), initializer=RandomNormal(mean=0.0, stddev=0.05), trainable=True)

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
