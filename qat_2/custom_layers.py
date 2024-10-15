import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
import numpy as np
import logging

eps_float32 = np.finfo(np.float32).eps

log_dir = ""

def setup_logger(new_log_dir, logs):
    # Set up the first logger for total loss
    global log_dir
    # Update the global log_dir
    log_dir = new_log_dir

    if logs != []:
        for log in logs:
            # Set up the logger
            total_loss_logger = tf.get_logger()
            total_loss_handler = logging.FileHandler(f'{new_log_dir}/{log}', mode='a')
            total_loss_handler.setFormatter(logging.Formatter('%(message)s'))
            total_loss_logger.addHandler(total_loss_handler)
            total_loss_logger.setLevel(logging.INFO)

            # Clear the content
            with open(f'{new_log_dir}/{log}', 'w'):
                pass

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
        #self.scale_w = self.add_weight(shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
        #self.scale_b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

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

        num_bins_w = tf.reduce_max(tf.abs(w / scale_w), axis=0)
        num_bins_b = tf.reduce_max(tf.abs(b / scale_b))

        bin_penalty_w = tf.reduce_mean(num_bins_w)
        bin_penalty_b = num_bins_b
        
        total_bin_penalty = bin_penalty_w + bin_penalty_b

        # Log bins
        tf.print(bin_penalty_w, output_stream=f'file://{log_dir}/bins_w.log')
        tf.print(bin_penalty_b, output_stream=f'file://{log_dir}/bins_b.log')
        tf.print(total_bin_penalty, output_stream=f'file://{log_dir}/bins_total.log')

        bin_threshold = 10.0  # Adjust based on experiment

        # Conditionally adjust the scale gradients based on the number of bins
        grad_vars = tf.cond(
            total_bin_penalty > bin_threshold,
            lambda: [dw, db, grad_scale_w * 0.1, grad_scale_b * 0.1],  # We reduce quantization adjustment
            lambda: [dw, db, grad_scale_w, grad_scale_b]  # Allow full scale updates
        )

        return grad, grad_vars

    return output, custom_grad


class RowWiseQuantized(tf.keras.layers.Layer):
    """
    This is a custom layer that implements a dense (fully connected) layer with
    learned quantization of weights and biases without the straight through estimator.
    """    
    def __init__(self, units, log_dir, activation=None):
        super(RowWiseQuantized, self).__init__()
        self.units = units
        self.log_dir = log_dir

        # Define logs
        self.logs = [
#            "gradients_w.log",
#            "gradients_b.log",
#            "bins_w.log",
#            "bins_b.log",
#            "bins_total.log",
        ]

        setup_logger(self.log_dir, self.logs)

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
        return custom_op(inputs, self.w, self.b, self.scale_w, self.scale_b)

    def get_scale_w(self):
        return self.scale_w
    
    def get_scale_b(self):
        return self.scale_b


# assume you can access the layer
@tf.custom_gradient
def quantize_layer_gradient(inputs, scale_w):
    w_quantized_nonrounded = inputs / scale_w

    w_quantized_rounded = tf.stop_gradient(tf.floor(w_quantized_nonrounded))

    w_quantized_scaled_back = w_quantized_rounded * scale_w

    output = w_quantized_scaled_back

    def custom_grad(dy):
        #dy is the gradient of the loss with respect to the output of this custom operation
        # Last layer
        # dy has shape              (32, 10) where 32 is the batch size
        # tf.transpose(w) has shape (10, 128) 
        # grad has shape            (32, 128) 
        #grad = dy @ tf.transpose(w) 
        
        # inputs has shape          (32, 128)
        # dw has shape              (128, 10)
        #dw = tf.transpose(inputs) @ dy 
         #db has shape              (10,)
        #db = tf.reduce_sum(dy, axis=0)

        # grad_scale_w has shape    (128, 1)
        #grad_scale_w = tf.zeros_like(scale_w)
        
        return dy, tf.zeros_like(scale_w)

    return output, custom_grad


class QuantizedLayer(tf.keras.layers.Layer):
    """
    This is a custom layer that implements a dense (fully connected) layer with
    learned quantization of weights and biases without the straight through estimator.
    """    
    def __init__(self, shape, initializer, orientation):
        super(QuantizedLayer, self).__init__()
        self.shape = shape
        self.initializer = initializer
        self.orientation = orientation

    def build(self, input_shape):
        """
        Example shapes:
        self.w:         (784, 128)
        self.b:         (128, )
        self.scale_w:   (784, 1) applied row-wise
        self.scale_b:   (1, 1)
        """
#        self.w = self.add_weight(shape=self.shape, initializer=self.initializer, trainable=True)
     ##   self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
      #  self.scale_w = self.add_weight(shape=(input_shape[-1], 1), initializer=RandomNormal(mean=0.0, stddev=0.0000001), trainable=True)
      #  self.scale_b = self.add_weight(shape=(1,1), initializer=RandomNormal(mean=0.0, stddev=0.0000001), trainable=True)
      #  self.scale_b = self.add_weight(shape=(1,1), initializer=tf.keras.initializers.Constant(eps_float32*100), trainable=True, constraint = MinValueConstraint(eps_float32))
        if self.orientation == "rowwise":
            self.scale_w = self.add_weight(shape=(self.shape[0], 1), initializer=tf.keras.initializers.Constant(eps_float32*100), trainable=True, constraint = MinValueConstraint(eps_float32))
        else:
            self.scale_w = self.add_weight(shape=(1, self.shape[1]), initializer=tf.keras.initializers.Constant(eps_float32*100), trainable=True, constraint = MinValueConstraint(eps_float32))
        
        # column vector that we're quantizing with
        # 1, self.shape[1]

    def call(self, inputs): 
        return quantize_layer_gradient(inputs, self.scale_w)

    def get_scale_w(self):
        return self.scale_w
        

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
       # self.w = QuantizedLayer(shape=(input_shape[-1], self.units), initializer="random_normal", orientation="rowwise")
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.scale_w = QuantizedLayer
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
        #self.scale_w = self.add_weight(shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
        #self.scale_b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs): 
        output = tf.matmul(inputs, self.w) + self.b
       
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_scale_w(self):
        return None
    
    def get_scale_b(self):
        return None


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
