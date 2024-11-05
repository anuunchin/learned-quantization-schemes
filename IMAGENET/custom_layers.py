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


# assume you can access the layer
@tf.custom_gradient
def quantize_data_layer_gradient(inputs, scale):
    inputs_quantized_nonrounded = inputs / scale # changes in the scale affect the output inversely

    inputs_quantized_rounded = tf.stop_gradient(tf.floor(inputs_quantized_nonrounded))

    inputs_quantized_scaled_back = inputs_quantized_rounded * scale

    output = inputs_quantized_scaled_back

    def custom_grad(dy, variables=None): #dL / doutput = dy
#        return dy, tf.zeros_like(scale) this corresponds to vanilla

        # d L / d inputs = (d L / d output) * (d output / d inputs) 
        #   
        # d output / d inputs = (d / d inputs) * (inputs/ scale) = 1 / scale
        #
        # d L / d inputs = dy * 1 / scale = dy / scale


        scale_w_broadcast = tf.broadcast_to(scale, tf.shape(inputs))
        grad_inputs = dy / scale_w_broadcast


        # d L / d scale = (dL / d output )  * ( d output / d scale)
        #  
        # d output / d scale =  - inputs / scale^2

        grad_scale_w = dy * (-1) * (inputs / tf.square(scale_w_broadcast))

        grad_scale_w = tf.reduce_sum(grad_scale_w, axis=0, keepdims=True) 

        grad_scale_w *= 10

        return grad_inputs, grad_scale_w

    return output, custom_grad


class QuantizedDataLayer(tf.keras.layers.Layer):

    def __init__(self, initializer):
        super(QuantizedDataLayer, self).__init__()
        self.initializer = initializer

    def build(self, input_shape):
            # How do we initialize the scales for the input data in a meaningful way? 
            # Initializing with 119 is not exactly suitable because even during the first run all values are divided into two bins
            # So it's actually meaningful to initialize with a smaller number
            # and let the model increase them to the point where it's optimal - maybe even resulting in binary binning
            # also this is probably working onlyon the mnist data
        self.scale = self.add_weight(
            shape=(1, input_shape[1]), 
            initializer=tf.keras.initializers.Constant(255.0/100), 
            trainable=True, 
            constraint = MinValueConstraint(1.0)
            )


    def call(self, inputs): 
        return quantize_data_layer_gradient(inputs, self.scale)

    def get_scale(self):
        return self.scale
    

@tf.custom_gradient
def quantize_layer_gradient(parameter, scale):
    inputs_quantized_nonrounded = parameter / scale # changes in the scale affect the output inversely

    inputs_quantized_rounded = tf.stop_gradient(tf.floor(inputs_quantized_nonrounded))

    inputs_quantized_scaled_back = inputs_quantized_rounded * scale

    output = inputs_quantized_scaled_back

    def custom_grad(dy, variables=None):

        scale_broadcasted = tf.broadcast_to(scale, tf.shape(parameter))

        parameter_grads = dy / scale_broadcasted

        scale_grads = dy  * (-1) * (parameter / tf.square(scale_broadcasted))

        if len(scale_grads.shape) == 1:
            # this needs a graceful handling according to the direction of quantization
            scale_grads = tf.reduce_sum(scale_grads, axis=0, keepdims=True) 
        else:
            scale_grads = tf.reduce_sum(scale_grads, axis=1, keepdims=True) 

        return parameter_grads, scale_grads

    return output, custom_grad

class QuantizedDenseLayer(tf.keras.layers.Layer):

    def __init__(self, initializer, orientation):
        super(QuantizedDenseLayer, self).__init__()
        self.initializer = initializer
        self.orientation = orientation

        setup_logger("logs/test", ["call_values.log"])

    def build(self, input_shape):
        shape = list(input_shape)
        
        # no None dimensions, so adjust
        if len(shape) == 1:
            shape = [1,1]
        elif input_shape[0] is None:
            shape[0] = 1
        elif input_shape[1] is None:
            shape[1] = 1

        if self.orientation == "rowwise":
            self.scale = self.add_weight(name="Rowwise-scaler", shape=(shape[0], 1), initializer=tf.keras.initializers.Constant(eps_float32*100), trainable=True, constraint = MinValueConstraint(eps_float32))
        elif self.orientation == "columnwise": # aroun 1k times lower than max
            self.scale = self.add_weight(name="Columnwise-scaler", shape=(1, shape[1]), initializer=tf.keras.initializers.Constant(eps_float32*100), trainable=True, constraint = MinValueConstraint(eps_float32))
        else:
            self.scale = self.add_weight(name="Scalar-scaler",shape=(1, ), initializer=tf.keras.initializers.Constant(eps_float32*100), trainable=True, constraint = MinValueConstraint(eps_float32))


    def call(self, inputs): 
        tf.print("Called", output_stream=f'file://logs/test/call_values.log')

        return quantize_layer_gradient(inputs, self.scale)

    def get_scale(self):
        return self.scale
    

class CustomDenseLayerNew(tf.keras.layers.Layer):

    def __init__(self, units, orientation="rowwise", l2_factor=0.0001):
        super(CustomDenseLayerNew, self).__init__()
        self.units = units
        self.nested_q_w_layer = QuantizedDenseLayer(initializer=None, orientation=orientation)
        self.nested_q_b_layer = QuantizedDenseLayer(initializer=None, orientation="scalar")

        self.l2_factor = l2_factor

        setup_logger("logs/test", ["default_call_values.log"])

    def build(self, input_shape):
        self.W = self.add_weight(
            name="Weights",
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            regularizer=tf.keras.regularizers.l2(self.l2_factor),
            trainable=True
        )

        self.b = self.add_weight(
            name="Bias",
            shape=(self.units,),
            initializer="random_normal",
            regularizer=tf.keras.regularizers.l2(self.l2_factor),
            trainable=True
        )

        print("W shape", self.W.shape)
        print("b shape", self.b.shape)


    def call(self, inputs): 

        qw = self.nested_q_w_layer(self.W)
        qb = self.nested_q_b_layer(self.b)

        tf.print("Called", output_stream=f'file://logs/test/default_call_values.log')

        return tf.add(tf.matmul(inputs, qw), qb)

