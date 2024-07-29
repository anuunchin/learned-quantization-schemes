import tensorflow as tf
import numpy as np

eps_float32 = np.finfo(np.float32).eps

class SCCE:
    def __init__(self, layers, penalty_rate):
        self.weight_scales = [layer.get_scale_w() for layer in layers]
        self.bias_scales = [layer.get_scale_b() for layer in layers]
        self.weights = [layer.w for layer in layers]
        self.biases = [layer.b for layer in layers]
        self.penalty_rate = penalty_rate
        
    def compute_total_loss(self, y_true, y_pred):
        """
        Computes the sparse categorical cross-entropy loss.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return cross_entropy_loss

    def compute_scale_penalty(self):
        """
        Returns 0.0 because no custom loss is added. The existence of this method is justified by the structure of other loss classes.
        """
        return tf.constant(0.0, dtype=tf.float32)

    def get_name(self):
        return "SCCE"


class SCCEInverse:
    def __init__(self, layers, penalty_rate):
        self.weight_scales = [layer.get_scale_w() for layer in layers]
        self.bias_scales = [layer.get_scale_b() for layer in layers]
        self.weights = [layer.w for layer in layers]
        self.biases = [layer.b for layer in layers]
        self.penalty_rate = penalty_rate

    def compute_total_loss(self, y_true, y_pred):
        """
        Computes a combined loss that includes sparse categorical cross-entropy and the inverse of the average of scaling factor values.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        scale_penalty = self.compute_scale_penalty()        
        total_loss = cross_entropy_loss + scale_penalty
        return total_loss

    def compute_scale_penalty(self):
        """
        Computes the inverse of the average of scaling factor values multiplied by the penalty rate.
        Effectively punishes small scale factor values.
        """
        scale_penalty = 0

        for layer_index in range(len(self.weight_scales)):
            
            layer_weight_scales = self.weight_scales[layer_index]
            layer_bias_scales = self.bias_scales[layer_index]

            # Check if the layer has scale factor values
            if layer_weight_scales is None and layer_bias_scales is None:
                return tf.constant(0.0, dtype=tf.float32)

            mean_inverse_w = tf.reduce_mean(1.0 / (tf.abs(layer_weight_scales) + eps_float32))
            mean_inverse_b = tf.reduce_mean(1.0 / (tf.abs(layer_bias_scales) + eps_float32))

            mean_inverse = (mean_inverse_w + mean_inverse_b) / 2
            # The above thing isn't ding the broadcasting correctly - most likely
            # This issue for all

            scale_penalty += mean_inverse

        scale_penalty /= len(self.weight_scales)

        return scale_penalty * self.penalty_rate
    
    def get_name(self):
        return "SCCEInverse"


class SCCEMinMaxBin:
    def __init__(self, layers, penalty_rate, row_wise):
        self.weight_scales = [layer.get_scale_w() for layer in layers]
        self.bias_scales = [layer.get_scale_b() for layer in layers]
        self.weights = [layer.w for layer in layers]
        self.biases = [layer.b for layer in layers]
        self.penalty_rate = penalty_rate
        self.application_of_scale_factors = row_wise

    def compute_total_loss(self, y_true, y_pred):
        """
        Computes a combined loss that includes sparse categorical cross-entropy and a penalty based on the range of quantization bins.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        scale_penalty = self.compute_scale_penalty()
        
        total_loss = cross_entropy_loss + scale_penalty

        return total_loss

    def compute_scale_penalty(self):
        """
        Computes the penalty based on the range of quantization bins for weight and bias scales multiplied by the penalty rate.
        """
        # Needs to be adjusted more in the parts where tf.abs is applied
        scale_penalty = 0

        for layer_index in range(len(self.weight_scales)):
            # For each layer
            layer_weight_scales = self.weight_scales[layer_index]
            layer_bias_scales = self.bias_scales[layer_index]

            # Check if the layer has scale factor values
            if layer_weight_scales is None and layer_bias_scales is None:
                return tf.constant(0.0, dtype=tf.float32)
            
            layer_weights = self.weights[layer_index]
            layer_biases = self.biases[layer_index]

            max_w = tf.reduce_max(tf.abs(layer_weights), axis=self.application_of_scale_factors) 
            min_w = tf.reduce_min(tf.abs(layer_weights), axis=self.application_of_scale_factors) 

            max_b = tf.reduce_max(tf.abs(layer_biases)) # scalar
            min_b = tf.reduce_min(tf.abs(layer_biases)) # scalar

            max_w_quantized = tf.floor((max_w / (tf.abs(layer_weight_scales) + eps_float32))) # 10
            min_w_quantized = tf.floor((min_w / (tf.abs(layer_weight_scales) + eps_float32))) # 1
            max_w_scaled_back = max_w_quantized * tf.abs(layer_weight_scales) # 20
            min_w_scaled_back = min_w_quantized * tf.abs(layer_weight_scales) # 2

            range_of_quant_w_bins = (max_w_quantized - min_w_quantized) / tf.abs(layer_weight_scales) + 1 # for each row

            max_b_quantized = tf.floor((max_b / (tf.abs(layer_bias_scales) + eps_float32)))
            min_b_quantized = tf.floor((min_b / (tf.abs(layer_bias_scales) + eps_float32)))
            max_b_scaled_back = max_b_quantized * tf.abs(layer_bias_scales)
            min_b_scaled_back = min_b_quantized * tf.abs(layer_bias_scales)

            range_of_quant_b_bins = (max_b_quantized - min_b_quantized) / tf.abs(layer_bias_scales) + 1

            scale_penalty += tf.reduce_mean(range_of_quant_w_bins)
            scale_penalty += tf.reduce_mean(range_of_quant_b_bins)

        scale_penalty /= len(self.weight_scales)

        return scale_penalty * self.penalty_rate
    
    def get_name(self):
        return "SCCEMinMaxBin"


class SCCEMaxBin:
    def __init__(self, layers, penalty_rate, row_wise):
        self.weight_scales = [layer.get_scale_w() for layer in layers]
        self.bias_scales = [layer.get_scale_b() for layer in layers]
        self.weights = [layer.w for layer in layers]
        self.biases = [layer.b for layer in layers]
        self.penalty_rate = penalty_rate
        self.application_of_scale_factors = row_wise

    def compute_total_loss(self, y_true, y_pred):
        """
        Computes a combined loss that includes sparse categorical cross-entropy and a penalty based on the number of bins calculated from the max weights divided by the quantization factor.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        scale_penalty = self.compute_scale_penalty()
        
        total_loss = cross_entropy_loss + scale_penalty

        return total_loss

    def compute_scale_penalty(self):
        """
        Computes the penalty based on the number of bins calculated from the max weights divided by the quantization factor.        
        Effectively punished large number of bins.
        """

        scale_penalty = 0

        for layer_index in range(len(self.weight_scales)):

            layer_weight_scales = self.weight_scales[layer_index]
            layer_bias_scales = self.bias_scales[layer_index]

            # Check if the layer has scale factor values
            if layer_weight_scales is None and layer_bias_scales is None:
                return tf.constant(0.0, dtype=tf.float32)

            layer_weights = self.weights[layer_index]
            layer_biases = self.biases[layer_index]

            max_w_per_row = tf.reduce_max(tf.abs(tf.floor(layer_weights / layer_weight_scales)), axis=self.application_of_scale_factors)            
            max_b = tf.reduce_max(tf.abs(tf.floor(layer_biases / layer_bias_scales)))

            bins_w = max_w_per_row + 1 # 1 accounts for 0 - assuming bits for signs(+,-) can be ignored
            bins_b = max_b + 1

#            average_bins = (tf.reduce_mean(bins_w) + tf.reduce_mean(bins_b)) / 2

            scale_penalty += tf.reduce_sum(bins_w)
            scale_penalty += tf.reduce_sum(bins_b)

        scale_penalty /= len(self.weight_scales)

        return scale_penalty * self.penalty_rate
    
    def get_name(self):
        return "SCCEMaxBin"


class SCCEDifference:
    def __init__(self, layers, penalty_rate):
        self.weight_scales = [layer.get_scale_w() for layer in layers]
        self.bias_scales = [layer.get_scale_b() for layer in layers]
        self.weights = [layer.w for layer in layers]
        self.biases = [layer.b for layer in layers]
        self.penalty_rate = penalty_rate

    def compute_total_loss(self, y_true, y_pred):
        """
        Computes a combined loss that includes sparse categorical cross-entropy and a penalty based on the difference
        between the original and quantized-scaled weights and biases.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        scale_penalty = self.compute_scale_penalty()
        total_loss = cross_entropy_loss + scale_penalty
        return total_loss
    
    def compute_scale_penalty(self):
        """
        Computes the penalty based on the difference between the original and quantized-scaled weights and biases.
        """
        # Something wrong here, larger differences should be encouraged, thus we should take the inverse of smth alike
        scale_penalty = 0

        for layer_index in range(len(self.weight_scales)):
           
            layer_weight_scales = self.weight_scales[layer_index]
            layer_bias_scales = self.bias_scales[layer_index]

            # Check if the layer has scale factor values
            if layer_weight_scales is None and layer_bias_scales is None:
                return tf.constant(0.0, dtype=tf.float32)

            layer_weights = self.weights[layer_index]
            layer_biases = self.biases[layer_index]

            w_quantized_rounded = tf.floor(layer_weights / layer_weight_scales)
            w_quantized_scaled_back = w_quantized_rounded * layer_weight_scales

            b_quantized_rounded = tf.floor(layer_biases / layer_bias_scales)
            b_quantized_scaled_back = b_quantized_rounded * layer_bias_scales

            diff_w = tf.reduce_mean(tf.abs(layer_weights - w_quantized_scaled_back))
            diff_b = tf.reduce_mean(tf.abs(layer_biases - b_quantized_scaled_back))

            diff = (diff_w + diff_b) / 2 # the shapes are being broadcasted wrong - most likely

            scale_penalty += diff

        return scale_penalty * self.penalty_rate

    def get_name(self):
        return "SCCEDifference"