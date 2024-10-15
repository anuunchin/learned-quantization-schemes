import tensorflow as tf
import numpy as np
import logging


def setup_logger(log_dir, logs):
    if logs != []:
        for log in logs:
            # Set up the logger
            total_loss_logger = tf.get_logger()
            total_loss_handler = logging.FileHandler(f'{log_dir}/{log}', mode='a')
            total_loss_handler.setFormatter(logging.Formatter('%(message)s'))
            total_loss_logger.addHandler(total_loss_handler)
            total_loss_logger.setLevel(logging.INFO)

            # Clear the content
            with open(f'{log_dir}/{log}', 'w'):
                pass    


eps_float32 = np.finfo(np.float32).eps

class SCCE:
    def __init__(self, layers, penalty_rate, row_wise, log_dir):
        self.weight_scales = [layer.get_scale_w() for layer in layers]
        self.bias_scales = [layer.get_scale_b() for layer in layers]
        self.weights = [layer.w for layer in layers]
        self.biases = [layer.b for layer in layers]
        self.penalty_rate = penalty_rate
        self.log_dir = log_dir

        setup_logger(self.log_dir)

    def compute_total_loss(self, y_true, y_pred):
        """
        Computes the sparse categorical cross-entropy loss.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        scale_penalty = self.compute_scale_penalty()

        total_loss = cross_entropy_loss + scale_penalty

        tf.print(tf.reduce_mean(total_loss), output_stream=f'file://{self.log_dir}/total_loss_log.log')

        return total_loss

    def compute_scale_penalty(self):
        """
        Returns 0.0 because no custom loss is added. The existence of this method is justified by the structure of other loss classes.
        """
        scale_penalty = tf.constant(0.0, dtype=tf.float32)

        tf.print(tf.reduce_mean(scale_penalty), output_stream=f'file://{self.log_dir}/scale_loss_log.log')

        return scale_penalty

    def get_name(self):
        return "SCCE"


class SCCEInverse:
    def __init__(self, layers, penalty_rate, row_wise, log_dir):
        self.weight_scales = [layer.get_scale_w() for layer in layers]
        self.bias_scales = [layer.get_scale_b() for layer in layers]
        self.weights = [layer.w for layer in layers]
        self.biases = [layer.b for layer in layers]
        self.penalty_rate = penalty_rate
        self.log_dir = log_dir

        setup_logger(self.log_dir)

    def compute_total_loss(self, y_true, y_pred):
        """
        Computes a combined loss that includes sparse categorical cross-entropy and the inverse of the average of scaling factor values.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        scale_penalty = self.compute_scale_penalty()        
        
        total_loss = cross_entropy_loss + scale_penalty

        tf.print(tf.reduce_mean(total_loss), output_stream=f'file://{self.log_dir}/total_loss_log.log')

        return total_loss


    def compute_scale_penalty(self):
        """
        Computes the inverse of the average of scaling factor values multiplied by the penalty rate.
        Effectively punishes small scale factor values.
        """
        scale_penalty = 0

        scale_num = 0

        for layer_index in range(len(self.weight_scales)):
            
            layer_weight_scales = self.weight_scales[layer_index]
            layer_bias_scales = self.bias_scales[layer_index]

            # Check if the layer has scale factor values
            if layer_weight_scales is None and layer_bias_scales is None:
                return tf.constant(0.0, dtype=tf.float32)


            dim_w = layer_weight_scales.shape[0]
            dim_b = 1
            scale_num = dim_w + dim_b

            mean_inverse_w = tf.reduce_mean(1.0 / layer_weight_scales) # Resulint in layer_weight_scales
            mean_inverse_b = tf.reduce_mean(1.0 / layer_bias_scales)

            # tf.reduce_mean(concat(layer_weight_scale, layer_bias_scales)) -> makes next line unnecessary

            mean_inverse = mean_inverse_w * dim_w + mean_inverse_b * dim_b

            scale_penalty += mean_inverse

            # this doesn't account for the size of the layer (the number of cells, bc 128 != 784)
        scale_penalty /= scale_num

        scale_penalty *= self.penalty_rate

        tf.print(tf.reduce_mean(scale_penalty), output_stream=f'file://{self.log_dir}/scale_loss_log.log')

        return scale_penalty
    
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

        # Clear the contents of both log files by opening them in write mode and then closing them
        with open('logs/total_loss_log.txt', 'w'), open('logs/scale_loss_log.txt', 'w'):
            pass        


    def compute_total_loss(self, y_true, y_pred):
        """
        Computes a combined loss that includes sparse categorical cross-entropy and a penalty based on the range of quantization bins.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        scale_penalty = self.compute_scale_penalty()
        
        total_loss = cross_entropy_loss + scale_penalty

        tf.print("Loss:", tf.reduce_mean(total_loss), output_stream='file://logs/total_loss_log.txt')

        return total_loss

    def compute_scale_penalty(self):
        """
        Computes the penalty based on the range of quantization bins for weight and bias scales multiplied by the penalty rate.
        """
        # Needs to be adjusted more in the parts where tf.abs is applied
        scale_penalty = 0

        scale_num = 0

        for layer_index in range(len(self.weight_scales)):
            # For each layer
            layer_weight_scales = self.weight_scales[layer_index]
            layer_bias_scales = self.bias_scales[layer_index]

            # Check if the layer has scale factor values
            if layer_weight_scales is None and layer_bias_scales is None:
                return tf.constant(0.0, dtype=tf.float32)
            
            layer_weights = self.weights[layer_index]
            layer_biases = self.biases[layer_index]

            #max_w = tf.reduce_max(tf.abs(layer_weights), axis=self.application_of_scale_factors) 
            #min_w = tf.reduce_min(tf.abs(layer_weights), axis=self.application_of_scale_factors) ss
            max_w_quantized = tf.floor(tf.reduce_max(tf.abs(layer_weights / layer_weight_scales), axis=self.application_of_scale_factors))
            min_w_quantized = tf.floor(tf.reduce_min(tf.abs(layer_weights / layer_weight_scales), axis=self.application_of_scale_factors))

            print("TESTING SHAPE OF W: ", layer_weights.shape)
            print("TESTING SHAPE OF max_w_quantized: ", max_w_quantized.shape)
            print("TESTING SHAPE OF layer_weight_scales: ", layer_weight_scales.shape)

            #max_b = tf.reduce_max(tf.abs(layer_biases)) # scalar
            #min_b = tf.reduce_min(tf.abs(layer_biases)) # scalar

            max_b_quantized = tf.floor(tf.reduce_max(tf.abs(layer_biases / layer_bias_scales)))
            min_b_quantized = tf.floor(tf.reduce_min(tf.abs(layer_biases / layer_bias_scales)))

            #max_w_quantized = tf.floor((max_w / (tf.abs(layer_weight_scales) + eps_float32))) # 10
            #min_w_quantized = tf.floor((min_w / (tf.abs(layer_weight_scales) + eps_float32))) # 1
            #max_w_scaled_back = max_w_quantized * tf.abs(layer_weight_scales) # 20
            #min_w_scaled_back = min_w_quantized * tf.abs(layer_weight_scales) # 2

            range_of_quant_w_bins = tf.divide(
                tf.reshape(max_w_quantized - min_w_quantized, (-1, 1)), 
                layer_weight_scales
            )
            #print("TERMS:", (max_w_quantized - min_w_quantized).shape, layer_weight_scales.shape)
            #print("RANGE:", range_of_quant_w_bins.shape)

            #max_b_quantized = tf.floor((max_b / (tf.abs(layer_bias_scales) + eps_float32)))
            #min_b_quantized = tf.floor((min_b / (tf.abs(layer_bias_scales) + eps_float32)))
            #max_b_scaled_back = max_b_quantized * tf.abs(layer_bias_scales)
            #min_b_scaled_back = min_b_quantized * tf.abs(layer_bias_scales)

            #range_of_quant_b_bins = (max_b_quantized - min_b_quantized) / tf.abs(layer_bias_scales) + 1

            range_of_quant_b_bins = tf.divide(
                tf.reshape(max_b_quantized - min_b_quantized, (-1, 1)), 
                layer_bias_scales
            )

            dim_w = range_of_quant_w_bins[0]
            dim_b = 1
            scale_num += dim_b + dim_w

            average_range_w_bins = tf.reduce_mean(range_of_quant_w_bins) * dim_w + tf.reduce_mean(range_of_quant_b_bins) * dim_b

            scale_penalty += average_range_w_bins

        scale_penalty /= scale_num

        scale_penalty *= self.penalty_rate

        tf.print("Loss:", tf.reduce_mean(scale_penalty), output_stream='file://logs/scale_loss_log.txt')

        return scale_penalty
    
    def get_name(self):
        return "SCCEMinMaxBin"

    
class SCCEMaxBin:
    def __init__(self, layers, penalty_rate, row_wise, log_dir, l2_lambda=0.01):
        self.weight_scales = [layer.get_scale_w() for layer in layers]
        self.bias_scales = [layer.get_scale_b() for layer in layers]
        self.weights = [layer.w for layer in layers]
        self.biases = [layer.b for layer in layers]
        self.penalty_rate = penalty_rate
        self.application_of_scale_factors = row_wise
        self.log_dir = log_dir
        self.l2_lambda = l2_lambda
        self.logs = [
            "total_loss_log.log",
            "scale_loss_log.log",
            "l2_loss_log.log",
            "bins_w.log",
            "bins_b.log",
            "bins_average.log"
        ]


        setup_logger(self.log_dir, self.logs)

    def compute_total_loss(self, y_true, y_pred):
        """
        Computes a combined loss that includes sparse categorical cross-entropy and a penalty based on the number of bins calculated from the max weights divided by the quantization factor.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        scale_penalty = self.compute_scale_penalty()
        
        l2_loss = self.compute_l2_regularization()

        total_loss = cross_entropy_loss + scale_penalty + l2_loss

        tf.print(tf.reduce_mean(total_loss), output_stream=f'file://{self.log_dir}/total_loss_log.log')

        return total_loss

    def compute_l2_regularization(self):
        """
        Computes the L2 regularization term based on the sum of squared weights and biases.
        """
        l2_loss = 0.0
        for layer_weights, layer_biases in zip(self.weights, self.biases):
            l2_loss += tf.reduce_sum(tf.square(layer_weights))  
            l2_loss += tf.reduce_sum(tf.square(layer_biases))    

        l2_loss *= self.l2_lambda  

        tf.print(tf.reduce_mean(l2_loss), output_stream=f'file://{self.log_dir}/l2_loss_log.log')

        return l2_loss

    def compute_scale_penalty(self):
        """
        Computes the penalty based on the number of bins calculated from the max weights divided by the quantization factor.        
        Effectively punishes large number of bins.
        """

        scale_penalty = 0

        scale_num = 0

        for layer_index in range(len(self.weight_scales)):

            layer_weight_scales = self.weight_scales[layer_index]
            layer_bias_scales = self.bias_scales[layer_index]

            # Check if the layer has scale factor values
            if layer_weight_scales is None and layer_bias_scales is None:
                return tf.constant(0.0, dtype=tf.float32)

            layer_weights = self.weights[layer_index]
            layer_biases = self.biases[layer_index]

            max_w_per_row = tf.reduce_max(tf.abs(layer_weights / layer_weight_scales), axis=self.application_of_scale_factors)            
            max_b = tf.reduce_max(tf.abs(layer_biases / layer_bias_scales))

            bins_w = max_w_per_row
            bins_b = max_b

            # Log the floored mean values of bins
            tf.print(tf.floor(tf.reduce_mean(bins_w)), output_stream=f'file://{self.log_dir}/bins_w.log')
            tf.print(tf.floor(tf.reduce_mean(bins_b)), output_stream=f'file://{self.log_dir}/bins_b.log')

            dim_w = tf.cast(bins_w.shape[0], dtype=tf.float32)
            dim_b = tf.constant(1.0, dtype=tf.float32)

            average_bins = (tf.reduce_mean(bins_w) * dim_w + tf.reduce_mean(bins_b) * dim_b) / (dim_w + dim_b)
            tf.print(tf.floor(average_bins), output_stream=f'file://{self.log_dir}/bins_average.log')

            penalty_contribution, scale_num_contribution = tf.cond(
                tf.reduce_mean(average_bins) < 1,
                lambda: (tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)),  # Zero penalty and no contribution to scale_num
                lambda: (average_bins, dim_b + dim_w)  # Regular penalty and scale_num update
            )

            scale_penalty += penalty_contribution
            scale_num += scale_num_contribution

        scale_penalty = tf.cond(
            scale_num > 0,
            lambda: scale_penalty / scale_num,
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
        scale_penalty *= self.penalty_rate

        tf.print(tf.reduce_mean(scale_penalty), output_stream=f'file://{self.log_dir}/scale_loss_log.log')

        return scale_penalty
    
    def get_name(self):
        return "SCCEMaxBin"


class SCCEMaxBinVanilla:
    def __init__(self, layers, penalty_rate, row_wise, log_dir, l2_lambda=0.01):
        self.weight_scales = [layer.get_scale_w() for layer in layers]
        self.bias_scales = [layer.get_scale_b() for layer in layers]
        self.weights = [layer.w for layer in layers]
        self.biases = [layer.b for layer in layers]
        self.penalty_rate = penalty_rate
        self.application_of_scale_factors = row_wise
        self.log_dir = log_dir
        self.l2_lambda = l2_lambda
        self.logs = [
            "total_loss_log.log",
            "scale_loss_log.log",
            "l2_loss_log.log",
            "bins_w.log",
            "bins_b.log",
            "bins_average.log"
        ]

        setup_logger(self.log_dir, self.logs)

    def compute_total_loss(self, y_true, y_pred):
        """
        Computes a combined loss that includes sparse categorical cross-entropy and a penalty based on the number of bins calculated from the max weights divided by the quantization factor.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        scale_penalty = self.compute_scale_penalty()
        
        l2_loss = self.compute_l2_regularization()

        total_loss = cross_entropy_loss + scale_penalty + l2_loss

        tf.print(tf.reduce_mean(total_loss), output_stream=f'file://{self.log_dir}/total_loss_log.log')

        return total_loss

    def compute_l2_regularization(self):
        """
        Computes the L2 regularization term based on the sum of squared weights and biases.
        """
        l2_loss = 0.0
        for layer_weights, layer_biases in zip(self.weights, self.biases):
            l2_loss += tf.reduce_sum(tf.square(layer_weights))  
            l2_loss += tf.reduce_sum(tf.square(layer_biases))    

        l2_loss *= self.l2_lambda  

        tf.print(tf.reduce_mean(l2_loss), output_stream=f'file://{self.log_dir}/l2_loss_log.log')

        return l2_loss

    def compute_scale_penalty(self):
        """
        Computes the penalty based on the number of bins calculated from the max weights divided by the quantization factor.        
        Effectively punishes large number of bins.
        """

        scale_penalty = 0

        scale_num = 0

        for layer_index in range(len(self.weight_scales)):

            layer_weight_scales = self.weight_scales[layer_index]
            layer_bias_scales = self.bias_scales[layer_index]

            # Check if the layer has scale factor values
            if layer_weight_scales is None and layer_bias_scales is None:
                return tf.constant(0.0, dtype=tf.float32)

            layer_weights = self.weights[layer_index]
            layer_biases = self.biases[layer_index]

            max_w_per_row = tf.reduce_max(tf.abs(layer_weights / layer_weight_scales), axis=self.application_of_scale_factors)            
            max_b = tf.reduce_max(tf.abs(layer_biases / layer_bias_scales))

            bins_w = max_w_per_row
            bins_b = max_b

            dim_w = bins_w.shape[0]
            dim_b = 1
            scale_num += dim_b + dim_w

            average_bins = tf.reduce_mean(bins_w) * dim_w + tf.reduce_mean(bins_b) * dim_b

            scale_penalty += average_bins

        scale_penalty /= scale_num

        scale_penalty *= self.penalty_rate

        tf.print(tf.reduce_mean(scale_penalty), output_stream=f'file://{self.log_dir}/scale_loss_log.log')

        return scale_penalty
    
    def get_name(self):
        return "SCCEMaxBin"



class SCCEDifference:
    def __init__(self, layers, penalty_rate, row_wise, log_dir, l2_lambda=0.01):
        self.weight_scales = [layer.get_scale_w() for layer in layers]
        self.bias_scales = [layer.get_scale_b() for layer in layers]
        self.weights = [layer.w for layer in layers]
        self.biases = [layer.b for layer in layers]
        self.penalty_rate = penalty_rate
        self.application_of_scale_factors = row_wise
        self.log_dir = log_dir
        self.l2_lambda = l2_lambda
        self.logs = [
            "total_loss_log.log",
            "scale_loss_log.log",
            "l2_loss_log.log",
            "bins_w.log",
            "bins_b.log",
            "bins_average.log"
        ]

        print("TESTING")

        setup_logger(self.log_dir, self.logs)

    def compute_l2_regularization(self):
        """
        Computes the L2 regularization term based on the sum of squared weights and biases.
        """
        l2_loss = 0.0
        for layer_weights, layer_biases in zip(self.weights, self.biases):
            l2_loss += tf.reduce_sum(tf.square(layer_weights))  
            l2_loss += tf.reduce_sum(tf.square(layer_biases))    

        l2_loss *= self.l2_lambda  

        tf.print(tf.reduce_mean(l2_loss), output_stream=f'file://{self.log_dir}/l2_loss_log.log')

        return l2_loss

    def compute_total_loss(self, y_true, y_pred):
        """
        Computes a combined loss that includes sparse categorical cross-entropy and a penalty based on the difference
        between the original and quantized-scaled weights and biases.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        scale_penalty = self.compute_scale_penalty()

        l2_loss = self.compute_l2_regularization()

        scale_penalty_inverse = self.compute_scale_penalty_inverse()

        total_loss = cross_entropy_loss + scale_penalty + l2_loss + scale_penalty_inverse

        tf.print(tf.reduce_mean(total_loss), output_stream=f'file://{self.log_dir}/total_loss_log.log')

        return total_loss
    
    def compute_scale_penalty(self):
        """
        Computes the penalty based on the difference between the original and quantized-scaled weights and biases.
        """
        scale_penalty = 0

        scale_num = 0

        for layer_index in range(len(self.weight_scales)):
           
            layer_weight_scales = self.weight_scales[layer_index]
            layer_bias_scales = self.bias_scales[layer_index]

            # Check if the layer has scale factor values
            if layer_weight_scales is None and layer_bias_scales is None:
                return tf.constant(0.0, dtype=tf.float32)

            layer_weights = self.weights[layer_index]
            layer_biases = self.biases[layer_index]

#            def soft_round(x, beta=1.0):
#                return x - tf.sigmoid(beta * (x - tf.round(x)))


            #w_quantized_rounded = tf.floor(layer_weights / layer_weight_scales)
#            w_quantized_rounded = soft_round(layer_weights / layer_weight_scales)

#            w_quantized_scaled_back = w_quantized_rounded * layer_weight_scales

            #b_quantized_rounded = tf.floor(layer_biases / layer_bias_scales)
#            b_quantized_rounded = soft_round(layer_biases / layer_bias_scales)
#            b_quantized_scaled_back = b_quantized_rounded * layer_bias_scales


            w_quantized_scaled_back = tf.floor(layer_weights / layer_weight_scales) * layer_weight_scales
            b_quantized_scaled_back = tf.floor(layer_biases / layer_bias_scales) * layer_bias_scales

            diff_w = tf.reduce_mean(tf.abs(layer_weights - w_quantized_scaled_back))
            diff_b = tf.reduce_mean(tf.abs(layer_biases - b_quantized_scaled_back))

            dim_w = w_quantized_scaled_back.shape[0]
            dim_b = 1
            scale_num += dim_w + dim_b
            
            # Reward larger differences between original and quantized (before scaling back)
#            quant_diff_w = tf.reduce_mean(tf.abs(layer_weights - w_quantized_rounded))
#            quant_diff_b = tf.reduce_mean(tf.abs(layer_biases - b_quantized_rounded))

            diff = diff_w * dim_w + diff_b * dim_b

            # Reward smaller differences between original and quantized scaled 

#            quant_diff = quant_diff_w * dim_w + quant_diff_b * dim_b

            scale_penalty += diff

        scale_penalty /= scale_num

        scale_penalty *= self.penalty_rate

        tf.print(tf.reduce_mean(scale_penalty), output_stream=f'file://{self.log_dir}/scale_loss_log.log')

        return scale_penalty

    def compute_scale_penalty_inverse(self):
        """
        Computes the penalty based on the number of bins calculated from the max weights divided by the quantization factor.        
        Effectively punishes large number of bins.
        """

        scale_penalty = 0

        scale_num = 0

        for layer_index in range(len(self.weight_scales)):

            layer_weight_scales = self.weight_scales[layer_index]
            layer_bias_scales = self.bias_scales[layer_index]

            # Check if the layer has scale factor values
            if layer_weight_scales is None and layer_bias_scales is None:
                return tf.constant(0.0, dtype=tf.float32)

            layer_weights = self.weights[layer_index]
            layer_biases = self.biases[layer_index]

            max_w_per_row = tf.reduce_max(tf.abs(layer_weights / layer_weight_scales), axis=self.application_of_scale_factors)            
            max_b = tf.reduce_max(tf.abs(layer_biases / layer_bias_scales))

            bins_w = max_w_per_row
            bins_b = max_b

            # Log the floored mean values of bins
            tf.print(tf.floor(tf.reduce_mean(bins_w)), output_stream=f'file://{self.log_dir}/bins_w.log')
            tf.print(tf.floor(tf.reduce_mean(bins_b)), output_stream=f'file://{self.log_dir}/bins_b.log')

            dim_w = tf.cast(bins_w.shape[0], dtype=tf.float32)
            dim_b = tf.constant(1.0, dtype=tf.float32)

            average_bins = (tf.reduce_mean(bins_w) * dim_w + tf.reduce_mean(bins_b) * dim_b) / (dim_w + dim_b)
            tf.print(tf.floor(average_bins), output_stream=f'file://{self.log_dir}/bins_average.log')

            penalty_contribution, scale_num_contribution = tf.cond(
                tf.reduce_mean(average_bins) < 1,
                lambda: (tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)),  # Zero penalty and no contribution to scale_num
                lambda: (average_bins, dim_b + dim_w)  # Regular penalty and scale_num update
            )

            scale_penalty += penalty_contribution
            scale_num += scale_num_contribution

        scale_penalty = tf.cond(
            scale_num > 0,
            lambda: scale_penalty / scale_num,
            lambda: tf.constant(0.0, dtype=tf.float32)
        )
        scale_penalty *= self.penalty_rate

        tf.print(tf.reduce_mean(scale_penalty), output_stream=f'file://{self.log_dir}/scale_loss_log.log')

        return scale_penalty

    def get_name(self):
        return "SCCEDifference"