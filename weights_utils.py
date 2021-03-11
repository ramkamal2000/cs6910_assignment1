import numpy as np
import tensorflow as tf

def wandb_initializer(nn_shape, weights_list, biases_list, type='random', mu = 0, sigma = 1):
  
  # random initialization
  if (type=='random'):
    initializer = tf.keras.initializers.TruncatedNormal(mean=mu, stddev=sigma)
  # xavier initialization
  else if (type=='xavier'):
    initializer = tf.keras.initializers.GlorotNormal()

  weights_list.append(initializer(shape=(nn_shape['hidden_layer_size'], nn_shape['input_layer_size'])).numpy())
  biases_list.append(initializer(shape=(nn_shape['hidden_layer_size'], 1)).numpy())
  for i in range(nn_shape['num_hidden_layers'] - 1):
    weights_list.append(initializer(shape=(nn_shape['hidden_layer_size'], nn_shape['hidden_layer_size'])).numpy())
    biases_list.append(initializer(shape=(nn_shape['hidden_layer_size'], 1)).numpy())

  weights_list.append(initializer(shape=(nn_shape['output_layer_size'], nn_shape['hidden_layer_size'])).numpy())
  biases_list.append(initializer(shape=(nn_shape['output_layer_size'], 1)).numpy())

  return weights_list, biases_list
'''
  @staticmethod
  def random_uni(nn_shape, weights_list, biases_list, min=-1, max=1):
    
    weights_list.append(np.random.uniform(min, max, (nn_shape['hidden_layer_size'], nn_shape['input_layer_size'])))
    biases_list.append(np.random.uniform(min, max, (nn_shape['hidden_layer_size'], 1)))

    for i in range(nn_shape['num_hidden_layers'] - 1):
      weights_list.append(np.random.uniform(min, max, (nn_shape['hidden_layer_size'], nn_shape['hidden_layer_size'])))
      biases_list.append(np.random.uniform(min, max, (nn_shape['hidden_layer_size'], 1)))

    weights_list.append(np.random.uniform(min, max, (nn_shape['output_layer_size'], nn_shape['hidden_layer_size'])))
    biases_list.append(np.random.uniform(min, max, (nn_shape['output_layer_size'], 1)))

    return weights_list, biases_list

  @staticmethod
  def xavier(nn_shape, weights_list, biases_list):
    pass
'''