import numpy as np

class wandb_initializer:
  @staticmethod
  def random(nn_shape, weights_list, biases_list, mu = 0, sigma = 1):
    
    weights_list.append(np.random.normal(mu, sigma, (nn_shape['hidden_layer_size'], nn_shape['input_layer_size'])))
    biases_list.append(np.random.normal(mu, sigma, (nn_shape['hidden_layer_size'], 1)))

    for i in range(nn_shape['num_hidden_layers'] - 1):
      weights_list.append(np.random.normal(mu, sigma, (nn_shape['hidden_layer_size'], nn_shape['hidden_layer_size'])))
      biases_list.append(np.random.normal(mu, sigma, (nn_shape['hidden_layer_size'], 1)))

    weights_list.append(np.random.normal(mu, sigma, (nn_shape['output_layer_size'], nn_shape['hidden_layer_size'])))
    biases_list.append(np.random.normal(mu, sigma, (nn_shape['output_layer_size'], 1)))

    return weights_list, biases_list

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