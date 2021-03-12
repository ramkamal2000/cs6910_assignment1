import numpy as np
from weights_utils import wandb_initializer

class neural_network:

  # constructor function - initializes weights
  def __init__(self, dict_layers, initializer):

    self.weights_list = []
    self.biases_list = []
    self.dict_layers= dict_layers

    self.weights_list, self.biases_list = wandb_initializer(dict_layers, self.weights_list, self.biases_list, initializer)

  # function to compute forward propogation
  def forward_prop(self, W, b, X, Y, activation_func):

    A = []
    H = []
    
    H_pre = X
    
    L = self.dict_layers['num_hidden_layers']
    for i in range(L):
      
      A.append(W[i] @ H_pre + b[i])
      H_pre = getattr(activation, activation_func)(A[i])
      H.append(H_pre)
    
    A.append(W[L] @ H_pre + b[L])
    
    Y_hat = activation.softmax(A[L])
    
    return {
        'A' : A,
        'H' : H,
        'Y_hat' : Y_hat
    }

  def self_forward_prop(self, X, Y, activation_func) :

    temp = self.forward_prop(self.weights_list,self.biases_list, X, Y, activation_func)
    return temp

  def back_prop(self, W, b, A, H, Y_hat, X, Y,activation_func):

    batch_size = len(Y)
    
    del_w = [] 
    del_b = []

    L = self.dict_layers['num_hidden_layers']
    
    E = np.zeros(Y_hat.shape)
    
    # E[np.arange(Y.size), Y] = 1
    for j in range(len(Y)):
        E[int(Y[j])][j] = 1
    
    # what shape do you need y_hat and e to be in? Column or row vector?
    grad_A = -(E - Y_hat)
    #print('grad_a', grad_a.shape)

    for i in range(L,-1,-1) :

      temp1 = grad_A.reshape(-1,batch_size)
      if i==0 :
        temp2 = X.reshape((batch_size ,-1))
      else :
        temp2 = H[i-1].reshape((batch_size ,-1))
      del_w.append(temp1 @ temp2)
      del_b.append(grad_A)

      if(i!=0) :
        grad_H = W[i].T @ grad_A      
        act = activation()
        grad_A = grad_H * getattr(act,activation_func+'_der')(H[i-1])

    return {
        'dw' : del_w,
        'db' : del_b
    }

  def self_back_prop(self, A, H, Y_hat, X, Y,activation_func) :
    temp = self.back_prop(self.weights_list,self.biases_list, A, H, Y_hat, X, Y, activation_func)
    return temp

  def grad_wandb(self, W, b, X, Y,activation_func):
    
    X = X.T.reshape((784,-1))
    Y = Y.reshape((-1, 1))
    
    temp = self.forward_prop(W, b, X, Y, activation_func)
    temp2 = self.back_prop(W, b, temp['A'], temp['H'], temp['Y_hat'], X, Y, activation_func)

    return {
        'dw' : temp2['dw'],
        'db' : temp2['db']
    }

  def self_grad_wandb(self, X, Y, activation_func) :
    temp = self.grad_wandb(self.weights_list, self.biases_list, X, Y,activation_func)
    return temp

  def predict(self, X, activation_func):
    temp = self.forward_prop(self.weights_list,self.biases_list, X, 0, activation_func)
    return {
      'y' : np.argmax(temp['y_hat']),
      'y_hat' : temp['y_hat']
    }

  def update_vals(self,dw,db) :
    L = len(self.weights_list)
    for i in range(L) :
      self.weights_list[i] =self.weights_list[i] - dw[L-i-1]

    for i in range(len(self.biases_list)) :
      self.biases_list[i] =self.biases_list[i] - db[L-i-1]
##################################################################################
class activation:
  
  @staticmethod
  def sigmoid(z):
    z = np.array(z,dtype=np.longdouble)
    return 1 / (1 + np.exp(-z))
  
  @staticmethod
  def relu(z):
    return (z>0) * z

  @staticmethod
  def tanh(z):
    return np.tanh(z)

  @staticmethod
  def sigmoid_der(z) :
    return z * (1-z)
  
  @staticmethod
  def relu_der(z) :
    return (z>0)

  @staticmethod
  def tanh_der(z):
    return 1 - z*z

  @staticmethod
  def softmax(x):
    x = np.array(x,dtype=np.longdouble)
    e_x = np.exp(x - np.max(x)+200)
    return e_x / e_x.sum()

def set_nn_shape(verbose=True, num_hidden_layers=-1, hidden_layer_size=-1):

  input_layer_size = 784
  hidden_layer_size = hidden_layer_size
  num_hidden_layers = num_hidden_layers
  output_layer_size = 10

  if (verbose):
    print("\nNumber Of Hidden Layers:")
    num_hidden_layers = int(input())

    print("\nSize Of Each Hidden Layer:")
    hidden_layer_size = int(input())

    print(f"\nThe Neural Network Has {num_hidden_layers+2} Layers In Total!")
  
  return {"input_layer_size": input_layer_size, "hidden_layer_size": hidden_layer_size, "output_layer_size": output_layer_size, "num_hidden_layers": num_hidden_layers}