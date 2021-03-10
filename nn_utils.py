import numpy as np
from weights_utils import wandb_initializer

class neural_network:

  # constructor function - initializes weights
  def __init__(self, dict_layers, initializer):

    self.weights_list = []
    self.biases_list = []
    self.dict_layers= dict_layers

    self.weights_list, self.biases_list = wandb_initializer.random(dict_layers, self.weights_list, self.biases_list)

  # function to compute forward propogation
  def forward_prop(self, W, b, x, y):

    a = []
    h = []
    
    h_pre = np.reshape(x, (-1, 1))

    L = self.dict_layers['num_hidden_layers']
    for i in range(L):
      
      a.append(W[i] @ h_pre + b[i])
      #print(a[i])
      h_pre = activation.sigmoid(a[i])
      h.append(h_pre)
    
    a.append(W[L] @ h_pre + b[L])
    y_hat = activation.softmax(a[L])

    #print(y_hat)
    return {
        'a' : a,
        'h' : h,
        'y_hat' : y_hat
    }

  def self_forward_prop(self,x,y) :

    temp = self.forward_prop(self.weights_list,self.biases_list,x,y)
    return temp

  def back_prop(self, W, b, a, h, y_hat, x, y):

    del_w = [] 
    del_b = []

    L = self.dict_layers['num_hidden_layers']
    e = np.zeros(self.dict_layers['output_layer_size'])
    e[int(y)] = 1
    # what shape do you need y_hat and e to be in? Column or row vector?
    grad_a = -(e.reshape((-1,1))-y_hat.reshape((-1,1)))
    #print('grad_a', grad_a.shape)

    for i in range(L,-1,-1) :

      temp1 = grad_a.reshape(-1,1)
      if i==0 :

        temp2 = x.reshape((1,-1))
      else :

        temp2 = h[i-1].reshape((1,-1))
      del_w.append(temp1 @ temp2)
      del_b.append(grad_a)

      if(i!=0) :

        grad_h = W[i].T @ grad_a
        grad_a = grad_h * activation.sigmoid_der(h[i-1])

    return {
        'dw' : del_w,
        'db' : del_b
    }

  def self_back_prop(self, a, h, y_hat, x, y) :
    temp = self.back_prop(self.weights_list,self.biases_list, a, h, y_hat, x, y)
    return temp

  def grad_wandb(self, W, b, x, y):
    x = x.reshape((-1,1))
    temp = self.forward_prop(W,b,x,y)
    temp2 = self.back_prop(W, b, temp['a'], temp['h'], temp['y_hat'], x, y)

    return {
        'dw' : temp2['dw'],
        'db' : temp2['db']
    }

  def self_grad_wandb(self,x,y) :
    temp = self.grad_wandb(self.weights_list, self.biases_list, x, y)
    return temp

  def predict(self, x):
    temp = self.forward_prop(self.weights_list,self.biases_list,x, 0)
    return np.argmax(temp['y_hat'])

  def update_vals(self,dw,db,w_etas,b_etas) :
    L = len(self.weights_list)
    for i in range(L) :
      self.weights_list[i] =self.weights_list[i] - w_etas[L-i-1]*dw[L-i-1]

    for i in range(len(self.biases_list)) :
      self.biases_list[i] =self.biases_list[i] - b_etas[L-i-1]*db[L-i-1]

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

def set_nn_shape(verbose=True, hidden_layer_size=-1, num_hidden_layers=-1):

  input_layer_size = 784
  hidden_layer_size = -1
  num_hidden_layers = -1
  output_layer_size = 10

  if (verbose):
    print("\nNumber Of Hidden Layers:")
    num_hidden_layers = int(input())

    print("\nSize Of Each Hidden Layer:")
    hidden_layer_size = int(input())

    print(f"\nThe Neural Network Has {num_hidden_layers+2} Layers In Total!")
  
  return {"input_layer_size": input_layer_size, "hidden_layer_size": hidden_layer_size, "output_layer_size": output_layer_size, "num_hidden_layers": num_hidden_layers}