from keras.datasets import fashion_mnist
import numpy as np

def load_fashion_mnist(return_images=False):

  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

  train_shuffler = np.random.shuffle(np.arange(50000))
  x_train, y_train = x_train[train_shuffler][0], y_train[train_shuffler][0]

  test_shuffler = np.random.shuffle(np.arange(10000))
  x_test, y_test = x_test[test_shuffler][0], y_test[test_shuffler][0]

  x_train = np.array(x_train/255).astype('float32')
  x_test = np.array(x_test/255).astype('float32')

  x_train, x_val = x_train[:54000], x_train[54000:]
  y_train, y_val = y_train[:54000], y_train[54000:]


  if (return_images==False):
    return {
        'train': {
            'X': x_train.reshape([54000, 784]),
            'Y': y_train.reshape([54000])
        },
        'val': {
            'X': x_val.reshape([6000, 784]),
            'Y': y_val.reshape([6000])
        },
        'test': {
            'X': x_test.reshape([10000, 784]),
            'Y': y_test.reshape([10000])
        }
  }

  else :
    return {
      'train': {
          	'X': x_train,
          	'Y': y_train
      },
      'val': {
            'X': x_val,
            'Y': y_val
      },
      'test': {
            'X': x_test,
            'Y': y_test
      }
    }
