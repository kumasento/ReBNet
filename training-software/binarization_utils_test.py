r"""
Test the binarization utilities defined in binarization_utils.py

"""

import unittest

import numpy as np
from keras.layers import Input
from keras.models import Model

import binarization_utils as bin_utils

class TestBinarizeLayer(unittest.TestCase):
  """ Test BinarizeLayer """

  def test_forward(self):
    x = Input(shape=(32,))
    y = bin_utils.BinarizeLayer()(x)
    model = Model(x, y)

    x_val = np.random.random((1, 32)) * 256 - 128
    y_val = model.predict(x_val)
    print(x_val)
    print(y_val)


if __name__ == '__main__':
  unittest.main()
