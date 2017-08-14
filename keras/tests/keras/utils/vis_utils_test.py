import os
import sys

import pytest

from keras import Conv2D
from keras import Dense
from keras import Flatten
from keras import LSTM
from keras import Sequential
from keras import TimeDistributed
from keras import vis_utils


@pytest.mark.skipif(sys.version_info > (3, 0), reason='pydot-ng currently supports python 3.4')
def test_plot_model():
    model = Sequential()
    model.add(Conv2D(filters=2, kernel_size=(2, 3), input_shape=(3, 5, 5), name='conv'))
    model.add(Flatten(name='flat'))
    model.add(Dense(5, name='dense1'))
    vis_utils.plot_model(model, to_file='model1.png', show_layer_names=False)
    os.remove('model1.png')

    model = Sequential()
    model.add(LSTM(16, return_sequences=True, input_shape=(2, 3), name='lstm'))
    model.add(TimeDistributed(Dense(5, name='dense2')))
    vis_utils.plot_model(model, to_file='model2.png', show_shapes=True)
    os.remove('model2.png')


if __name__ == '__main__':
    pytest.main([__file__])
