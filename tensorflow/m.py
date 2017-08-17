import numpy as np

print(np.linspace(-1, 1, 300).shape)
print(np.linspace(-1, 1, 300)[np.newaxis].shape) #(1, 300)
print(np.linspace(-1, 1, 300)[:, np.newaxis])
print(np.linspace(-1, 1, 300)[:, np.newaxis][:, np.newaxis].shape) #(300, 1, 1)
# [[[-1.        ]]
#  [[-0.99331104]]
#  [[-0.98662207]]
print(np.linspace(-1, 1, 300)[:, np.newaxis][np.newaxis].shape) #(1, 300, 1)
# [[[-1.        ]]
#  [[-0.99331104]]
#  [[-0.98662207]]
