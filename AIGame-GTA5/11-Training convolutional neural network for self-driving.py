# train_model.py

import numpy as np
from alexnet import alexnet
from random import shuffle

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 5
MODEL_NAME = '/Users/w/project/tmp/python/AIGame-GTA5/11-train_model/pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

hm_data = 1
for i in range(EPOCHS):
    for i in range(1,hm_data+1):
        # train_data = np.load('/Users/w/project/tmp/python/AIGame-GTA5/train/training_data-{}-balanced.npy'.format(i))
        #train_data = np.load('/Users/w/project/tmp/python/AIGame-GTA5/train/training_data_balanced.npy')
        train_data = np.load('/Users/w/project/tmp/python/AIGame-GTA5/train/training_data.npy')
        shuffle(train_data)
        train = train_data[:-1000]
        test = train_data[-1000:]

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)

'''
---------------------------------
Run id: 11-train_model/pygta5-car-fast-0.001-alexnetv2-10-epochs-300K-data.model
Log directory: log/
---------------------------------
Training samples: 1400
Validation samples: 100
--
Training Step: 177  | total loss: 0.62741 | time: 1.584s
| Momentum | epoch: 009 | loss: 0.62741 - acc: 0.7331 -- iter: 0064/1400
Training Step: 178  | total loss: 0.61127 | time: 3.026s
| Momentum | epoch: 009 | loss: 0.61127 - acc: 0.7348 -- iter: 0128/1400
Training Step: 179  | total loss: 0.59662 | time: 4.453s
| Momentum | epoch: 009 | loss: 0.59662 - acc: 0.7426 -- iter: 0192/1400
Training Step: 180  | total loss: 0.58382 | time: 5.875s
| Momentum | epoch: 009 | loss: 0.58382 - acc: 0.7543 -- iter: 0256/1400
Training Step: 181  | total loss: 0.58382 | time: 7.291s
| Momentum | epoch: 009 | loss: 0.58382 - acc: 0.7663 -- iter: 0320/1400
Training Step: 182  | total loss: 0.55676 | time: 8.675s
| Momentum | epoch: 009 | loss: 0.55676 - acc: 0.7663 -- iter: 0384/1400
Training Step: 183  | total loss: 0.55228 | time: 10.084s
| Momentum | epoch: 009 | loss: 0.55228 - acc: 0.7646 -- iter: 0448/1400
Training Step: 184  | total loss: 0.54140 | time: 11.484s
| Momentum | epoch: 009 | loss: 0.54140 - acc: 0.7679 -- iter: 0512/1400
Training Step: 185  | total loss: 0.53824 | time: 12.885s
| Momentum | epoch: 009 | loss: 0.53824 - acc: 0.7723 -- iter: 0576/1400
Training Step: 186  | total loss: 0.53473 | time: 14.345s
| Momentum | epoch: 009 | loss: 0.53473 - acc: 0.7717 -- iter: 0640/1400
Training Step: 187  | total loss: 0.52158 | time: 15.849s
| Momentum | epoch: 009 | loss: 0.52158 - acc: 0.7757 -- iter: 0704/1400
Training Step: 188  | total loss: 0.52341 | time: 17.321s
| Momentum | epoch: 009 | loss: 0.52341 - acc: 0.7747 -- iter: 0768/1400
Training Step: 189  | total loss: 0.72105 | time: 18.816s
| Momentum | epoch: 009 | loss: 0.72105 - acc: 0.7254 -- iter: 0832/1400
Training Step: 190  | total loss: 0.70928 | time: 20.322s
| Momentum | epoch: 009 | loss: 0.70928 - acc: 0.7278 -- iter: 0896/1400
Training Step: 191  | total loss: 0.69501 | time: 21.845s
| Momentum | epoch: 009 | loss: 0.69501 - acc: 0.7332 -- iter: 0960/1400
Training Step: 192  | total loss: 0.66801 | time: 23.392s
| Momentum | epoch: 009 | loss: 0.66801 - acc: 0.7442 -- iter: 1024/1400
Training Step: 193  | total loss: 0.64797 | time: 24.954s
| Momentum | epoch: 009 | loss: 0.64797 - acc: 0.7511 -- iter: 1088/1400
Training Step: 194  | total loss: 0.64843 | time: 26.374s
| Momentum | epoch: 009 | loss: 0.64843 - acc: 0.7431 -- iter: 1152/1400
Training Step: 195  | total loss: 0.63881 | time: 27.789s
| Momentum | epoch: 009 | loss: 0.63881 - acc: 0.7407 -- iter: 1216/1400
Training Step: 196  | total loss: 0.63294 | time: 29.236s
| Momentum | epoch: 009 | loss: 0.63294 - acc: 0.7432 -- iter: 1280/1400
Training Step: 197  | total loss: 0.61339 | time: 30.637s
| Momentum | epoch: 009 | loss: 0.61339 - acc: 0.7454 -- iter: 1344/1400
Training Step: 198  | total loss: 0.60878 | time: 33.172s
| Momentum | epoch: 009 | loss: 0.60878 - acc: 0.7530 | val_loss: 0.30045 - val_acc: 0.8400 -- iter: 1400/1400
--
---------------------------------
Run id: 11-train_model/pygta5-car-fast-0.001-alexnetv2-10-epochs-300K-data.model
Log directory: log/
---------------------------------
Training samples: 1400
Validation samples: 100
--
Training Step: 199  | total loss: 0.59524 | time: 1.581s
| Momentum | epoch: 010 | loss: 0.59524 - acc: 0.7605 -- iter: 0064/1400
Training Step: 200  | total loss: 0.59117 | time: 3.078s
| Momentum | epoch: 010 | loss: 0.59117 - acc: 0.7626 -- iter: 0128/1400
Training Step: 201  | total loss: 0.57423 | time: 4.520s
| Momentum | epoch: 010 | loss: 0.57423 - acc: 0.7707 -- iter: 0192/1400
Training Step: 202  | total loss: 0.56196 | time: 5.997s
| Momentum | epoch: 010 | loss: 0.56196 - acc: 0.7718 -- iter: 0256/1400
Training Step: 203  | total loss: 0.54018 | time: 7.427s
| Momentum | epoch: 010 | loss: 0.54018 - acc: 0.7821 -- iter: 0320/1400
Training Step: 204  | total loss: 0.52395 | time: 8.852s
| Momentum | epoch: 010 | loss: 0.52395 - acc: 0.7836 -- iter: 0384/1400
Training Step: 205  | total loss: 0.52395 | time: 10.280s
| Momentum | epoch: 010 | loss: 0.52395 - acc: 0.7836 -- iter: 0448/1400
Training Step: 206  | total loss: 0.50856 | time: 11.752s
| Momentum | epoch: 010 | loss: 0.50856 - acc: 0.7905 -- iter: 0512/1400
Training Step: 207  | total loss: 0.49636 | time: 13.344s
| Momentum | epoch: 010 | loss: 0.49636 - acc: 0.7974 -- iter: 0576/1400
Training Step: 208  | total loss: 0.49636 | time: 14.746s
| Momentum | epoch: 010 | loss: 0.49636 - acc: 0.7974 -- iter: 0640/1400
Training Step: 209  | total loss: 0.47308 | time: 16.230s
| Momentum | epoch: 010 | loss: 0.47308 - acc: 0.8048 -- iter: 0704/1400
Training Step: 210  | total loss: 0.47308 | time: 17.741s
| Momentum | epoch: 010 | loss: 0.47308 - acc: 0.8048 -- iter: 0768/1400
Training Step: 211  | total loss: 0.66814 | time: 19.170s
| Momentum | epoch: 010 | loss: 0.66814 - acc: 0.7631 -- iter: 0832/1400
Training Step: 212  | total loss: 0.63400 | time: 20.611s
| Momentum | epoch: 010 | loss: 0.63400 - acc: 0.7759 -- iter: 0896/1400
Training Step: 213  | total loss: 0.62142 | time: 22.097s
| Momentum | epoch: 010 | loss: 0.62142 - acc: 0.7780 -- iter: 0960/1400
Training Step: 214  | total loss: 0.60014 | time: 23.607s
| Momentum | epoch: 010 | loss: 0.60014 - acc: 0.7877 -- iter: 1024/1400
Training Step: 215  | total loss: 0.59445 | time: 25.015s
| Momentum | epoch: 010 | loss: 0.59445 - acc: 0.7855 -- iter: 1088/1400
Training Step: 216  | total loss: 0.57596 | time: 26.477s
| Momentum | epoch: 010 | loss: 0.57596 - acc: 0.7882 -- iter: 1152/1400
Training Step: 217  | total loss: 0.57596 | time: 28.049s
| Momentum | epoch: 010 | loss: 0.57596 - acc: 0.7882 -- iter: 1216/1400
Training Step: 218  | total loss: 0.55483 | time: 29.500s
| Momentum | epoch: 010 | loss: 0.55483 - acc: 0.7914 -- iter: 1280/1400
Training Step: 219  | total loss: 0.53406 | time: 30.973s
| Momentum | epoch: 010 | loss: 0.53406 - acc: 0.7966 -- iter: 1344/1400
Training Step: 220  | total loss: 0.51625 | time: 33.475s
| Momentum | epoch: 010 | loss: 0.51625 - acc: 0.8045 | val_loss: 0.20279 - val_acc: 0.9100 -- iter: 1400/1400
--

Process finished with exit code 0

'''

# tensorboard --logdir=./





