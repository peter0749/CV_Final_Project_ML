from __future__ import print_function
import sys
import os
import numpy as np
import scipy
from scipy.io import loadmat
import pickle
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import keras
from keras.layers import CuDNNLSTM
from keras.models import Sequential, load_model, Model
from sklearn.decomposition import PCA
import socket

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

eprint('Loading LSTM model...')
model = load_model('model.h5')
eprint('Loaded!')

eprint('Loading preprocessor...')
with open('pcas.pkl', 'rb') as pcafile:
    pcas = pickle.load(pcafile)
if not pcas:
    raise Exception('Cannot load proprocessor')
eprint('Loaded!')

HOST=''

PORT = 18763
TARGET_N = 10
RESPONSE = ['0bad','1fair','2good']

eprint('Creating socket...')
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST,PORT))
s.listen(TARGET_N)

eprint('Socket opend: %s:%d, maximum users %d'%(HOST,PORT,TARGET_N))

data = np.array(loadmat('dataset.mat')['test_data'])
data = np.reshape(data, (-1, data.shape[-1])) # flatten 0, 1 axis

def pcas_transform(data):
    result = np.empty((1,len(pcas),pcas[0][0].n_components_))
    for i, (pca, pca_sum_var) in enumerate(pcas):
        result[:,i,:] = pca.transform(np.expand_dims(data[i], 0))
    return result

try:
    while True:
        conn, addr = s.accept()
        eprint('From: %s'%str(addr))
        try:
            while True: ## one to one
                msg = conn.recv(64)
                eprint('Received request: %s'%msg)
                if msg=='exit':
                    break
                elif msg=='label': 
                    test_rand = np.random.randint(220, data.shape[0])
                    current = pcas_transform(data[(test_rand-220):test_rand])
                    #current = pcas_transform(data[-220:])
                    label = np.argmax(model.predict(current, batch_size=1, verbose=0)[0])
                    ans = RESPONSE[label%3]
                    conn.send(ans)
                else:
                    conn.send('invalid query, closing connection...')
                    break
        except:
            eprint('Unexpected ERROR occur! Lost connection to %s'%str(addr))
        conn.close()
except KeyboardInterrupt:
    eprint('Bye!')
s.close()

