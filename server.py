from __future__ import print_function
import sys
import os
import numpy as np
import scipy
from scipy.io import loadmat
import pickle
from sklearn.decomposition import PCA
import socket
import threading 

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

eprint('Loading preprocessor...')
with np.load('MISC/pcas_trans.npz') as pcat:
    pcas_trans = pcat['arr_0']
with np.load('MISC/pcas_means.npz') as pcam:
    pcas_means = pcam['arr_0']
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
    result = np.empty((1,pcas_trans.shape[0],pcas_trans.shape[1]))
    for i in range(pcas_trans.shape[0]):
        pca_t = pcas_trans[i]
        pca_m = pcas_means[i]
        result[:,i,:] = np.transpose(np.matmul(pca_t, np.expand_dims(data[i]-pca_m, -1)), (1,0)) 
    return result

predicted_label = 0 # Hey! You can get label here!

class StoppableThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), daemon=True):
        super(StoppableThread, self).__init__(group=group,target=target,name=name,args=args)
        self.daemon=daemon
        self._stop_event = threading.Event()
    def __enter__(self):
        self.start()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        if not self.stopped():
            self.stop()
    def stop(self):
        self._stop_event.set()
    def stopped(self):
        return self._stop_event.is_set()

def prediction():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    import keras
    from keras.layers import CuDNNLSTM
    from keras.models import Sequential, load_model, Model
    global predicted_label
    eprint('Loading LSTM model...')
    model = load_model('MISC/model.h5')
    eprint('Loaded!')
    while True:
        test_rand = np.random.randint(220, data.shape[0])
        current = pcas_transform(data[(test_rand-220):test_rand])
        #current = pcas_transform(data[-220:])
        predicted_label = np.argmax(model.predict(current, batch_size=1, verbose=0)[0])

try:
    with StoppableThread(target=prediction, daemon=True) as threaded_prediction:
        while True:
            conn, addr = s.accept()
            conn.settimeout(5)
            eprint('From: %s'%str(addr))
            try:
                try:
                    while True: ## one to one
                        msg = conn.recv(64)
                        msg = msg.decode('utf-8')
                        eprint('Received request: %s'%msg)
                        if msg[:4]=='exit':
                            break
                        elif msg[:5]=='label': 
                            label = predicted_label # get label from thread
                            ans = RESPONSE[label%3]
                            conn.send(ans.encode('utf-8'))
                        else:
                            conn.send('invalid query, closing connection...'.encode('utf-8'))
                            break
                except socket.timeout:
                    eprint('Time out')
            except:
                eprint('Unexpected ERROR occur! Lost connection to %s'%str(addr))
            conn.close()
except KeyboardInterrupt:
    eprint('Bye!')
s.close()
