'''
Server program

Receive data and recognize it using models
'''
import pickle
import socketserver
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

warnings.filterwarnings("ignore")

HOST = 'localhost'
PORT = 1234

DEFAULT_GRAPH = tf.get_default_graph()

class MyTcpHandler(socketserver.StreamRequestHandler):
    '''
    Server handler for handle requests
    '''
    def handle(self):
        global DEFAULT_GRAPH
        print('[%s] is connected' %self.client_address[0])
        data = self.request.recv(1024)
        nparr = pd.read_json(data.decode()).as_matrix()
        score = 0
        with DEFAULT_GRAPH.as_default():
            for model, name in zip(self.server.model, self.server.model_names):
                answer = model.predict(nparr)
                if name == 'Autoencoded Deep Learning':                        #keras deep learning
                    mse = np.mean(np.power(nparr - answer, 2))
                    if mse > 5:
                        score = score + 1

                elif name == 'Random forest':
                    if answer[0] == 0:
                        score = score + 1

                if score >= self.server.pass_score:
                    break
            if score >= self.server.pass_score:
                self.request.send(bytes(str('0,%d' % score), 'utf8'))
            else:
                self.request.send(bytes(str('1,%d' % score), 'utf8'))


class ThreadedServer(socketserver.ThreadingTCPServer):
    '''
    Threaded server class
    '''
    def __init__(self, listen_addr):
        socketserver.ThreadingTCPServer.__init__(self, listen_addr, MyTcpHandler)
        self.model = []
        self.model_names = []
        self.pass_score = 0

    def start(self):
        '''
        Start server and serve forever.
        '''
        print("------- Server start -------")
        try:
            self.serve_forever()
        except:
            print("-------- Server end ---------")


def run_server(listen_addr, model_paths, model_names, pass_score=1):
    '''
    Run server using path to model(s)

    :parameter
        model_paths(array): array of path to model(s)
        model_names(array): array of name of model(s)
        pass_score(int): threshold for multi model input.
                        Server recognize it is normal when model passes >= pass_score
    '''
    server = ThreadedServer(listen_addr)
    for path in model_paths:
        if path[-3:] == '.h5':    # if it's keras model, use keras-load_model
            server.model.append(load_model(path))
        else:                     # else it's pickle dumped model.
            server.model.append(pickle.load(open(path, 'rb')))

    server.pass_score = pass_score
    server.model_names = model_names
    server.start()
    return server

if __name__ == '__main__':
    run_server((HOST, PORT),
               ['./CCFD/models/model1.sav', './CCFD/models/fraud_dl.h5'],
               ['Random forest', 'Autoencoded Deep Learning'], 2)
