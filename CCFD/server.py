'''
Server program

Receive data and recognize it using models
'''
import pickle
import socketserver
import threading
import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from model_names import ModelNames

warnings.filterwarnings("ignore")

HOST = 'localhost'
PORT = 1234

class MyTcpHandler(socketserver.StreamRequestHandler):
    '''
    Server handler for handle requests
    '''
    def handle(self):
        global DEFAULT_GRAPH
        now = time.localtime()
        connected_time = '%04d-%02d-%02d %02d:%02d:%02d' % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
        print(connected_time, end='')
        print('[%s] is connected' %self.client_address[0])
        data = self.request.recv(1024)
        nparr = pd.read_json(data.decode()).as_matrix()
        score = 0
        flag = []
        with self.server.graph.as_default():
            for model, scaler, name in zip(self.server.model, self.server.scaler, self.server.model_names):
                scaled_arr = scaler.transform(nparr)
                answer = model.predict(scaled_arr)
                if name == ModelNames.AUTOENCODED_DEEP_LEARNING:                        #keras deep learning
                    mse = np.mean(np.power(scaled_arr - answer, 2))
                    if mse < 5:
                        flag.append(0)
                        score = score + 1
                    else:
                        flag.append(1)
                else:
                    if answer[0] == 0:
                        flag.append(0)
                        score = score + 1
                    else:
                        flag.append(1)
                if score >= self.server.pass_score:
                    break
            print(flag)
            if score >= self.server.pass_score:
                self.request.send(bytes(str('0,%d' % score), 'utf8'))
            else:
                '''
                여기에 팝업창띄우는 코드 삽입예정
                '''
                logfile = open('./fraudlog.txt', 'a')
                logfile.write(connected_time + ' ' + self.client_address[0] + '\n')
                logfile.close()
                self.request.send(bytes(str('1,%d' % score), 'utf8'))


class ThreadedServer(socketserver.ThreadingTCPServer):
    '''
    Threaded server class
    '''
    def __init__(self, listen_addr):
        socketserver.ThreadingTCPServer.__init__(self, listen_addr, MyTcpHandler)
        self.graph = tf.get_default_graph()
        self.model = []
        self.model_names = []
        self.scaler = []
        self.pass_score = 0

    def start(self):
        '''
        Start server and serve forever.
        '''
        print("------- Server start -------")
        self.serve_forever()
        print("-------- Server end ---------")

class StoppableThread(threading.Thread):
    '''
    Thread class with a stop() method.
    The thread will stop for the stop() call
    '''

    def __init__(self, target, args):
        '''
        :parameters
            target: target function
            args: arguments as tuple
        '''
        super(StoppableThread, self).__init__()
        self._stop_event = threading.Event()
        self.server: ThreadedServer
        self.target = target
        self.args = args

    def run(self):
        self.server = self.target(*self.args)
        self.server.start()

    def stop(self):
        self._stop_event.set()
        if hasattr(self, 'server'):
            self.server.shutdown()
            self.server.server_close()
        K.clear_session()

    def stopped(self):
        return self._stop_event.is_set()
def __set_server(listen_addr, model_paths, scaler_paths, model_names, pass_score):
    '''
    Run server using path to model(s)

    :parameter
        model_paths(array): array of path to model(s).
        model_names(array): array of enumerated value of name of model(s). see model_names.py
                example:
                    RandomForest, DeepLearning -> [ModelNames(0), ModelNames(1)]
        pass_score(int): threshold for multi model input.
                         Server recognize it is normal when model passes >= pass_score
    '''
    server = ThreadedServer(listen_addr)
    for model_path, scaler_path, model_name in zip(model_paths, scaler_paths, model_names):
        if model_name == ModelNames.AUTOENCODED_DEEP_LEARNING:    # if it's keras model, use keras-load_model
            server.model.append(load_model(model_path))
            server.model[-1]._make_predict_function()
        else:                     # else it's pickle dumped model.
            server.model.append(pickle.load(open(model_path, 'rb')))
        
        server.scaler.append(pickle.load(open(scaler_path, 'rb')))

    server.pass_score = pass_score
    server.model_names = model_names
    return server

def run_server(listen_addr, model_paths, scaler_paths, model_names, pass_score=1):
    thread = StoppableThread(target=__set_server, args=(listen_addr, model_paths, scaler_paths, model_names, pass_score))
    thread.start()
    return thread

if __name__ == '__main__':
    run_server((HOST, PORT),
               ['./CCFD/models/model_rf1.sav', './CCFD/models/model_svm_rbf1.sav', './CCFD/models/fraud_dl.h5'],
               ['./CCFD/scalers/scaler_rf1.sav', './CCFD/scalers/scaler_svm_rbf1.sav', './CCFD/scalers/scaler_dl.sav'],
               [ModelNames.RANDOM_FOREST, ModelNames.SVM, ModelNames.AUTOENCODED_DEEP_LEARNING], 3)
