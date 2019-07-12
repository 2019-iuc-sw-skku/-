import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from keras.backend import argmax

LABELS = ["Normal","Fraud"]
RANDOM_SEED = 1398
TEST_PCT = 0.2

df = pd.read_csv("./CCFD/creditcard.csv")
normal_df = df[df.Class == 0]
fraud_df = df[df.Class == 1]

df_norm = df
df_norm['Time'] = StandardScaler().fit_transform(df_norm['Time'].values.reshape(-1, 1))
df_norm['Amount'] = StandardScaler().fit_transform(df_norm['Amount'].values.reshape(-1, 1))

train_x, test_x = train_test_split(df_norm, test_size=TEST_PCT, random_state=RANDOM_SEED)
train_x = train_x[train_x.Class == 0]
train_x = train_x.drop(['Class'], axis=1)

test_y = test_x['Class']
test_x = test_x.drop(['Class'], axis=1)

train_x = train_x.values
test_x = test_x.values

input_dimension = train_x.shape[1]
nb_epoch = 100
batch_size = 128
encoding_dimension = 32
learning_rate = 0.0000001

input_layer = Input(shape=(input_dimension, ))

Encoder1 = Dense(encoding_dimension, activation="tanh", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
Encoder2 = Dense(int(encoding_dimension/2), activation="relu")(Encoder1)
Encoder3 = Dense(int(encoding_dimension/4), activation="tanh")(Encoder2)
Decoder1 = Dense(int(encoding_dimension/4), activation="relu")(Encoder3)
Decoder2 = Dense(int(encoding_dimension/2), activation="tanh")(Decoder1)
Decoder3 = Dense(input_dimension, activation="relu")(Decoder2)

AutoEncoderModel = Model(inputs=input_layer, outputs=Decoder3)

AutoEncoderModel.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')

cp = ModelCheckpoint(filepath="./CCFD/models/fraud_dl.h5", save_best_only=True)

tb = TensorBoard(log_dir='./dllogs', write_graph=True, write_images=True)

history = AutoEncoderModel.fit(train_x, train_x, epochs=nb_epoch, batch_size=batch_size, shuffle=True, validation_data=(test_x, test_x), verbose=1, callbacks=[cp, tb]).history

threshold_fixed = 5

test_x_predictions = AutoEncoderModel.predict(test_x)

mse = np.mean(np.power(test_x - test_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': test_y})

'''
#presicion, recall, f1-score 계산
precision = precision_score(test_y, label_predictions)
print('Precision: %f' % precision)

recall = recall_score(test_y, label_predictions)
print('Recall: %f' % recall)

f1 = f1_score(test_y, label_predictions)
print('F1 score: %f' % f1)
'''
#report 출력을 위한 reduce
label_predictions=np.argmax(np.round(test_x_predictions), axis=1)

report=classification_report(test_y, label_predictions)
print(report)

pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.True_class, pred_y)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
