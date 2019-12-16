from DL_model import *
import pickle
import pandas as pd

def prepare_timeseries(Y, time_step):
    dataX = []
    length = len(Y) - time_step + 1
    for i in range(0, length):
        dataX.append(Y[i:(i + time_step)])
    return np.array(dataX)

def Prediction(test, number_head = 15):

    model =  load_model('model/model_Close.h5')
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))

    test_norm = scaler.transform(np.expand_dims(np.asarray(test), axis=1))

    testX = prepare_timeseries(test_norm, 15)
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1] * testX.shape[2]))

    pred_y = model.predict(testX)

    for i in range(number_head):
        test_norm = np.expand_dims(np.append(test_norm, pred_y[-1]), axis=1)
        testX = prepare_timeseries(test_norm, 15)
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1] * testX.shape[2]))
        pred_y = model.predict(testX)

    pred_y = scaler.inverse_transform(pred_y)

    return pred_y


