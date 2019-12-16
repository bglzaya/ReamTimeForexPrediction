import pandas as pd
from DL_model import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
import math
import matplotlib.pyplot as plt
import pickle

def evaluation(true_Y, pred_Y):
    rmse = math.sqrt(mean_squared_error(true_Y, pred_Y))
    mae = (mean_absolute_error(true_Y, pred_Y))
    smape = np.mean(np.abs((true_Y - pred_Y) / (true_Y + pred_Y)))
    r_squared = r2_score(true_Y, pred_Y)

    return rmse, mae, smape, r_squared


data = pd.read_csv('data/XAU_USD.csv')
test = pd.read_csv('data/test.csv')
method = 'mlp'
var='Close'

performance = {'Data': [], 'Method': [],
               'RMSE': [], 'MAE': [],
               'sMAPE': [], 'R_squared': []}


train_set = data[var]
test_set = test[var]

scaler = MinMaxScaler()
scaler_scaling = MinMaxScaler().fit(np.expand_dims(np.asarray(train_set), axis=1))
training = scaler_scaling.transform(np.expand_dims(np.asarray(train_set), axis=1))
test = scaler_scaling.transform(np.expand_dims(np.asarray(test_set), axis=1))

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler_scaling, f)

data_preprocess= data_preparing(training, training, test, test, time_steps=15)
trainX, valX, trainY, valY, testX, testY = data_preprocess.data_preprocessing()

modelling = mlp_models(trainX, valX, trainY, valY,
         mlp_layers=[16, 1], time_steps=15, batch_size=512,
         early_stopping=100, model_path='model/model_%s' %(var),
        method=method, var=var, epoch=3000, lr=0.01, reg_lambda= 0.00001)

final_model = modelling.proposed_model()

testX = np.reshape(testX, (testX.shape[0], testX.shape[1] * testX.shape[2]))

pred_y = final_model.predict(testX)

testY = scaler_scaling.inverse_transform(testY)
pred_y = scaler_scaling.inverse_transform(pred_y)

plt.plot(testY)
plt.plot(pred_y)
plt.show()

rmse, mae, smape, r_squared = evaluation(true_Y=testY, pred_Y=pred_y)

performance['Data'].append(var)
performance['Method'].append('Proposed_model')
performance['RMSE'].append(rmse)
performance['MAE'].append(mae)
performance['sMAPE'].append(smape)
performance['R_squared'].append(r_squared)


res = pd.DataFrame.from_dict(performance)
res.to_csv('model/performance.csv', index=False, mode='a')








