import pymongo
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from matplotlib.animation import FFMpegWriter
from Prediction import *

connection = pymongo.MongoClient('localhost', 27017)
db = connection.Forex   #database

data = db.USD_XAU   #collection
Pred_data = db.Prediction
pred = db['Pred']

length=60
pred_length=15

init = data.find().skip(data.count() - length)
Close_init = []

for item in init:
    Close_init.append(item['Close'])

Time = np.arange(length)
Close_init=np.asarray(Close_init).astype(float)

closed_rate = np.empty((length, 1))
predicted_rate = np.empty((length, 1))

closed_rate[:, 0] = Close_init[0:]
predicted_rate[:, 0] = Close_init[0:]

fig, ax = plt.subplots()

Close_plot_pred, = ax.plot(Time, predicted_rate, label='Predicted')
Close_plot, = ax.plot(Time, closed_rate, label='Actual')
ax.set_ylim(np.nanmin(closed_rate)-0.1, np.nanmax(closed_rate)+0.1)


def init():
    Close_plot_pred.set_ydata(predicted_rate)
    Close_plot.set_ydata(closed_rate)
    return Close_plot, Close_plot_pred,

with open('data/test.csv') as csvFile:
    csvReader = csv.DictReader(csvFile)

    def animate(i):
        for rows in csvReader:
            test = {}
            id = str(rows['Date'] + rows['Time'])
            test[id] = rows
            print(rows)
            data.insert_one(test[id])   #insert new data

            Close = []
            sub = data.find().skip(data.count() - (length-pred_length))
            for item in sub:
                Close.append(item['Close'])

            print(Close)
            Close_array = np.asarray(Close).astype(float)

            closed_rate[:len(Close_array), 0] = Close_array[0:]
            closed_rate[len(Close_array):, 0] = np.nan

            pred = Prediction(Close_array, number_head=pred_length)

            predicted_rate[(len(predicted_rate)-len(pred)):, 0] = pred[0:, 0]
            predicted_rate[:(len(predicted_rate) - len(pred)), 0] = np.nan

            Close_plot.set_ydata(np.expand_dims(closed_rate, axis=1))  # update the data.

            ax.set_ylim(np.nanmin(closed_rate)-1, np.nanmax(closed_rate)+1)
            Close_plot_pred.set_ydata(predicted_rate)  # update the data.
            legend = plt.legend()



            return Close_plot, Close_plot_pred, legend,


    ani = animation.FuncAnimation(
        fig, animate, init_func=init, interval=2, blit=True, save_count=120)

    plt.rcParams['animation.ffmpeg_path'] = 'C:/ffmpeg/bin/ffmpeg.exe'
    FFwriter = animation.FFMpegWriter(fps=10, extra_args=['-vcodec', 'libx264'])
    ani.save('movie_120.mp4', writer=FFwriter)








