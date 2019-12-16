import csv
import pandas as pd
import pymongo

conn = pymongo.MongoClient('localhost', 27017)
db = conn['Forex']  # mongo db name
collection = 'USD_XAU'  #mongo db collection name
db_coll = db[collection]
db_pred = db['Prediction']

with open ('data/XAU_USD.csv') as csvFile:
    csvReader = csv.DictReader(csvFile)
    for rows in csvReader:
        data = {}
        id = str(rows['Date'] + rows['Time'])
        data[id] = rows
        db_coll.insert_one(data[id])

with open ('data/XAU_USD.csv') as csvFile:
    csvReader = csv.DictReader(csvFile)
    for rows in csvReader:
        data = {}
        id = str(rows['Date'] + rows['Time'])
        data[id] = rows
        db_pred.insert_one(data[id])
