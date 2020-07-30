from   flask import Flask, render_template, jsonify, request
from   datetime import datetime
import json
import numpy as np
import pandas as pd
import os
import io
import requests


app = Flask(__name__, template_folder='webapp/templates', static_folder='webapp/static')

products = {
    "99197": {
        "class": 1067,
        "family": "GROCERY I",
        "perishable": 0
    },
    "105574": {
        "class": 1045,
        "family": "GROCERY I",
        "perishable": 0
    },
    "1963838": {
        "class": 3024,
        "family": "CLEANING",
        "perishable": 0
    }
}

def np2csv(arr):
    csv = io.BytesIO()
    np.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()

TENANT = os.getenv('TENANT', 'local')
FLUENTD_HOST = os.getenv('FLUENTD_HOST')
FLUENTD_PORT = os.getenv('FLUENTD_PORT')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def get_prediction():

  date_string = request.args.get('date')
  date        = datetime.strptime(date_string, '%Y-%m-%d')
  product     = products[request.args.get("item_nbr")]

  with open ("family_encoder.json", "r") as fp:
      family_encoder = json.load (fp)

  if date.weekday() >= 5:
    dayoff = 1
  else:
    dayoff = 0

  data = {
    "id": 0,
    "item_nbr": int(request.args.get("item_nbr")),
    "family": family_encoder[product['family']],
    "class": product['class'],
    "perishable": product['perishable'],
    "transactions": 1000,
    "year": date.year,
    "month": date.month,
    "day": date.day,
    "dayofweek": date.weekday(),
    "days_til_end_of_data": 0,
    "dayoff": dayoff
  }

  df      = pd.DataFrame(data, index=['data'])

  BODY    = '[[' + np2csv(df) + ']]'
  URL     = "https://0y29p03pyl.execute-api.us-east-1.amazonaws.com/test/predictdemand"
  PARAMS  = {"Content-Type": "application/json"}
  HEADERS = {'AccessKey': 'AKIATZZ6HBAUNAMYHM6F',
             'SecretKey': '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'
            }

  response = requests.post (url = URL, params = PARAMS, headers=HEADERS, data=BODY)
  data     = response.json()

  return "%d" % int (data[0])

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=5005)
