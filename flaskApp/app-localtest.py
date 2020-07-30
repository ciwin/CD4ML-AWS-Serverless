import pandas as pd
import io
import json
import requests
import numpy as np
from datetime import datetime



# Convert a np array to a csv file:
def np2csv(arr):
    csv = io.BytesIO()
    np.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()


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

date_string = "2020-08-30"
prod_string = "99197"



date    = datetime.strptime(date_string, '%Y-%m-%d')
product = products[prod_string]

with open ("family_encoder.json", "r") as fp:
    family_encoder = json.load (fp)

if date.weekday() >= 5:
    dayoff = 1
else:
    dayoff = 0

data = {
    "id": 0,
    "item_nbr": int(prod_string),
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

df = pd.DataFrame(data, index=['data'])

BODY    = '[[' + np2csv(df) + ']]'
URL     =   "https://0y29p03pyl.execute-api.us-east-1.amazonaws.com/test/predictdemand"
PARAMS  =   {"Content-Type": "application/json"}
HEADERS =   {'AccessKey': 'AKIATZZ6HBAUNAMYHM6F',
             'SecretKey': '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'
            }

response = requests.post (url = URL, params = PARAMS, headers=HEADERS, data=BODY)

data = response.json()

print(response.status_code)
print(response.content)
print(data[0])

# return "%d" % result[0]

# runtime= boto3.client('runtime.sagemaker')

#response = runtime.invoke_endpoint(
#      EndpointName= 'sagemaker-scikit-learn-2020-07-29-11-49-57-111',
#      Body = df.values,
      # Body='[[0,99197,11,1067,0,1000,2020,7,30,3,0,0]]',
#      ContentType='application/json')

#  result = json.loads(response['Body'].read().decode())
