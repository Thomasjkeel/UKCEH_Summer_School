from datetime import datetime
import io
import json
import requests
import zipfile

import pandas as pd

BASE_URL = 'https://cosmos-api.ceh.ac.uk'

def get_api_response(url, csv=False):
    """ Helper function to send request to API and get the response

    :param str url: The URL of the API request
    :param bool csv: Whether this is a CSV request. Default False.
    :return: API response
    """
    # Send request and read response
    print(url)
    response = requests.get(url)

    if csv:
        return response
    else:
        # Decode from JSON to Python dictionary
        return json.loads(response.content)


def get_collection_parameter_info(params):
    """ A function for wrangling the collection information into a more visually appealing format!
    """
    df = pd.DataFrame.from_dict(params)
    df = df.T[['label', 'description', 'unit', 'sensorInfo']]

    df['unit_symbol'] = df['unit'].apply(lambda x: x['symbol']['value'])
    df['unit_label'] = df['unit'].apply(lambda x: x['label'])
    df['sensor_depth'] = df['sensorInfo'].apply(lambda x: None if pd.isna(x) else x['sensor_depth']['value'])

    df = df.drop(['sensorInfo', 'unit'], axis=1)

    return df