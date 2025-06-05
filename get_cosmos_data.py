"""
    Functions for accessing COSMOS data.
    Please see https://cosmos-api.ceh.ac.uk/python_examples for code examples
    Please see https://cosmos-api.ceh.ac.uk/docs for more details
"""
import json
import requests

import pandas as pd

BASE_URL = 'https://cosmos-api.ceh.ac.uk'


def get_single_station_data(station_name, start_date, end_date, variable_list, site_info=None):
    if not site_info:
        site_info = get_cosmos_station_metadata()
    start_date = format_datetime(start_date)
    end_date = format_datetime(end_date)
    query_date_range = f"{start_date}/{end_date}"

    # Extracting COSMOS data for variables the station over the required period into a pandas dataframe
    query_url = f'{BASE_URL}/collections/1D/locations/{station_name}?datetime={query_date_range}&parameter-name={",".join(variable_list)}'
    resp = get_api_response(query_url)
    df = read_json_collection_data(resp)
    df = df.reset_index()
    return df


def get_single_station_metadata(station_name, site_info=None):
    if not site_info:
        site_info = get_cosmos_station_metadata()
    site_info_df = pd.DataFrame.from_dict(site_info).T
    s_df = site_info_df[site_info_df.index == station_name]
    return s_df


def get_cosmos_station_metadata():
    """ Get COSMOS station metadata

    :return: all COSMOS stations metadata
    """
    site_info_url = f"{BASE_URL}/collections/1D/locations"
    site_info_response = get_api_response(site_info_url)

    site_info = {}
    for site in site_info_response["features"]:
        site_id = site["id"]
        site_name = site["properties"]["label"]
        coordinates = site["geometry"]["coordinates"]
        date_range = site["properties"]["datetime"]
        start_date, end_date = date_range.split("/")

        other_info = site["properties"]["siteInfo"]
        other_info = {key: d["value"] for key, d in other_info.items()}

        site_info[site_id] = {
            "site_name": site_name,
            "coordinates": coordinates,
            "start_date": start_date,
            "end_date": end_date,
        } | other_info
    return site_info


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


def format_datetime(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json_collection_data(json_response):
    """Wrangle the response JSON from a COSMOS-API data collection request into a more usable format - in this case a Pandas Dataframe

    :param dict json_response: The JSON response dictionary returned from a COSMOS-API data collection request
    :return: Dataframe of data
    :rtype: pd.DataFrame
    """
    # The response is a list of dictionaries, one for each requested site

    # You can choose how you want to build your dataframes.  Here, I'm just loading all stations into one big dataframe.
    # But you could modify this for your own use cases.  For example you might want to build a dictionary of {site_id: dataframe}
    # to keep site data separate, etc.
    master_df = pd.DataFrame()

    for site_data in json_response["coverages"]:
        # Read the site ID
        site_id = site_data["dct:identifier"]

        # Read the time stamps of each data point
        time_values = pd.DatetimeIndex(site_data["domain"]["axes"]["t"]["values"])

        # Now read the values for each requested parameter at each of the time stamps
        param_values = {
            param_name: param_data["values"]
            for param_name, param_data in site_data["ranges"].items()
        }

        # And put everything into a dataframe
        site_df = pd.DataFrame.from_dict(param_values)
        site_df["datetime"] = time_values
        site_df["site_id"] = site_id

        site_df = site_df.set_index(["datetime", "site_id"])
        master_df = pd.concat([master_df, site_df])

    return master_df