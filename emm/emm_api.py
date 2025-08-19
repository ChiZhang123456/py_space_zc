'''
This python script contains helper functions that enable downloading science data from the
EMM Science Data Center.  The most straightforward to use is the function "emm_search_and_download", which encapsulates
several of the other functions.

Modified: 2021-10-06
Author: emm-sdc@lists.lasp.colorado.edu
'''

'''
All query/download functions accept the following as filters

Single-value parameters 
    :param instrument: "emr", "exi", or "emu"
    :param level: "l0", "l1", "l2a", etc
    :param latest: 'true' will only return the latest version of the data, and 'false' will return it all.  Default is 'true'.  Must be in string format.
    :param pred_rec: Can be 'p', 'r', 'n', or 'f', which filters files based on usage of "predicted", "reconstructed", "final", or "N/A" SPICE kernels
    :param orbit_num: An integer that represents the orbit number of files to search
    :param mode: A string the represents the instrument's observation mode ("os4", "xos5", etc)
    
Ranged query parameters:
    :param start_date: A string date time formatted as "YYYY-MM-DD" which represents the UTC start date of the data to query
    :param end_date: A string date time formatted as "YYYY-MM-DD" which represents the UTC end date of the data to query
    :param start_sc_lst: A string time formatted as "HH:MM:SS" which represents the start of the Local Solar Time of the sub-spacecraft point 
    :param end_sc_lst: A string time formatted as "HH:MM:SS" which represents the end of the local Solar Time of the sub-spacecraft point 
    :param start_mars_ls: A number which represents the start of the Mars Season to query
    :param end_mars_ls: A number which represents the end of the Mars Season to query
    :param start_sc_lat: A number which represents the start of the IAU MARS Planetographic sub-spacecraft latitude  
    :param end_sc_lat: A number which represents the end of the IAU MARS Planetographic sub-spacecraft latitude  
    :param start_sc_lon: A number which represents the start of the IAU MARS Planetographic sub-spacecraft longitude  
    :param end_sc_lon: A number which represents the end of the IAU MARS Planetographic sub-spacecraft longitude
    :param start_sc_alt: A number which represents the start of the IAU MARS Planetographic spacecraft altitude  
    :param end_sc_alt: A number which represents the end of the IAU MARS Planetographic spacecraft altitude
    :param start_subsolar_lat: A number which represents the start of the IAU MARS Planetographic subsolar latitude  
    :param end_subsolar_lat: A number which represents the end of the IAU MARS Planetographic subsolar latitude  
    :param start_subsolar_lon: A number which represents the start of the IAU MARS Planetographic subsolar longitude  
    :param end_subsolar_lon: A number which represents the end of the IAU MARS Planetographic subsolar longitude
'''

import requests
import json
import os
import datetime


USER_TOKEN = None
LOGIN_TIME = None



def _set_user_token(t):
    global USER_TOKEN
    global LOGIN_TIME

    LOGIN_TIME = datetime.datetime.now()
    USER_TOKEN = t


def _get_user_token():
    global USER_TOKEN
    global LOGIN_TIME
    if LOGIN_TIME is None:
        print("New login needed.  Login is valid for 60 minutes.")
    elif (datetime.datetime.now() - LOGIN_TIME).total_seconds() >= 3600:
        print("Login expired.  Please log in again.")
    else:
        return USER_TOKEN

    t = get_sdc_token()

    return t


def get_sdc_token(user_name=None, password=None):
    '''
    This function authenticates the user with the EMM SDC.  An access token is automatically stored in the USER_TOKEN
    variable in this file, and functions will attempt to find a valid user token in that variable.

    :param user_name: User's EMM SDC username
    :param password: User's EMM SDC password

    :return: A string that also gets stored in the USER_TOKEN variable in this file.  You don't need this string unless
             you plan on making your own API calls, using functions outside of this file.
    '''
    user_name='Chi_Zhang_bu'
    password='ZcloveQf13126819596.'
    
    '''if user_name is None:
        user_name = input("Username:")
    if password is None:
        import getpass
        password = getpass.getpass("Password for " + user_name + ":")
    '''
    authentication_url = "https://cognito-idp.eu-west-1.amazonaws.com/"
    authentication_headers = {'X-Amz-Target': 'AWSCognitoIdentityProviderService.InitiateAuth',
                              'Content-Type': 'application/x-amz-json-1.1'}
    data = json.dumps({"ClientId": "n5e6d97bl4ba76rrdtm0qaq6n", "AuthFlow": "USER_PASSWORD_AUTH",
                       "AuthParameters": {"USERNAME": user_name, "PASSWORD": password}})

    # Attempt to grab the SDC token.
    try:
        token_response = requests.post(authentication_url, data=data, headers=authentication_headers)
        t = token_response.json()['AuthenticationResult']['AccessToken']
    except KeyError:
        print("Invalid username and/or password.  Please try again.  ")
        return

    _set_user_token(t)

    return t


def query_science_files(instrument=None, level=None,
                        latest='true', pred_rec=None, orbit_num=None,
                        mode=None, token=None, **kwargs):
    '''
    This function queries the SDC for files based on the input parameters, and returns a list of dictionaries, with each
    dictionary containing information about the file (version, date, local solar time, etc)

    :param kwargs: See the header of the file for a list of all query parameters that the API accepts.
    :param token: The Token string (retrieved from "get_sdc_token") that validates the user. This is not needed unless
                  you plan on retrieving your token using a method not found in this module.

    :return: A JSON object describing the science data discovered by the query
    '''

    if token is None:
        token = _get_user_token()

    headers = {"Authorization": token}
    query_url = "https://mdhkq4bfae.execute-api.eu-west-1.amazonaws.com/prod/science-files-metadata?"

    query_parameters = []
    if level is not None:
        query_parameters.append("data_level=" + level)
    if instrument is not None:
        query_parameters.append('instrument_id='+instrument)
    if pred_rec is not None:
        query_parameters.append("pred_rec=" + pred_rec)
    if orbit_num is not None:
        query_parameters.append("orbit_num=" + str(orbit_num))
    if mode is not None:
        query_parameters.append("mode=" + str(mode))
    for kw in kwargs:
        query_parameters.append(kw + "=" + str(kwargs[kw]))

    query_parameters.append("latest=" + latest)

    query_parameters = '&'.join(query_parameters)

    query_url_with_parameters = query_url + query_parameters

    try:
        file_list = requests.get(query_url_with_parameters, headers=headers)
    except Exception as e:
        print(f"Could not finish query due to error {str(e)}")
        return
    if (file_list.status_code == 504) or (file_list.status_code == 502):
        print("Too many files found in API query.  Please narrow search. For example, "
              "try breaking up search parameters with start_date/end_date to only search a few months at a time.")
        return

    return file_list.json()


def download_science_tar(file_path='', instrument=None, level=None,
                         pred_rec=None, orbit_num=None,
                         latest='true', token=None, **kwargs):
    '''
    A function that downloads matching files as a .tgz file.

    :param file_path: (required) A string name to give the downloaded zip file
    :param kwargs: See the header of the file for a list of all query parameters that the API accepts.
    :param token: The Token string (retrieved from "get_sdc_token") that validates the user.  This is not needed unless
                  you plan on retrieving your token using a method not found in this module.

    :return: None, but downloads a zip file of all matching data.
    '''

    if token is None:
        token = _get_user_token()

    headers = {"Authorization": token}
    query_url = "https://mdhkq4bfae.execute-api.eu-west-1.amazonaws.com/prod/science-files-download?"

    query_parameters = []
    if level is not None:
        query_parameters.append("data_level=" + level)
    if instrument is not None:
        query_parameters.append('instrument_id='+instrument)
    if pred_rec is not None:
        query_parameters.append("pred_rec=" + pred_rec)
    if orbit_num is not None:
        query_parameters.append("orbit_num=" + str(orbit_num))
    for kw in kwargs:
        query_parameters.append(kw + "=" + str(kwargs[kw]))
    query_parameters.append("latest=" + latest)

    query_parameters = '&'.join(query_parameters)

    query_url_with_parameters = query_url + query_parameters

    try:
        zip_file_location = requests.get(query_url_with_parameters, headers=headers)
    except Exception as e:
        print(f"Could not finish query due to error {str(e)}")
        return
    if (zip_file_location.status_code == 504) or (zip_file_location.status_code == 502):
        print("Too many files found in API query.  Please narrow search. For example, "
              "try breaking up search parameters with start_date/end_date to only search a few months at a time.")
        return
    elif (zip_file_location.status_code == 404):
        print("No files were found matching the search parameters.")
        return

    file_location = requests.get(zip_file_location.content.decode('utf-8'))

    open(file_path, 'wb').write(file_location.content)

    return


def download_from_query(query_results, token=None, download_dir = '', use_sdc_dir_structure=True):
    '''
    A function that downloads files based on the output of "query_science_files".

    :param query_results: The JSON object returned from the "query_science_files" functions
    :param token: The Token string (retrieved from "get_token") that validates the user. This is not needed unless
                  you plan on retrieving your token using a method not found in this module.
    :param download_dir: The location where you would like the files to be saved.

    :return: None, but downloads the files to "download_dir"
    '''

    if token is None:
        token = _get_user_token()

    headers = {"Authorization": token}
    query_url = "https://mdhkq4bfae.execute-api.eu-west-1.amazonaws.com/prod/science-files-download?"

    i = 0
    for f in query_results:
        i+=1
        query_parameters = []
        query_parameters.append("file_name=" + f['file_name'])
        query_parameters = '&'.join(query_parameters)
        query_url_with_parameters = query_url + query_parameters
        if use_sdc_dir_structure:
            download_dir_full_path = os.path.join(download_dir, f['directory_path'].replace("s3://data-sdc-emmsdc-mbrsc/", ""))

        try:
            if not os.path.exists(download_dir_full_path):
                os.makedirs(download_dir_full_path)

            file_name_and_path = os.path.join(download_dir_full_path, f['file_name'])
            if os.path.exists(file_name_and_path):
                continue

            with open(file_name_and_path, 'wb') as file:
                print(f"Downloading {file_name_and_path}")
                single_file_location = requests.get(query_url_with_parameters, headers=headers)
                file_location = requests.get(single_file_location.content.decode('utf-8'))
                file.write(file_location.content)

        except Exception as e:
            print(f"Could not download file {f['file_name']} due to error {str(e)}")

    return


def emm_search_and_download(download_dir,
                            instrument=None,
                            level=None,
                            latest='true',
                            orbit_num=None,
                            pred_rec=None,
                            use_sdc_dir_structure=True,
                            **kwargs):
    '''
    This function will download files to a local directory on the user's machine.  It will prompt the user to enter a
    username and password.

    :param download_dir: (Required) The location where you would like the files to be saved.
    :param kwargs: See the header of the file for a list of all query parameters that the API accepts.

    :return: None, but files are downloaded that match the input parameters
    '''

    # Query the SDC for relevant files
    q = query_science_files(instrument=instrument, level=level, latest=latest, pred_rec=pred_rec,
                            orbit_num=orbit_num, **kwargs)

    # Download all matching files
    if q is not None:
        download_from_query(q, download_dir=download_dir, use_sdc_dir_structure=use_sdc_dir_structure)

def download_data(start_date, end_date, mode, level = "l2a"):
    #这个可能后面是需要修改的。
    download_dir = "F:\data";
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    current = start

    while current < end:
            # 计算当前周期的结束日期，但不超过设定的结束日期
          next_date = min(current + datetime.timedelta(days=20), end)
            
            # 格式化日期为字符串，以便传递给 download_data 函数
          start_str = current.strftime("%Y-%m-%d")
          next_str = next_date.strftime("%Y-%m-%d")
          emm_search_and_download(download_dir = download_dir,instrument='emu',
                                  latest='true',mode = mode,level = level,
                                  start_date=start_str,end_date=next_str)
          current = next_date
          

    
if __name__ == "__main__":

    # start_date='2023-02-01';
    # end_date='2023-02-28';
    # download_data(start_date, end_date, 'os2', "l2a")
    # download_data(start_date, end_date, 'osr', "l2a")
    # download_data(start_date, end_date, 'os1', "l2a")
    
    start_date='2022-12-01';
    end_date='2022-12-31';
    download_data(start_date, end_date, 'os2', "l2a")
    download_data(start_date, end_date, 'osr', "l2a")
    download_data(start_date, end_date, 'os1', "l2a")
    
    start_date='2022-11-01';
    end_date='2022-11-30';
    download_data(start_date, end_date, 'os2', "l2a")
    download_data(start_date, end_date, 'osr', "l2a")
    download_data(start_date, end_date, 'os1', "l2a")
    
    start_date='2022-10-01';
    end_date='2022-10-31';
    download_data(start_date, end_date, 'os2', "l2a")
    download_data(start_date, end_date, 'osr', "l2a")
    download_data(start_date, end_date, 'os1', "l2a")
    #download_data(start_date, end_date, 'osr')