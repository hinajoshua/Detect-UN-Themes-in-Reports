# RW THEMES DATA RETRIEVAL FROM API AND PICKLE
# @author: Hina Joshua

import pandas as pd
import numpy as np
import requests
import os

def un_theme_data_retrievel_df(baseurl, limit, offset):

  '''
  Calls the ReliefWeb API to return the following fields from reports:
  'theme_id','theme_name', 'report_id', 'title', 'text', 'url

      @type url: str
      @param Relief Web base URL for API
      @type limit: int
      @param number of reports to be returned
      @type offset: int
      @param id number of first report called
      @rtype dataframe
      @rparam: dataframe with columns ['theme_id','theme_name', 'report_id', 'title', 'text', 'url]
  '''

  #keeping count of number of reports added to final df
  report_number = 0

  #initializing df
  df = pd.DataFrame(columns = ['theme_id', 'theme_name', 'report_id', 'text'])

  url = f'{baseurl}{limit}&offset={offset}'

  number_of_API_calls = 1
  while report_number < number_of_API_calls * int(limit):

    response = requests.get(url)
    rw_dict = response.json()

    all_reports_data=[]
    for i in np.arange(int(limit)):
      all_reports_data.append(rw_dict['data'][i]['fields'])


    for data in all_reports_data:
      try:
      # #some reports don't have a theme
        theme_id = []
        theme_name = []

        for i in data['theme']:
          theme_id.append(list(i.values())[0])
          theme_name.append(list(i.values())[1])

        df_theme = pd.DataFrame(list(zip(theme_id, theme_name)), columns=['theme_id','theme_name'])

        #add report_id to df_theme to merge df_text on later
        id_col = []
        for i in np.arange(len(df_theme)):
          id_col.append(data['id'])
        df_theme['report_id'] = id_col

        #create report text df
        report_id = [data['id']]
        text = [data['body']]
        title = [data['title']]
        current_url = [list(rw_dict['links']['next'].values())[0]]
        df_text = pd.DataFrame(list(zip(report_id, title, text, [data['url']])), columns = ['report_id', 'title', 'text', 'url'])

        #merge theme and text dfs
        df_merged = df_theme.merge(df_text, how = 'right', on= 'report_id' )

        #update df
        df= pd.concat([df, df_merged])

        #update report count
        report_number += 1

        #update url for next report API call
        url= list(rw_dict['links']['next'].values())[0]

      except:
        #update url for next API call
        url= list(rw_dict['links']['next'].values())[0]


  print(f"number of reports = {report_number}")
  print(url)

  return df.reset_index().drop('index', axis = 1)


# #call the above function to return themes for report no. 1
# baseurl = "https://api.reliefweb.int/v1/reports?appname=apidoc&query[fields][]=language&query[value]=English&profile=full&limit="
# print(un_theme_data_retrievel_df(baseurl, 1, 1)['theme_name'])

#Call the data retrieval function
#create pickle files from RW API text and 'theme' data
#use the offset parameter to keep adding reports

baseurl = "https://api.reliefweb.int/v1/reports?appname=apidoc&query[fields][]=language&query[value]=English&profile=full&limit="
limit = str(1000) #API limit is 1000
offset = str(100000) #add 4000 to offset to grab next 4000 reports/calls

for i in np.arange(4): #adds 4000 more reports/calls
  df = un_theme_data_retrievel_df(baseurl, limit, offset)
  df.to_pickle(r'data\RW_UN_themes_data_pickle'+offset)
#   df.to_pickle(r'C:\Users\hinam\Desktop\Data4Good\d4g-backend\Theme Detection\data\themes_data'+offset+'.pkl')
  offset = str(int(offset) + int(limit))

#Read in the stored pickle files
df_0 = pd.read_pickle(r'data\themes_data10.pkl')
print(f'The fields returned by the above API call include: \n \n{list(df_0.columns)}')
# Get the list of all pickle files in directory
path = r'data'
dir_list = os.listdir(path)
# prints all files
print(f'''Files and directories in {path} :\n{dir_list}\n Total number of file = {len(dir_list)}''')