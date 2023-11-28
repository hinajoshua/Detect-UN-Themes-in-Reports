# RW THEMES DATASET CLEAN AND SAVE
# @author: Hina Joshua

import pandas as pd
import os
from collections import Counter

#Read in the stored pickle files and print fields 
df_0 = pd.read_pickle(r'data\RW_UN_themes_data_pickle0')
print(f'The fields returned by the API call include: \n \n{list(df_0.columns)}')
#Get the list of all pickle files in directory
path = r'data'
dir_list = os.listdir(path)
# prints all filenames
# print(f'''Files and directories in {path} :\n{dir_list}\n Total number of file = {len(dir_list)}''')

#Read in all pickle files in a single df
base_path = r'data'
def read_pickles(base_path):
  '''
    @type base_path: str
    @path to folder with all pickle files
    @rtype df: dataframe
    @rparam contains RW themes and text data from all pickled files
  '''
  dir_list = os.listdir(base_path)
  df = pd.DataFrame(columns =['theme_id', 'theme_name', 'report_id', 'text', 'title', 'url'])

  for filename in dir_list:
    path = f'{base_path}/{filename}'
    #read content pickle content into df
    df_new = pd.read_pickle(path)
    #update df
    df= pd.concat([df, df_new])
  return df

df = read_pickles(base_path)

#checking df shape and number of reports represented
df_shape = df.shape
print(f'Shape of the Dataframe (multiple rows per unique report) = {df_shape[0]} rows and {df_shape[1]} columns')
report_number = len(df['report_id'].unique())

print(f'Number of unique ReliefWeb reports retrieved = {report_number}')

#Count of unique themes
#Creating a Counter class object using list as an iterable data container
theme_names = df['theme_name']
theme_counts = Counter(theme_names)

#printing theme counts
print(f'There are {len(theme_counts.keys())} Themes represented in this dataset.\nThey are: \n{list((theme_counts.keys()))}')
print(f'\n')
print(f'Counts of each unique "theme" in this dataset:\n{sorted(dict(theme_counts).items(), key=lambda item: item[1])}')

#Theme List
theme_list= list(df['theme_name'].unique())

#Binarize the features(themes)
#create 20 feature columns, fill with binary values. To be used for MultiLabel Classification

def binarize_themes(df):
  '''
    @type df: DataFrame
    @param dataframe contains RW themes and text data from all pickled files
    @rtype df: DataFrame
    @rparam df containing 20 additional columns (one per theme) with binary values
  '''
  for theme in theme_list:
    values = []
    for ind, row in df.iterrows():
        if row['theme_name'] == theme:
            values.append(1)
        else:
            values.append(0)
    df[theme] = values

  #Keeps single row for unique report_id
  df2 = df.groupby(['report_id'])[theme_list].max().reset_index()

  df3 = df[['report_id', 'text', 'title', 'url']].merge(df2, on = 'report_id', how = 'left')
  df_binarized = df3.drop_duplicates()
  print(f"Shape of Final df is {df_binarized.shape}")
  return df_binarized

# Create Balanced Dataset
theme_list = list(theme_counts.keys())
min_count = min(theme_counts.values())
def balance_dataset(df, min_count):
  '''
    @type df: DataFrame
    @param df containing 20 additional columns (one per theme) with binary values
    @type min_count: int
    @param minimum count of theme type
    @rtype df: DataFrame
    @rparam balanced and reduced dataframe with themes equally represented
  '''
  df_balanced = pd.DataFrame(columns = df.columns)
  for theme in theme_list:
    df_new = df[df[theme] == 1][:min_count]
    # update df_balanced
    df_balanced= pd.concat([df_balanced, df_new])
  return df_balanced

df_all_binarized = binarize_themes(df)
df_balanced_binarized = balance_dataset(df_all_binarized, min_count)
print(f"the shape of balanced_df is {df_balanced_binarized.shape}")

#Save df as csv 
df_all_binarized.to_csv(r'rw_themes_multilabeldf.csv')
df_balanced_binarized.to_csv(r'rw_themes_balanced_df.csv')