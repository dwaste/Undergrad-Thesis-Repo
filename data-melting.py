import pandas as pd 
import re

sputnik_mundo_data = pd.read_csv("C:/Users/Dylan/Desktop/Undergrad-Thesis-Repo/transformed-data/sputnik-mundo-transformed-data.csv", encoding="utf-8")
#sputnik_brasil_data = pd.read_csv("/Users/dwaste/Desktop/Undergrad-Thesis-Repo/transformed-data/sputnik-brasil-transformed-data.csv", encoding="utf-8")
twit_data = pd.read_csv("C:/Users/Dylan/Desktop/Undergrad-Thesis-Repo/transformed-data/twitter-transformed-data.csv", encoding="utf-8")
#tele_data = pd.read_csv("/Users/dwaste/Desktop/Undergrad-Thesis-Repo/transformed-data/telegram-transformed-data.csv", encoding="utf-8")

def merge_dataframes(df1, df3):
    # add platform column to each dataframe
    df1['platform'] = 'Sputnik Mundo'
    #df2['platform'] = 'Sputnik Brasil'
    df3['platform'] = 'Twitter'
    #df4['platform'] = 'Telegram'
    
    # add row_index column to each dataframe
    df1['row_index'] = range(len(df1))
    #df2['row_index'] = range(len(df2))
    df3['row_index'] = range(len(df3))
    #df4['row_index'] = range(len(df4))
    
    # rename columns in each dataframe
    df1 = df1.rename(columns={'title': 'title_or_account'})
    #df2 = df2.rename(columns={'title': 'title_or_account'})
    df3 = df3.rename(columns={'UserScreenName': 'title_or_account', 'Timestamp' : 'date'})
    #df4 = df4.rename(columns={'UserScreenName': 'title_or_account'})
    
    # merge dataframes
    merged_df = pd.concat([df1, df3], ignore_index=True)
    
    # add label column and fill null values
    merged_df['Economía'] = merged_df['Economía'].fillna(-1)
    merged_df['Defensa'] = merged_df['Defensa'].fillna(-1)
    merged_df['Internacional'] = merged_df['Internacional'].fillna(-1)
    merged_df['América Latina'] = merged_df['América Latina'].fillna(-1)
    merged_df['Neonazismo en Ucrania'] = merged_df['Neonazismo en Ucrania'].fillna(-1)
    merged_df.loc[merged_df['platform'] == 'Sputnik Mundo', 'Economía'] = df1['Economía']
    merged_df.loc[merged_df['platform'] == 'Sputnik Mundo', 'Defensa'] = df1['Defensa']
    merged_df.loc[merged_df['platform'] == 'Sputnik Mundo', 'Internacional'] = df1['Internacional']
    merged_df.loc[merged_df['platform'] == 'Sputnik Mundo', 'América Latina'] = df1['América Latina']
    merged_df.loc[merged_df['platform'] == 'Sputnik Mundo', 'Neonazismo en Ucrania'] = df1['Neonazismo en Ucrania']

    # re-order columns
    merged_df = merged_df[['platform', 'row_index', 'formatted_text', 'title_or_account', 'date', 'Economía', 'Defensa', 'Internacional', 'América Latina', 'Neonazismo en Ucrania']]

    return merged_df

merged_df = merge_dataframes(sputnik_mundo_data, twit_data)

# write the DataFrame to a new CSV file need to have header=True, when appending header=False
merged_df.to_csv('C:/Users/Dylan/Desktop/Undergrad-Thesis-Repo/transformed-data/combined_transformated_data.csv', mode = 'w', index=False, header=True, encoding='utf-8')

