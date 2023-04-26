# ChatGPT and GitHub Copilot were used to develop this code
import re
import pandas as pd
import numpy as np

# set example text so that function can be created before CSV file is read into script
text = '🇷🇺🇧🇷 En el marco de su misión a Brasil, el Canciller ruso, Serguéi Lavrov, se reunió con su homólogo brasileño, el Ministro de Relaciones Exteriores Mauro Vieira, para mantener negociaciones bilaterales. 📍 Brasilia, 17 de abril'
   
# define text cleaning function to collapse text information with into readable written format
# define text cleaning function to collapse text information with into readable written format

def format_body_text(text):
    # remove emojis
    cleaned_text = re.sub(r'[^\w\s,().]+|[\uD800-\uDBFF][\uDC00-\uDFFF]', '', text)
    # remove all non-letter characters except for spaces, ()'s, and commas
    cleaned_text = re.sub(r'[^a-zA-Z0-9 áéíóúüñç(),.]+(?<!\s\W)(?!\W\s)', ' ', text)
    # replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # capitalize the first letter of each sentence
    cleaned_text = re.sub(r' body text ', ' ', cleaned_text)
    cleaned_text = re.sub(r' tags ', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+\d+|\W+\s+|\t', ' ', cleaned_text)
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]
    # join the sentences back together
    formatted_text = ' '.join(capitalized_sentences)
    print(formatted_text)
    return formatted_text

def create_country_columns(df, given_country_dict):
    new_columns = {}
    for col_name, country_names in given_country_dict.items():
        if isinstance(country_names, str):
            country_names = [country_names]
        pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, country_names)))
        new_col = df['formatted_text'].str.contains(pattern, case=False)
        if col_name in df.columns:
            new_columns[col_name] = np.where(new_col | (df[col_name] == 'YES'), 'YES', 'NO')
        else:
            new_columns[col_name] = np.where(new_col1, 'YES', 'NO')
    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return new_df

# read the CSV file into a pandas DataFrame
df = pd.read_csv('/Users/dwaste/Desktop/Undergrad-Thesis-Repo/sptunik-and-SOCINT-data-unlabeled/telegram-msgs-dataset.csv', encoding='utf-8', usecols=["channel_name", "cleaned_message", "date", "views", "number_replies", "number_forwards", "is_forward"])

# rename columns and reformat text
df = df.rename(columns={'cleaned_message': 'formatted_text'})

df['formatted_text'] = df['formatted_text'].apply(format_body_text)

# add country names to create new indentification columns
given_country_names = {
   "China" : ["China", "República Popular de China"],
   "United States" : ["Estados Unidos", "EE. UU.", "EEUU"],
   "LIO" : ["Fondo Monetario Internacional", "FMI" "Organização do Tratado do Atlântico Norte", "OTAN", "BM", "Summit of the Americas", "Banco Mundial", "Organización de Comercio Mundial"],
   "France" : ["Francia", "França"],
   "Germany" : ["Alemania", "Alemanha"], 
   "India" : ["India", "Índia"], 
   "Iran" : ["Irán", "Irã"], 
   "NIO" : ["BRICS", "Nuevo Banco de Desarrollo", "Organización de Cooperación de Shanghai", "Novo Banco de Desenvolvimento", "Organização de Cooperação de Xangai"], 
   "European Union" : ["Unión Europea", "Comisión Europea", "União Europeia", "Comissão Europeia", "UE" "CE"],
   "Russia" : ["Rusia", "Rússia"],
   "Ukraine" : ["Ucrania", "Ucrânia", "Kiev", "Kyiv"],
   "United Kingdom" : ["Reino Unido"],
   "Donbass" : ["Donbás", "Donbass"],
   "Taiwan" : ["Taiwán", "Taiwan"],
   "Poland" : ["Polonia", "Polônia"],
   "Baltic States" : ["Estonia", "Letonia", "Lituania", "Estônia", "Letônia", "Lituânia"],
   "Japan" : ["Japón", "Japão"]
   }

# drop rows with missing values
df = df.dropna()

# write the DataFrame to a new CSV file need to have header=True, when appending header=False
df.to_csv('/Users/dwaste/Desktop/Undergrad-Thesis-Repo/transformed-data/telegram-data-transformed.csv', mode = 'w', index=False, header=True, encoding='utf-8')
        