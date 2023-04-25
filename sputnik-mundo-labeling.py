import pandas as pd 
import numpy as np
import re 

# read in data
df = pd.read_csv("/Users/dwaste/Desktop/Undergrad-Thesis-Repo/transformed-data/sputnik-data-transformed.csv", encoding='utf-8')

def create_scope_columns(df, given_scope_dict):
    for col_name, narrative_scopes in given_scope_dict.items():
        if isinstance(narrative_scopes, str):
            narrative_scopes = [narrative_scopes]
        pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, narrative_scopes)))
        new_col = df['formatted_tags'].str.contains(pattern, case=False)
        if col_name not in df.columns:
            df[col_name] = np.where(new_col, 1, 0)
        else:
            df[col_name] = np.where(new_col | (df[col_name] == 1), 1, 0)
    return df

given_scope_dict = {
    "Economía" : ["sanciones", "economicas", "Mercados y finanzas", "gas", "petróleo", "Consecuencias económicas de las sanciones occidentales contra Rusia", "Compañías", "embargo", "inflación", "Industria militar"],
    "Defensa" : ["Operación rusa de desmilitarización y desnazificación de Ucrania", "Industria militar"], 
    "Internacional" : ["Internacional"],
    "América Latina" : ["América Latina"],
    "Neonazismo en Ucrania" : ["Neonazismo en Ucrania", "Donbás. La historia de un genocidio", "Azov (batallón)", "Stepán Bandera", "víctimas civiles"], 
}

df = create_scope_columns(df, given_scope_dict)

df.to_csv('/Users/dwaste/Desktop/Undergrad-Thesis-Repo/transformed-data/sputnik-data-transformed.csv', mode = 'w', index=False, encoding='utf-8')