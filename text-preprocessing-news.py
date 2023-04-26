# ChatGPT and GitHub Copilot were used to develop this code
import re
import pandas as pd
import numpy as np

# set example text so that function can be created before CSV file is read into script
text = '[{"body-text":"Logramos un acuerdo para prohibir la exportación de petróleo ruso a la UE. Esto cubre enseguida más de dos tercios de las importaciones de petróleo desde Rusia, cortando una enorme fuente de financiación para su maquinaria bélica\", tuiteó."},{"body-text":"El sexto paquete de sanciones contra Moscú, según el jefe del Consejo Europeo, incluye también la desconexión del mayor banco ruso Sberbank del sistema SWIFT, prohibición de otras tres emisoras estatales rusas así como las medidas contra tres individuos \"responsables de crímenes de guerra\"."},{"body-text":"Asimismo, la UE reducirá sus importaciones al bloque comunitario en un 90% para finales de 2022, declaró la presidenta de la Comisión Europea, Ursula von der Leyen."},{"body-text":"\"Aplaudo el acuerdo del Consejo Europeo de esta noche [30 de mayo] sobre las sanciones petroleras contra Rusia. Esto reducirá efectivamente alrededor del 90% de las importaciones de petróleo de Rusia a la UE para finales de año\", tuiteó von der Leyen."},{"body-text":"A su vez, Mijaíl Uliánov, representante permanente ruso ante las organizaciones internacionales en Viena, sostuvo que en este caso Moscú encontrará otros importadores."},{"body-text":"\"Cabe destacar que ahora ella [von der Leyen] contradice sus declaraciones de ayer [30 de mayo]. Un cambio de mentalidad muy rápido demuestra que la UE no está en buena forma\", escribió Uliánov en Twitter, comentando la publicación de la titular de la CE."},{"body-text":"Asimismo, los representantes permanentes de la UE realizarán este 1 de junio los trámites legales sobre un nuevo paquete de sanciones antirrusas, anunció Charles Michel."},{"body-text":"También explicó que el petróleo ruso transportado por oleoducto no entrará en el sexto paquete de sanciones."},{"body-text":"El jefe del Consejo Europeo detalló que todos los países de la UE salida al mar podrán comprar este petróleo."},{"body-text":"Al mismo tiempo, Michel agregó que se trata de una exención temporal y que los países del bloque comunitario volverán a examinar esa cuestión."},{"body-text":"El embargo petrolero que la Unión Europea activará contra Rusia estipula una excepción temporal para la República Checa y Hungría, dijo el primer ministro de Bélgica, Alexander De Croo."}]"'
   
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
    return formatted_text

def create_country_columns(df, given_country_dict):
    new_columns = {}
    for col_name, country_names in given_country_dict.items():
        if isinstance(country_names, str):
            country_names = [country_names]
        pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, country_names)))
        new_col1 = df['formatted_text'].str.contains(pattern, case=False)
        new_col2 = df['title'].str.contains(pattern, case=False)
        if col_name in df.columns:
            new_columns[col_name] = np.where(new_col1 | new_col2 | (df[col_name] == 'YES'), 'YES', 'NO')
        else:
            new_columns[col_name] = np.where(new_col1 | new_col2, 'YES', 'NO')
    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return new_df

# read the CSV file into a pandas DataFrame
df = pd.read_csv('/Users/dwaste/Desktop/Undergrad-Thesis-Repo/sptunik-and-SOCINT-data-unlabeled/sputnik-brasil-smo-news-scraper.csv', encoding='utf-8')

# rename columns and reformat text
df = df.rename(columns={'body-text': 'formatted_text', 'link-href': 'link_href', 'tags': 'formatted_tags'})

df['formatted_text'] = df['formatted_text'].apply(format_body_text)
df['formatted_tags'] = df['formatted_tags'].apply(format_body_text)

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

# run extra-regional actor tags
df = create_country_columns(df, given_country_names)

df = df.drop("web-scraper-start-url", axis= 1)
df = df.drop("quote-text", axis= 1)
df = df.drop("web-scraper-order", axis= 1)
df = df.drop("announce-text", axis= 1)
df = df.drop("link", axis= 1)

# drop rows with missing values
df = df.dropna()

# write the DataFrame to a new CSV file need to have header=True, when appending header=False
df.to_csv('/Users/dwaste/Desktop/Undergrad-Thesis-Repo/transformed-data/combined-Sputnik-Brasil-data.csv', mode = 'w', index=False, header=True, encoding='utf-8')
        