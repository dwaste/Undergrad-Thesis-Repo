# ChatGPT and GitHub Copilot were used to develop this code
import json
import pandas as pd
import re

# set example text so that function can be created before CSV file is read into script
text = '[{"body-text":"Logramos un acuerdo para prohibir la exportación de petróleo ruso a la UE. Esto cubre enseguida más de dos tercios de las importaciones de petróleo desde Rusia, cortando una enorme fuente de financiación para su maquinaria bélica\", tuiteó."},{"body-text":"El sexto paquete de sanciones contra Moscú, según el jefe del Consejo Europeo, incluye también la desconexión del mayor banco ruso Sberbank del sistema SWIFT, prohibición de otras tres emisoras estatales rusas así como las medidas contra tres individuos \"responsables de crímenes de guerra\"."},{"body-text":"Asimismo, la UE reducirá sus importaciones al bloque comunitario en un 90% para finales de 2022, declaró la presidenta de la Comisión Europea, Ursula von der Leyen."},{"body-text":"\"Aplaudo el acuerdo del Consejo Europeo de esta noche [30 de mayo] sobre las sanciones petroleras contra Rusia. Esto reducirá efectivamente alrededor del 90% de las importaciones de petróleo de Rusia a la UE para finales de año\", tuiteó von der Leyen."},{"body-text":"A su vez, Mijaíl Uliánov, representante permanente ruso ante las organizaciones internacionales en Viena, sostuvo que en este caso Moscú encontrará otros importadores."},{"body-text":"\"Cabe destacar que ahora ella [von der Leyen] contradice sus declaraciones de ayer [30 de mayo]. Un cambio de mentalidad muy rápido demuestra que la UE no está en buena forma\", escribió Uliánov en Twitter, comentando la publicación de la titular de la CE."},{"body-text":"Asimismo, los representantes permanentes de la UE realizarán este 1 de junio los trámites legales sobre un nuevo paquete de sanciones antirrusas, anunció Charles Michel."},{"body-text":"También explicó que el petróleo ruso transportado por oleoducto no entrará en el sexto paquete de sanciones."},{"body-text":"El jefe del Consejo Europeo detalló que todos los países de la UE salida al mar podrán comprar este petróleo."},{"body-text":"Al mismo tiempo, Michel agregó que se trata de una exención temporal y que los países del bloque comunitario volverán a examinar esa cuestión."},{"body-text":"El embargo petrolero que la Unión Europea activará contra Rusia estipula una excepción temporal para la República Checa y Hungría, dijo el primer ministro de Bélgica, Alexander De Croo."}]"'
   

file =  json.load('/Users/dwaste/Desktop/Undergrad-Thesis-Repo/telegram-scrape-output/EmbajadaRusa_CR/EmbajadaRusa_CR_messages.json')
file_dict = json.to_dict(file)
b = json.load(file)
print(b)
# open the input CSV file and read the rows


# text cleaning function to collapse text information with into readable written format
def format_body_text(text):
    # remove all non-letter characters except for spaces, ()'s, periods, and commas
    cleaned_text = re.sub(r'[^a-zA-Z áéíóúüñç.()]+(?<!\s\W)(?!\W\s)', '', text)
    # replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', '', cleaned_text)
    # capitalize the first letter of each sentence
    cleaned_text = re.sub(r'bodytext', '', cleaned_text)
    sentences = cleaned_text.split('. ')
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]
    # join the sentences back together
    formatted_text = '. '.join(capitalized_sentences)
    return formatted_text


    
 # loop over each row and create a new CSV file with the formatted_text, date, title, and link.href columns
    #for i, row in enumerate(json_reader):
        # rename columns and reformat text
        #text = row['body-text']
        #formatted_text = format_body_text(text)
        #date = row['date']
        #title = row['title']
        #link = row['link-href']
        # set context dependent file path
        #new_filename = f'/Users/dwaste/Desktop/Undergrad-Thesis-Repo/russian-ground-narrative-text-files/{title}.csv'
        #with open(new_filename, mode='w', newline='', encoding= "utf-8") as new_csv_file:
            #fieldnames = ['formatted_text', 'date', 'title', 'link.href']
            #writer = csv.DictWriter(new_csv_file, fieldnames=fieldnames)
            #writer.writeheader()
            #writer.writerow({'formatted_text': formatted_text, 'date': date, 'title': title, 'link.href': link})
