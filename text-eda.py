import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

df = pd.read_csv("/Users/dwaste/Desktop/Undergrad-Thesis-Repo/transformed-data/sputnik-data-transformed.csv", encoding= "utf-8")

# function to plot the most common words in a text column
def plot_word_frequency(df, column_name, num_words=50):
    # remove punctuation and convert to lowercase
    text = ' '.join(df[column_name]).lower()
    text = re.sub(r'[^\w\s]','',text)
    # remove stop words
    stop_words = set(["algún","alguna","algunas","alguno","algunos","ambos","ampleamos","ante","antes","aquel","aquellas","aquellos","aqui","arriba","atras","bajo","bastante","bien","cada","cierta","ciertas","cierto","ciertos","como","con","conseguimos","conseguir","consigo","consigue","consiguen","consigues","cual","cuando","dentro","desde","donde","dos","el","ellas","ellos","empleais","emplean","emplear","empleas","empleo","en","encima","entonces","entre","era","eramos","eran","eras","eres","es","esta","estaba","estado","estais","estamos","estan","estoy","fin","fue","fueron","fui","fuimos","gueno","ha","hace","haceis","hacemos","hacen","hacer","haces","hago","incluso","intenta","intentais","intentamos","intentan","intentar","intentas","intento","ir","la","largo","las","lo","los","mientras","mio","modo","muchos","muy","nos","nosotros","otro","para","pero","podeis","podemos","poder","podria","podriais","podriamos","podrian","podrias","por","por qué","porque","primero","puede","pueden","puedo","quien","sabe","sabeis","sabemos","saben","saber","sabes","ser","si","siendo","sin","sobre","sois","solamente","solo","somos","soy","su","sus","también","teneis","tenemos","tener","tengo","tiempo","tiene","tienen","todo","trabaja","trabajais","trabajamos","trabajan","trabajar","trabajas","trabajo","tras","tuyo","ultimo","un","una","unas","uno","unos","usa","usais","usamos","usan","usar","usas","uso","va","vais","valor","vamos","van","vaya","verdad","verdadera","verdadero","vosotras","vosotros","voy","yo","él","ésta","éstas","éste","éstos","última","últimas","último","últimos","a","añadió","aún","actualmente","adelante","además","afirmó","agregó","ahí","ahora","al","algo","alrededor","anterior","apenas","aproximadamente","aquí","así","aseguró","aunque","ayer","buen","buena","buenas","bueno","buenos","cómo","casi","cerca","cinco","comentó","conocer","consideró","considera","contra","cosas","creo","cuales","cualquier","cuanto","cuatro","cuenta","da","dado","dan","dar","de","debe","deben","debido","decir","dejó","del","demás","después","dice","dicen","dicho","dieron","diferente","diferentes","dijeron","dijo","dio","durante","e","ejemplo","ella","ello","embargo","encuentra","esa","esas","ese","eso","esos","está","están","estaban","estar","estará","estas","este","esto","estos","estuvo","ex","existe","existen","explicó","expresó","fuera","gran","grandes","había","habían","haber","habrá","hacerlo","hacia","haciendo","han","hasta","hay","haya","he","hecho","hemos","hicieron","hizo","hoy","hubo","igual","indicó","informó","junto","lado","le","les","llegó","lleva","llevar","luego","lugar","más","manera","manifestó","mayor","me","mediante","mejor","mencionó","menos","mi","misma","mismas","mismo","mismos","momento","mucha","muchas","mucho","nada","nadie","ni","ningún","ninguna","ningunas","ninguno","ningunos","no","nosotras","nuestra","nuestras","nuestro","nuestros","nueva","nuevas","nuevo","nuevos","nunca","o","ocho","otra","otras","otros","parece","parte","partir","pasada","pasado","pesar","poca","pocas","poco","pocos","podrá","podrán","podría","podrían","poner","posible","próximo","próximos","primer","primera","primeros","principalmente","propia","propias","propio","propios","pudo","pueda","pues","qué","que","quedó","queremos","quién","quienes","quiere","realizó","realizado","realizar","respecto","sí","sólo","se","señaló","sea","sean","según","segunda","segundo","seis","será","serán","sería","sido","siempre","siete","sigue","siguiente","sino","sola","solas","solos","son","tal","tampoco","tan","tanto","tenía","tendrá","tendrán","tenga","tenido","tercera","toda","todas","todavía","todos","total","trata","través","tres","tuvo","usted","varias","varios","veces","ver","vez","y","ya"])
    words = [word for word in text.split() if word not in stop_words]
    # get word frequencies
    word_freq = Counter(words)
    top_words = dict(word_freq.most_common(num_words))
    # plot the top words
    plt.figure(figsize=(10,6))
    plt.barh(y=list(top_words.keys()), width=list(top_words.values()))
    plt.title('Top {} words in {}'.format(num_words, column_name))
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.show()

# function to plot the distribution of word lengths in a text column
def plot_word_length_distribution(df, column_name):
    # remove punctuation and convert to lowercase
    text = ' '.join(df[column_name]).lower()
    text = re.sub(r'[^\w\s]','',text)
    # get word lengths
    word_lengths = [len(word) for word in text.split()]
    # plot the distribution
    plt.figure(figsize=(10,6))
    plt.hist(word_lengths, bins=range(1, max(word_lengths)+2), align='left')
    plt.title('Distribution of Word Lengths in {}'.format(column_name))
    plt.xlabel('Word Length')
    plt.ylabel('Count')
    plt.show()

plot_word_length_distribution(df, 'formatted_tags')

plot_word_frequency(df, 'formatted_tags')

# Specify the name of the text column to analyze
text_column = 'formatted_tags'

binary_columns = ['Economía', 'Defensa', 'Internacional', 'América Latina', 'Neonazismo en Ucrania']

# function to plot the distribution of binary columns
def plot_binary_distribution(df, binary_columns):
    for col in binary_columns:
        counts = Counter(df[col])
        plt.bar(counts.keys(), counts.values())
        plt.title(col)
        plt.show()

plot_binary_distribution(df, binary_columns)