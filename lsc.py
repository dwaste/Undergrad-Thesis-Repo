import json

# Read in the first file
with open('/Users/dwaste/Desktop/positive_words_es.txt', mode ='r', encoding='utf-8') as file1:
    data1 = file1.read().split('\n')
    
# Read in the second file
with open('/Users/dwaste/Desktop/negative_words_es.txt', mode = 'r', encoding='utf-8') as file2:
    data2 = file2.read().split('\n')
    
# Create a dictionary object with the pieces from both files
# Create a dictionary object with the pieces from both files
data_dict = {
    'positive': [{'text': piece, 'sentiment': 1} for piece in data1 if piece],
    'negative': [{'text': piece, 'sentiment': -1} for piece in data2 if piece]
}

# Write the dictionary object to a JSON file
with open('/Users/dwaste/Desktop/Undergrad-Thesis-Repo/es-lsd.json', mode = 'w', encoding='utf-8') as outfile:
    json.dump(data_dict, outfile, ensure_ascii=False)


