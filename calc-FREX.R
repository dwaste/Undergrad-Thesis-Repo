library(tidyverse)
library(LSX)
library(quanteda)
library(corpus)
library(quanteda.textplots)
library(jsonlite)

sputnik_data <- read.csv("/Users/dwaste/Desktop/Undergrad-Thesis-Repo/transformed-data/sputnik-data-transformed.csv")

dict <- dictionary(file = "/Users/dwaste/Desktop/Undergrad-Thesis-Repo/lss-dictonary.yml")

corp <- sputnik_data %>%
  corpus_frame(sputnik_data, text = "formatted_text")

toks <- tokens(corp$formatted_text, remove_punct = TRUE, remove_symbols = TRUE, 
               remove_numbers = TRUE, remove_url = TRUE)

dfmt <- dfm(toks) %>%
    dfm_remove(stopwords("es"))

es_pos <- list(read.delim("/Users/dwaste/Desktop/positive_words_es.txt"))

es_neg <- list(read.delim("/Users/dwaste/Desktop/negative_words_es.txt"))

es_lsd <- read_json("/Users/dwaste/Desktop/Undergrad-Thesis-Repo/es-lsd.json")

seed <- as.seedwords(es_lsd)
term <- char_context(toks, pattern = dict$Economía, p = 0.01)
lss <- textmodel_lss(dfmt, seeds = seed, terms = term, cache = TRUE, 
                     include_data = TRUE, group_data = TRUE)

textplot_terms(lss)













