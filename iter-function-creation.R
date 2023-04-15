library(tidyverse)
library(readxl)
library(jsonlite)

# Read in json file
json <- fromJSON('/Users/dwaste/Desktop/Undergrad-Thesis-Repo/telegram-scrape-output/EmbajadaRusa_CR/EmbajadaRusa_CR_messages.json')

# Extract necessary data and combine into data frame
data <- data.frame()
for (i in 1:length(json['chats'])) {
  temp_data <- data.frame(
    username = json$messages[[i]]['username'],
    message = sapply(json$messages, function(x) x['message']),
    date = sapply(json$messages, function(x) x['date'])
  )
  data <- rbind(data, temp_data)
}

json$messages

print(json$messages[[1]]$username)

#write.csv(data, "/Users/dwaste/Desktop/Undergrad-Thesis-Repo/russian-ground-narrative-text-files/{username}.csv", row.names = FALSE)
