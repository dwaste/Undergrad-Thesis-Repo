library(tidyverse)

# update the folder path to the correct directory on your computer
folder_path <- '/Users/dwaste/Desktop/Undergrad-Thesis-Repo/russian-ground-narrative-text-files'

read_csv_files <- function(folder_path) {
  # create an empty array
  data <- c()

  # read in each CSV file in the folder
  files <- list.files(path = folder_path, pattern = "", full.names = TRUE)

  for (file in files) {
    df <- read.csv(file)
    # append the "formatted_text" column to the data array
    data$formatted_text <-  df$formatted_text
    data$title <-  df$title
    data$link.href <-  df$link.href
    data$date <-  df$date
  }

  return(data)
}

# call the function to read in the CSV files
csv_data <- read_csv_files(folder_path)

write.csv(data, file = '/Users/dwaste/Desktop/Undergrad-Thesis-Repo/russian-ground-narrative-text-files-combined.csv')
