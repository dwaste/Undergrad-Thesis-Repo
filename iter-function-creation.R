library(tidyverse)
library(readxl)

rus_embassy_comms <- read_excel("/Users/dwaste/Desktop/Undergrad-Thesis-Repo/russia-narrative-link-list.xlsx")

r <- rus_embassy_comms %>%
    filter(platform == "Twitter")

i = 0

for(i in r$account) {
    print(i)}

    lang= c("es","pt")