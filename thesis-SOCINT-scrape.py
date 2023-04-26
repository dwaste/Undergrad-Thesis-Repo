import pandas as pd
import numpy as np
import re
from Scweet.scweet import scrape
from Scweet.user import get_user_information, get_users_following, get_users_followers

# Path to env
env_path = "~\Desktop\.env.txt"

# load the data
rus_embassy_comms = pd.read_excel("russia-narrative-link-list.xlsx")

print(rus_embassy_comms)

# split link list file in lists of account names by platforms used
twit_dat_es = rus_embassy_comms[rus_embassy_comms["platform"] == "Twitter-es"]
twit_dat_pt = rus_embassy_comms[rus_embassy_comms["platform"] == "Twitter-pt"]
telegram_dat = rus_embassy_comms[rus_embassy_comms["platform"] == "Telegram"]

# coding the straggler twitter accounts
# included because not all scrapes preformed at the same time
# and technical reasons i.e. RAM overload issues on my Macbook
twit_dat_es_2 = twit_dat_es[twit_dat_es["account"].isin(["EmbajadaRusaEs", "EmbRusCuba"])]
twit_dat_pt_2 = twit_dat_pt[twit_dat_pt["account"].isin(["embaixadarussa", "sputnik_brasil"])]

# iterate function over relavent twitter accounts
#for i in twit_dat_es_2['account']:
    #twit_data_es = scrape(since="2022-01-01", until="2023-4-01", from_account = i, headless = False, lang = 'es')

for i in twit_dat_pt_2['account']:
    twit_data_pt = scrape(since="2022-01-01", until="2023-4-01", from_account = i, headless = False, lang = 'pt')
    # scrape function for one 
    # print some data to confirm scrape, no need to manually 
    # save because of Scweet package

# Telegram data was scraped from my Macbook Command Line Interface 
# all the code and CSV outputs for Telegram and Twitter scrapes can be found in this repository

