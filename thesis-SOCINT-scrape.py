import pandas as pd
import numpy as np
import re
from Scweet.scweet import scrape
from Scweet.user import get_user_information, get_users_following, get_users_followers
from facebook_scraper import get_posts
import snscrape.modules.telegram

# Path to env
env_path = "/Users/dwaste/Desktop/.env"

# load the data
rus_embassy_comms = pd.read_excel("/Users/dwaste/Desktop/Undergrad-Thesis-Repo/russia-narrative-link-list.xlsx")

twitter_dat = rus_embassy_comms[rus_embassy_comms["platform"] == "Twitter"]
telegram_dat = rus_embassy_comms[rus_embassy_comms["platform"] == "Telegram"]
facebook_dat = rus_embassy_comms[rus_embassy_comms["platform"] == "Facebook"]

# iterate function over all twitter accounts
for i in twitter_dat['account']:
    twit_data_es = scrape(since="2022-01-01", until="2023-4-01", from_account = i, headless = False, lang = 'es')
    twit_data_pt = scrape(since="2022-01-01", until="2023-4-01", from_account = i, headless = False, lang = 'pt')



# face_data = get_posts(account = "EmbRusPan", cookies= "/Users/dwaste/Desktop/www.facebook.com_cookies.json", extra_info=True)

#for f in facebook_dat['account']:
    #face_data = get_posts(account = f, cookies= "/Users/dwaste/Downloads/www.facebook.com_cookies.txt", extra_info=True) 
    #print(face_data)
# scrape function for one 
# print some data to confirm scrape, no need to manually 
# save because of Scweet package



