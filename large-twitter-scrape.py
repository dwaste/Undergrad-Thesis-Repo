import pandas as pd
import numpy as np
import re
from Scweet.scweet import scrape
from Scweet.user import get_user_information, get_users_following, get_users_followers

env_path = "/Users/dwaste/Desktop/.env"

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

twit_dat = scrape(words=["Ucrania"], since="2022-01-01", until="2023-4-01", headless = False, lang = 'es')
