import nlu 

text1 = "Todo menos la paz: ¿qué pretende lograr Occidente al incitar a Ucrania a negociaciones de paz?"
text2 = "Tudo menos paz: o que o Ocidente espera conseguir incitando a Ucrânia às negociações de paz?"

glove_df1 = nlu.load('xx.embed.glove.840B_300').predict(text1)
print(glove_df1)

glove_df2 = nlu.load('xx.embed.glove.840B_300').predict(text2)
print(glove_df2)
