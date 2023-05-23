import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split # for splitting data into train and test samples
from sklearn.svm import SVC # for Support Vector Classification baseline model
from sklearn.semi_supervised import SelfTrainingClassifier # for Semi-Supervised learning
from sklearn.metrics import classification_report # for model evaluation metrics
from elmoformanylangs import Embedder

df = pd.read_csv("/Users/dwaste/Desktop/Undergrad-Thesis-Repo/transformed-data/combined_transformated_data.csv", encoding="utf=8")

df_train, df_test = train_test_split(df, test_size=0.25, random_state=0)
print('Size of train dataframe: ', df_train.shape[0])
print('Size of test dataframe: ', df_test.shape[0])

# Show target value distribution
print('Target Value Distribution:')
print(df_train['Dependents_Target'].value_counts())

# Create a scatter plot
fig = px.scatter(df_train, x='', y='', opacity=1, color=df_train['Economia'].astype(str),
                 color_discrete_sequence=['lightgrey', 'red', 'blue'],
                )

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='white', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='white', 
                 showline=True, linewidth=1, linecolor='white')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='white', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='white', 
                 showline=True, linewidth=1, linecolor='white')

# Set figure title
fig.update_layout(title_text="pro-Russian Narrative Data - Labeled vs. Unlabeled")

# Update marker size
fig.update_traces(marker=dict(size=5))

fig.show()



