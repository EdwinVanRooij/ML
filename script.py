import pandas as pd
import numpy as np

pd.set_option('display.notebook_repr_html', False)

df = pd.read_csv('data.csv', sep=",", quotechar='"')
print(df.loc[0:7, [
    'What type of chocolate do you prefer?',
    "What is your gender?",
]])
