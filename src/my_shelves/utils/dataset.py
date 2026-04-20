import pandas as pd

def get_books(lang='ENG', nrows=None):
    file_name = f'../../data/books_{lang}_mini.csv'
    if(nrows==None):
        return pd.read_csv(file_name)
    else:
        return pd.read_csv(file_name, nrows=nrows)

def get_reviews(lang='ENG', nrows=None):
    file_name = f'../../data/reviews_{lang}_mini.csv'
    if(nrows==None):
        return pd.read_csv(file_name)
    else:
        return pd.read_csv(file_name, nrows=nrows)
