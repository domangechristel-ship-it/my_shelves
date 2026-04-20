import pandas as pd

def get_books(lang='ENG', nrows=None):
    '''
    get books
    '''
    file_name = f'../../data/books_{lang}_mini.csv'
    if nrows is None:
        return pd.read_csv(file_name)
    return pd.read_csv(file_name, nrows=nrows)

def get_reviews(lang='ENG', nrows=None):
    '''
    get reviews
    '''

    file_name = f'../../data/reviews_{lang}_mini.csv'
    if nrows is None:
        return pd.read_csv(file_name)
    return pd.read_csv(file_name, nrows=nrows)
