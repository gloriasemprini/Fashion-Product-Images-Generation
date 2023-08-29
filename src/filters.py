import pandas as pd

def getWatches(df):
    list = []
    for i, type in enumerate(df['articleType']):
        if(type == "Watches"):
           list.append(df.iloc[i])
    return pd.DataFrame(list)

def get_dataframe_by_article_type(df, article_type):
    list = []
    for i, type in enumerate(df['articleType']):
        if(type == article_type):
           list.append(df.iloc[i])
    return pd.DataFrame(list)