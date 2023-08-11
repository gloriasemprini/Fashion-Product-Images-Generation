import pandas as pd

def getWatches(df):
    list = []
    for i, type in enumerate(df['articleType']):
        if(type == "Watches"):
           list.append(df.iloc[i])
    return pd.DataFrame(list)
