
# coding: utf-8

# In[ ]:


def drop_bad_columns(df, null_portion=(1/3)):
    df.drop([column for column in df.columns if df[column].isnull().mean() >= null_portion],
            axis=1, 
            inplace=True)

def get_bad_columns(df, null_portion=(1/3)):
    return [column for column in df.columns if df[column].isnull().mean() >= null_portion]

def na_to_mode(data, column):
    modes = dict((column,data[column].mode()[0]) for column in data.columns)
    for column in data.columns:
        data[column].fillna(value=modes[column], inplace=True)

def mode_count(data, column):
    return sum((data[column] == data[column].mode()[0]).astype(int))

