import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



# Data Preprocessing using Python
def preprocess_data(df):
    '''
    The function processes and imputes missing data in python.
    df (DataFrame): The input dataframe
    df_new (DataFrame): The processed dataframe
    '''
    try:
        # Impute item weight with max values of different item identifier groups
        df['Item_Weight'] = df['Item_Weight'].fillna(df.groupby('Item_Identifier')['Item_Weight'].transform('max'))

        # impute missing values in outlet size with "Small"
        df['Outlet_Size'] = df['Outlet_Size'].fillna("Small")

        # delete rows which still have missing values
        df = df[df['Item_Weight'].notna()]

        # replace values
        df = df.replace({'Item_Fat_Content':{'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}})

        # feature engineering
        df['Outlet_Age']=2022 - df['Outlet_Establishment_Year']
        df = df.drop(columns=['Outlet_Establishment_Year'])

        # encode data using label encoder
        le = LabelEncoder()

        col_encode=['Item_Fat_Content','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']

        for i in col_encode:
            df[i]=le.fit_transform(df[i])

        df_new=df.drop(columns=['Item_Identifier'])
        df_new=pd.get_dummies(df_new)

    except Exception as e:
        print(e)

    else:
        return df_new

        