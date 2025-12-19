import math

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.model_selection import train_test_split

from config import Config


class LettuceDataset:
    def __init__(self, datapath, split):
        self.datapath = datapath
        self.df = load_csv(self.datapath)
        self.split = split
    
    def preprocess(self, seed=42):
        if self.split == "train":
            self.df = self.df.drop_duplicates()
            self.df = extract_province_city(self.df)
            
            #clip outliers
            cap_area = self.df['Area'].quantile(0.95)
            self.df['Area'] = self.df['Area'].clip(upper=cap_area)
            cap_frontage = self.df['Frontage'].quantile(0.95)
            self.df['Frontage'] = self.df['Frontage'].clip(upper=cap_frontage)
            
            #change categorical data to dummy
            legal_dummies = pd.get_dummies(self.df['Legal status'], prefix='legal', drop_first=True)
            province_dummies = pd.get_dummies(self.df['province_city'], prefix='prov', drop_first=True)
            
            #log area and price
            self.df['log_area'] = np.log(self.df['Area'])
            self.df['log_price'] = np.log(self.df['Price'])
            
            #drop unecessary/can't be used columns
            self.df = self.df.drop(['Balcony direction', 'House direction', 'Furniture state', 'Access Road', 'Address', 'Address_clean'], axis=1)
            
            X_numeric = self.df[['log_area', 'Frontage', 'Floors', 'Bedrooms', 'Bathrooms', 'Frontage_missing']]
            X = pd.concat([X_numeric, legal_dummies, province_dummies], axis=1)
            y = self.df['log_price']
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

            X_train_transformed = X.loc[X_train.index, :]
            X_val_transformed = X.loc[X_val.index, :]

            return X_train_transformed, X_val_transformed, y_train, y_val
        
def extract_province_city(df, address_col='Address'):
    #clean initials
    df['Address_clean'] = (
        df[address_col]
        .str.lower()
        .str.replace('thành phố ', '', regex=False)
        .str.replace('tp. ', '', regex=False)
        .str.replace('tp ', '', regex=False)
        .str.replace('tỉnh ', '', regex=False)
    )
    #split from the right
    address_parts = df['Address_clean'].str.rsplit(',', n=2, expand=True)
    #extract province/city
    df['province_city'] = address_parts[2].str.strip()
    df['province_city'] = df['province_city'].fillna('Unknown')
    #return df
    return df

def load_csv(file_path, encoding="latin-1"):
    df = pd.read_csv(file_path, encoding=encoding)
    return df