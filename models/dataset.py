import math

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.model_selection import train_test_split

import sys
import os

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
        
    def visualize(self, chart_type, features=None, title=None):
        figures = list()
        if chart_type == "corr":
            correlations_df = self.df.drop(['Area', 'Price', 'Legal status', 'province_city'], axis=1)
            corr = correlations_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            figures.append(
                go.Figure(go.Heatmap(z=corr.mask(mask), x=corr.columns, y=corr.columns, colorscale = 'Viridis'))
            )
            return plot_from_trace(figures, rows=len(figures), cols=1, vertical_spacing=0.5, title=title)
        
        elif chart_type == "dist":
            for feature in features:
                figures.append(go.Figure(go.Histogram(x=self.df[feature], name=feature,
                                       marker=dict(line=dict(width=1, colorscale='Viridis')),
                )))
            return plot_from_trace(figures, rows=math.ceil(len(figures)), title=title)
        
        elif chart_type == "bar":
            df = self.df.drop(["Date"], axis=1)
            df = df.corr()["Growth Days"].drop("Growth Days")
            largest_value = abs(df).max()
            colors = ['lightslategray' if abs(df[i]) != largest_value else 'crimson' for i in df.index]
            figures.append(
                go.Figure(go.Bar(x=df, y=df.index, orientation="h", marker_color=colors))
            )
            return plot_from_trace(figures, title=title)
        
        elif chart_type == "scatter":
            figures.append(
                go.Figure(go.Scatter(x=self.df[features[0]], y=self.df[features[1]],
                                     mode="markers",
                                     marker=dict(
                                         colorscale='Inferno',
                                         color=self.df["Plant_ID"],
                                        )
                                     ))
            )
            return plot_from_trace(figures, title=title, xaxis_title=features[0], yaxis_title=features[1])
    
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

def plot_from_trace(figures, rows=1, cols=1, title=None, xaxis_title=None, yaxis_title=None, zaxis_title=None, scene=None, **kwargs):
    fig = sp.make_subplots(rows=rows, cols=cols, **kwargs)
    for i, figure in enumerate(figures):
        row = (i//cols) + 1
        col = (i % cols) + 1
        for trace in figure["data"]:
            fig.append_trace(trace, row=row, col=col)
    
    fig.update_layout(
        title={
            'text': title,
        },
        autosize=False,
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black',
                  size=15),
        scene = scene,
        )
    
    if xaxis_title:
        fig.update_xaxes(title=xaxis_title)

    if yaxis_title:
        fig.update_yaxes(title=yaxis_title)
        
    return fig