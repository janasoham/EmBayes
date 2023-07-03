#### This code takes the hockey goal CSV files as input and separates the columns that include goas, positions, names etc. ###################


import pandas as pd;
import numpy as np;
import scipy as sp;
import matplotlib.pyplot as plt;



def read_hock(filename):
    df = pd.read_csv(filename);
    dfx = df[['Unnamed: 1', 'Scoring.1']]; #Goals
    dfx = dfx[1:]; # Delete first spurious row
    dfx.columns = ['name', 'goals'];
    dfx.set_index('name');
    return dfx;

def read_hock_position(filename,position):
    df = pd.read_csv(filename);
    dfx = df[['Unnamed: 1', 'Scoring.1', 'Unnamed: 4']]; #Goals
    dfx = dfx[1:]; # Delete first spurious row
    dfx.columns = ['name', 'goals', 'position'];
    dfx.set_index('name');
    if position=="winger":
        dfx = dfx.loc[(dfx["position"] == "LW") + (dfx["position"] == "RW")+(dfx["position"] == "W")]
    if position == "center":
        dfx = dfx.loc[(dfx["position"] == "C")]
    if position == "defender":
        dfx = dfx.loc[(dfx["position"] == "D")]
    del dfx["position"];
    return dfx;

def hockey_data(file1, file2):
    df1x = read_hock(file1);
    df2x = read_hock(file2);

    df = pd.merge(df1x,df2x, on='name', suffixes=('_past','_future'));
    G = np.asarray(df[['goals_past', 'goals_future']].astype('int32'));
    N = G.shape[0];

    gpast = G[:,0]; gfut = G[:,1];
    PX = {};
    for i in gpast:
        if i in PX:
            PX[i] += 1;
        else:
            PX[i] = 1;

    return (PX, gpast, gfut);

def hockey_data_position(file1, file2, position):

    df1x = read_hock_position(file1, position);
    df2x = read_hock_position(file2, position);


    df = pd.merge(df1x,df2x, on='name', suffixes=('_past','_future'));
    G = np.asarray(df[['goals_past', 'goals_future']].astype('int32'));
    N = G.shape[0];

    gpast = G[:,0]; gfut = G[:,1];
    PX = {};
    for i in gpast:
        if i in PX:
            PX[i] += 1;
        else:
            PX[i] = 1;

    return (PX, gpast, gfut);
