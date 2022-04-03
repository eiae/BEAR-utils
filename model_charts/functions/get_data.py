# %% Data

# Import packages
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import and wrangle data
# =============================================================================
xls = {}
data = {}
output = {}
label = {}

def getData(cc, DATA, res, res_bear, v, vv, t, cv, h, c):
    for i in res:
        data[i] = {}
        output[i] = {}
        label[i] = {}
        for j in cc:
            output[i][j] = {}
            label[i][j] = {}
    for i in cc:
        os.chdir(DATA[i])
        xls[i] = pd.ExcelFile("results_BVAR_"+i+".xlsx")
    
        for j, k in zip(res, res_bear):
            data[j][i] = pd.read_excel(xls[i], k, index_col=0, header=0)
            data[j][i].dropna(axis=0, how="all", inplace=True)
            data[j][i].dropna(axis=1, how="all", inplace=True)
    
            if j == "HD":
                for row in range(v):
                    for col in range(vv):
                        if pd.isna(data[j][i].iloc[t*row, cv*col+1]):
                            data[j][i].iloc[t*row, cv*col+1] = "median"
            else:
                for row in range(v):
                    for col in range(v):
                        if pd.isna(data[j][i].iloc[h*row+1, c*col]):
                            data[j][i].iloc[h*row+1, c*col] = data[j][i].iloc[h*row, c*col]
                            data[j][i].iloc[h*row, c*col] = np.nan
            data[j][i].dropna(axis=0, how="all", inplace=True)
    
            if j == "HD":
                for row in range(v):
                    for col in range(vv):
                        df =  pd.DataFrame(data=data[j][i].iloc[row*t+1:t*(row+1), col*cv+1:cv*(col+1)].values,
                                           index=data[j][i].iloc[row*t+1:t*(row+1), 0].values,
                                           columns=data[j][i].iloc[0, col*cv+1:cv*(col+1)].values)
                        lbl = data[j][i].iloc[t*row, cv*col]
                        srow = str(row+1)
                        scol = str(col+1)
                        output[j][i][j+"_"+srow+"_"+scol] = df
                        label[j][i][j+"_"+srow+"_"+scol] = lbl
            else:
                for row in range(v):
                    for col in range(v):
                        df = pd.DataFrame(data=data[j][i].iloc[row*(h-1)+1:(h-1)*(row+1), col*c+1:c*(col+1)].values,
                                          index=data[j][i].iloc[row*(h-1)+1:(h-1)*(row+1), 0].values,
                                          columns=data[j][i].iloc[0, col*c+1:c*(col+1)].values)
                        lbl = data[j][i].iloc[(h-1)*row, c*col]
                        srow = str(row+1)
                        scol = str(col+1)
                        output[j][i][j+"_"+srow+"_"+scol] = df
                        label[j][i][j+"_"+srow+"_"+scol] = lbl
                        
    return data, df, label, output
