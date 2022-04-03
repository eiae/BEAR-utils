"""
@author: Erik Andres Escayola

@title: Plotting country characteristics

@description:
  1. Import country characteristics data
  2. Reframe data to have coherent structure
  3. Generate plottings and save in suitable format for publication
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import pi

WDPATH = "C:\\Users\\eandr\\Documents\\1_UNIVERSITY\\A_MANNHEIM\\1_Notes\\A_TFM\\4_Model\\Model\\characteristics"
DATA = os.path.join(WDPATH, "data")
OUTPUTS = os.path.join(WDPATH, "output")


# Define variables + parameters
# =============================================================================
comparison_rad = {}
comparison_col = {}
values = {}
angles = {}
bars = []   
pos = [] 

d = 5  # number of indexes
e = 6  # number of countries
barwidth = 0.15

crises_periods = ["1997-2000", "2001-2006", "2007-2012", "2013-2016"]
categories = ["ERS", "MPI", "FDV", "CCP", "MPP"]
categories_long = ["Exchange rate stability", "Monetary policy independence", "Financial development", "Capital control policies", "Macroprudential policies"]
colours = ["g", "b", "k", "m", "c", "r"]
colours_fill = [(0, 0.5, 0, 0.2), (0, 0, 1, 0.2), (0, 0, 0, 0.2), (1, 0, 1, 0.2), (0, 0.93, 0.93, 0.2), (1, 0, 0, 0.2)]
country = ["ID", "IN", "KR", "MY", "PH", "TH"]
country_title = ["Indonesia", "India", "South Korea", "Malaysia", "The Philippines", "Thailand"]


# Import data + prepare indexes + plot + save
# =============================================================================
xls = pd.ExcelFile(os.path.join(DATA, "asia_characteristics.xlsx"))
data_rad = pd.read_excel(xls, "characteristics_radar", index_col=1, header=1)
data_col = pd.read_excel(xls, "characteristics_column", index_col=1, header=1)
data_rad.drop(columns=data_rad.columns[0], inplace=True)
data_col.drop(columns=data_col.columns[0], inplace=True)

summary = data_rad.describe()
mean = summary.loc["mean"]

for i in range(1, len(country)+1):
    comparison_rad["indexes"+"_"+str(country[i-1])] = mean.iloc[d*i-d:d*i]
for i in range(1, len(categories)+1):
    comparison_col["indexes"+"_"+str(categories[i-1])] = data_col.iloc[:, e*i-e:e*i]
for i in range(len(country)):
    # Radarplot
    values["values_"+str(country[i])] = comparison_rad["indexes_"+str(country[i])].values.tolist()
    values["values_"+str(country[i])] += values["values_"+str(country[i])][:1]
    angles["angles_"+str(country[i])] = []
    for j in range(d):
        angles["angles_"+str(country[i])].append(j/float(d)*2*pi)
    angles["angles_"+str(country[i])] += angles["angles_"+str(country[i])][:1]   
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True) 
    plt.xticks(angles["angles_"+str(country[i])][:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey")
    plt.ylim(0, 1)
    ax.plot(angles["angles_"+str(country[i])], values["values_"+str(country[i])], linewidth=2, linestyle='-', color=colours[i], alpha=0.8)
    ax.fill(angles["angles_"+str(country[i])], values["values_"+str(country[i])], color=colours[i], alpha=0.2)
    ax.set_title(country_title[i])  
    fig.savefig(os.path.join(OUTPUTS, "characteristics_"+str(country[i])+".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUTS, "characteristics_"+str(country[i])+".pdf"), dpi=200, bbox_inches="tight")
for i in range(len(categories)):
    # Barplot  
    fig, ax = plt.subplots(1, figsize=(15, 7))
    for j in range(len(country)):
        bars.append([comparison_col["indexes"+"_"+str(categories[i])].iloc[0:4, j].mean(), comparison_col["indexes"+"_"+str(categories[i])].iloc[4:10, j].mean(), comparison_col["indexes"+"_"+str(categories[i])].iloc[10:16, j].mean(), comparison_col["indexes"+"_"+str(categories[i])].iloc[16:20, j].mean()])
        pos.append(j*barwidth+np.arange(len(crises_periods)))
        ax.bar(pos[j+(i*len(country))], bars[j+(i*len(country))], color=colours_fill[j], width=barwidth, linewidth=2, edgecolor=colours[j], label=country[j])
    plt.ylim(0, 1)
    ax.tick_params(axis="both", labelsize=18)
    plt.xticks([k + ((-0.075+0.825)/2) for k in range(len(crises_periods))], [crises_periods[k] for k in range(len(crises_periods))])
    ax.patch.set_facecolor("white")
    ax.grid(color="k", alpha=0.2, linewidth=0.5, linestyle="--")
    ax.legend(loc="center left", fontsize="x-large", bbox_to_anchor=(1, 0.8), frameon=False)
    ax.set_ylabel(str(categories_long[i]), fontsize=18)
    #ax.set_ylabel("index")
    #ax.set_title(str(categories_long[i]))
    fig.savefig(os.path.join(OUTPUTS, "characteristics_"+str(categories[i])+".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUTS, "characteristics_"+str(categories[i])+".pdf"), dpi=200, bbox_inches="tight")
# end