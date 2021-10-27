"""
@author: Erik Andres Escayola

@title: Plotting model results

@description:
  1. Import data after model estimation (BEAR results)
  2. Reframe data to have coherent structure
  3. Generate plottings and save in suitable format for publication
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm

WDPATH = "C:\\Users\\eandr\\Documents\\1_PROJECTS\\A_PAPERS\\ST_LATAM\\models"
DATA = {}
OUTPUTS = {}

# Refactoring to do's:
# definining an instance variable(e.g. x) doing all the loops and only at the end save it in the corresponding dictionary location

# Define variables + parameters
# =============================================================================
xls = {}
data = {}
output = {}
label = {}
# n = "prelim_loc_press"  # name of the model version
v = 5  # number of variables
v_2 = 5 + 1  # extra variable since exogenous part in HD
r = 40 + 2  # h horizon +2
s = 68  # full sample
c = 4  # time/lw.bound/median/up.bound

lag = ["lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7", "lag8"]
tension = ["lassi", "lapsi", "epu"]
press = ["loc", "esp", "int"]

res = ["IRF"]  # , "FEVD", "HD"]
res_bear = ["IRF"]  # , "FEVD", "hist decomposition"]
country = ["LA"]
country_long = ["latin-america"]
colours = ["g", "b", "k", "m", "c", "r", "y", "k"]
# palettes = ["YlGn", "PuBu", "bone_r", "RdPu", "BrBG", "YlOrRd"]


# Define placeholders
# =============================================================================
for k in res:
    data["data_"+k] = {}
    output["output_"+k] = {}
    label["label_"+k] = {}
    for i in press:
        output["output_"+k]["output_"+k+"_"+i] = {}
        label["label_"+k]["label_"+k+"_"+i] = {}

# Define paths
# =============================================================================
for i in press:
    DATA["DATA_"+country[0]+"_"+i] = os.path.join(WDPATH,
                                                  "bvar_"+country[0],
                                                  "toolbox_4.2",
                                                  "results",
                                                  "prelim_"+i+"_"+"press")
    OUTPUTS["OUTPUTS_"+country[0]+"_"+i] = os.path.join(WDPATH,
                                                        "bvar_"+country[0],
                                                        "toolbox_4.2",
                                                        "results",
                                                        "prelim_"+i+"_"+"press")


# Load data and reframe it
# =============================================================================
for i in range(len(press)):
    # change directory since data stored in different folders
    os.chdir(DATA["DATA_"+country[0]+"_"+press[i]])
    for j in range(len(tension)):
        for k in range(len(lag)):
            # data loading and cleaning
            xls["xls_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]] = pd.ExcelFile("results_prelim_"+tension[j]+"_"+press[i]+"_"+"macro_"+lag[k]+".xlsx")              
            for m, l in zip(res, res_bear):
                data["data_"+m]["data_"+m+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]] = pd.read_excel(xls["xls_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]], l, index_col=0, header=0)
                data["data_"+m]["data_"+m+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]].dropna(axis=0, how="all", inplace=True)
                data["data_"+m]["data_"+m+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]].dropna(axis=1, how="all", inplace=True)
                # save reframed data
                for row in range(v):                   
                    for col in range(v):
                        if pd.isna(data["data_"+m]["data_"+m+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]].iloc[r*row+1, c*col]):
                            data["data_"+m]["data_"+m+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]].iloc[r*row+1, c*col] = data["data_"+m]["data_"+m+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]].iloc[r*row, c*col]
                            data["data_"+m]["data_"+m+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]].iloc[r*row, c*col] = np.nan
                data["data_"+m]["data_"+m+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]].dropna(axis=0, how="all", inplace=True)             
                # save data for plotting
                for row in range(v):
                    for col in range(v):
                        output["output_"+m]["output_"+m+"_"+press[i]][m+"_"+tension[j]+"_"+lag[k]+"_"+str(row+1)+"_"+str(col+1)] = pd.DataFrame(data=data["data_"+m]["data_"+m+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]].iloc[row*(r-1)+1:(r-1)*(row+1), col*c+1:c*(col+1)].values,
                                       index=data["data_"+m]["data_"+m+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]].iloc[row*(r-1)+1:(r-1)*(row+1), 0].values,
                                       columns=data["data_"+m]["data_"+m+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]].iloc[0, col*c+1:c*(col+1)].values)
                        label["label_"+m]["label_"+m+"_"+press[i]][m+"_"+tension[j]+"_"+lag[k]+"_"+str(row+1)+"_"+str(col+1)] = data["data_"+m]["data_"+m+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+lag[k]].iloc[(r-1)*row, c*col]

# Plot IRFs for different lag order
# =============================================================================
for i in range(len(press)):  
    for j in range(len(tension)):
        for row in range(v):
            for col in range(v):
                # IRF
                horizon = np.arange(1, 41, dtype=float)
                bar_l_0 = [q+1 for q in range(0,40)]
                # tick_pos_0 = [q+1 for q in range(0,40)]
                fig, ax = plt.subplots(1, figsize=(15, 7))
                ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+press[i]][res[0]+"_"+tension[j]+"_"+lag[0]+"_"+str(row+1)+"_"+str(col+1)]["median"], color=colours[0], linewidth=2, alpha=0.8, label="p=1")
                ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+press[i]][res[0]+"_"+tension[j]+"_"+lag[1]+"_"+str(row+1)+"_"+str(col+1)]["median"], color=colours[1], linewidth=2, alpha=0.8, label="p=2")
                ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+press[i]][res[0]+"_"+tension[j]+"_"+lag[2]+"_"+str(row+1)+"_"+str(col+1)]["median"], color=colours[2], linewidth=2, alpha=0.8, label="p=3")
                ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+press[i]][res[0]+"_"+tension[j]+"_"+lag[3]+"_"+str(row+1)+"_"+str(col+1)]["median"], color=colours[3], linewidth=2, alpha=0.8, label="p=4")
                ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+press[i]][res[0]+"_"+tension[j]+"_"+lag[4]+"_"+str(row+1)+"_"+str(col+1)]["median"], color=colours[4], linewidth=2, alpha=0.8, label="p=5")
                ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+press[i]][res[0]+"_"+tension[j]+"_"+lag[5]+"_"+str(row+1)+"_"+str(col+1)]["median"], color=colours[5], linewidth=2, alpha=0.8, label="p=6")
                ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+press[i]][res[0]+"_"+tension[j]+"_"+lag[6]+"_"+str(row+1)+"_"+str(col+1)]["median"], color=colours[6], linewidth=2, alpha=0.8, label="p=7")
                ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+press[i]][res[0]+"_"+tension[j]+"_"+lag[7]+"_"+str(row+1)+"_"+str(col+1)]["median"], color=colours[7], linewidth=2, alpha=0.4, label="p=8")
                # ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["lw. bound"], color=colours[i], linewidth=1, linestyle=(0, (5, 1)), alpha=0.6, label="16% credible set")
                # ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["up. bound"], color=colours[i], linewidth=1, linestyle=(0, (5, 1)), alpha=0.6, label="84% credible set")
                # ax.fill_between(horizon, np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["lw. bound"], dtype=float), np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["up. bound"], dtype=float), color=colours[i], alpha=0.2)
                ax.axhline(linewidth=2, color='k', alpha=0.6) ###
                ax.tick_params(axis="both", labelsize=18) ###
                #plt.xticks(tick_pos_0, tick_pos_0)
                ax.patch.set_facecolor("white")
                ax.grid(color="k", alpha=0.2, linewidth=0.5, linestyle="--")
                #ax.set_ylabel("Response of "+label["label_"+res[0]]["label_"+res[0]+"_"+press[i]][res[0]+"_"+tension[j]+"_"+lag[k]+"_"+str(row+1)+"_"+str(col+1)][12:], fontsize=18)                      
                ax.legend(loc="center left", fontsize=18, bbox_to_anchor=(1, 0.7), frameon=False)
                #ax.set_xlabel("horizon")
                #ax.set_ylabel("IRF")
                ax.set_title(res[0]+": "+label["label_"+res[0]]["label_"+res[0]+"_"+press[i]][res[0]+"_"+tension[j]+"_"+lag[0]+"_"+str(row+1)+"_"+str(col+1)], fontsize=20)
                fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+country[0]+"_"+press[i]], res[0]+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+str(row+1)+"_"+str(col+1)+".png"), dpi=200, bbox_inches="tight")
                # fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+country[0]+"_"+press[i]], res[0]+"_"+country[0]+"_"+tension[j]+"_"+press[i]+"_"+str(row+1)+"_"+str(col+1)+".pdf"), dpi=200, bbox_inches="tight")