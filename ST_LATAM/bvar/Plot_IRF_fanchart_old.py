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

WDPATH = "D:\\AAA_Erik\\PROJECTS\\ST_LATAM\\models"
DATA = {}
OUTPUTS = {}


# Define variables + parameters
# =============================================================================
xls = {}
data = {}
output = {}
label = {}
n = "external_baseline"  # name of the model
v = 13  # number of variables
r = 22  # h horizon +2
r_2 = 85  # t estimation sample +2
c = 4  # time/lw.bound/median/up.bound

res = ["IRF"]
res_bear = ["IRF"]
colours = ["g", "b", "k", "m", "c", "r"]
country = ["ID", "IN", "KR", "MY", "PH", "TH"]
country_long = ["indonesia", "india", "korea", "malaysia", "philippines", "thailand"]
intervals = ["84", "68", "90"]


# Import data + plot + save
# =============================================================================
for k in res:
    # define storage for outputs
    data["data_"+k] = {}
    output["output_"+k] = {}
    label["label_"+k] = {}
    for i in country:
        output["output_"+k]["output_"+k+"_"+i] = {}
        label["label_"+k]["label_"+k+"_"+i] = {}
for i in range(len(country)):
    # paths
    DATA["DATA_"+str(country[i])] = os.path.join(WDPATH, "bvar_"+str(country[i]), "toolbox_4.2", "results", n, "all_intervals") 
    OUTPUTS["OUTPUTS_"+str(country[i])] = os.path.join(WDPATH, "bvar_"+str(country[i]), "toolbox_4.2", "results", n, "all_intervals") 
    os.chdir(DATA["DATA_"+str(country[i])])
    # data
    for g in range(len(intervals)): 
        xls["xls_"+str(country[i])+"_"+str(intervals[g])] = pd.ExcelFile("results_"+str(country[i])+"_"+n+"_"+str(intervals[g])+".xlsx")  
        for k, l in zip(res, res_bear):
            data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(intervals[g])] = pd.read_excel(xls["xls_"+str(country[i])+"_"+str(intervals[g])], l, index_col=0, header=0)
            data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(intervals[g])].dropna(axis=0, how="all", inplace=True)
            data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(intervals[g])].dropna(axis=1, how="all", inplace=True)
            for row in range(v):
                for col in range(v):
                    # reframe data considering different types of results
                    if pd.isna(data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(intervals[g])].iloc[r*row+1, c*col]):
                        data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(intervals[g])].iloc[r*row+1, c*col] = data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(intervals[g])].iloc[r*row, c*col]
                        data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(intervals[g])].iloc[r*row, c*col] = np.nan
            data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(intervals[g])].dropna(axis=0, how="all", inplace=True) 
            for row in range(v):
                for col in range(v):
                    # save outputs individually considering different types of results
                    output["output_"+k]["output_"+k+"_"+str(country[i])][k+"_"+str(country[i]+"_"+str(intervals[g]))+"_"+str(row+1)+"_"+str(col+1)] = pd.DataFrame(data=data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(intervals[g])].iloc[row*(r-1)+1:(r-1)*(row+1), col*c+1:c*(col+1)].values,
                                   index=data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(intervals[g])].iloc[row*(r-1)+1:(r-1)*(row+1), 0].values,
                                   columns=data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(intervals[g])].iloc[0, col*c+1:c*(col+1)].values)
                    label["label_"+k]["label_"+k+"_"+str(country[i])][k+"_"+str(country[i]+"_"+str(intervals[g]))+"_"+str(row+1)+"_"+str(col+1)] = data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(intervals[g])].iloc[(r-1)*row, c*col]
    for row in range(v):
        for col in range(v):
            # IRF
            horizon = np.arange(1, 21, dtype=float)
            bar_l_0 = [i+1 for i in range(len(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[0])+"_"+str(row+1)+"_"+str(col+1)]["median"]))]
            tick_pos_0 = [i for i in bar_l_0]
            fig, ax = plt.subplots(1, figsize=(15, 7))
            ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[0])+"_"+str(row+1)+"_"+str(col+1)]["median"], color=colours[i], linewidth=2, alpha=0.8, label="median")
            ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[0])+"_"+str(row+1)+"_"+str(col+1)]["lw. bound"], color=colours[i], alpha=0.0, label="16% credible set")
            ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[0])+"_"+str(row+1)+"_"+str(col+1)]["up. bound"], color=colours[i], alpha=0.0, label="84% credible set")
            ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[1])+"_"+str(row+1)+"_"+str(col+1)]["lw. bound"], color=colours[i], alpha=0.0, label="32% credible set")
            ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[1])+"_"+str(row+1)+"_"+str(col+1)]["up. bound"], color=colours[i], alpha=0.0, label="68% credible set")
            ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[2])+"_"+str(row+1)+"_"+str(col+1)]["lw. bound"], color=colours[i], linewidth=1, linestyle=(0, (5, 1)), alpha=0.6, label="10% credible set")
            ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[2])+"_"+str(row+1)+"_"+str(col+1)]["up. bound"], color=colours[i], linewidth=1, linestyle=(0, (5, 1)), alpha=0.6, label="90% credible set")
            ax.fill_between(horizon, np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[0])+"_"+str(row+1)+"_"+str(col+1)]["lw. bound"], dtype=float), np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[0])+"_"+str(row+1)+"_"+str(col+1)]["up. bound"], dtype=float), color=colours[i], alpha=0.1)
            ax.fill_between(horizon, np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[1])+"_"+str(row+1)+"_"+str(col+1)]["lw. bound"], dtype=float), np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[1])+"_"+str(row+1)+"_"+str(col+1)]["up. bound"], dtype=float), color=colours[i], alpha=0.1)
            ax.fill_between(horizon, np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[2])+"_"+str(row+1)+"_"+str(col+1)]["lw. bound"], dtype=float), np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[2])+"_"+str(row+1)+"_"+str(col+1)]["up. bound"], dtype=float), color=colours[i], alpha=0.1)
            ax.axhline(linewidth=2, color='k', alpha=0.6) ###
            ax.tick_params(axis="both", labelsize=18) ###
            plt.xticks(tick_pos_0, bar_l_0)
            ax.patch.set_facecolor("white")
            ax.grid(color="k", alpha=0.2, linewidth=0.5, linestyle="--")
            ax.set_ylabel("Response of "+label["label_"+res[0]]["label_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(intervals[0])+"_"+str(row+1)+"_"+str(col+1)][12:], fontsize=18)
            #ax.legend(loc="center left", fontsize="small", bbox_to_anchor=(1, 0.7), frameon=False)
            #ax.set_xlabel("horizon")
            #ax.set_ylabel("IRF")
            #ax.set_title(res[0]+": "+label["label_"+res[0]]["label_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)])
            fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)+".png"), dpi=200, bbox_inches="tight")
            fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)+".pdf"), dpi=200, bbox_inches="tight")
# end