"""
@author: Erik Andres Escayola

@title: Plotting forecasting results

@description:
  1. Import data after model estimation (BEAR results)
  2. Reframe data to have coherent structure
  3. Generate plottings and save in suitable format for publication
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib import cm

WDPATH = "C:\\Users\\eandr\\Documents\\1_UNIVERSITY\\A_MANNHEIM\\1_Notes\\A_TFM\\4_Model\\Model"
DATA = {}
OUTPUTS = {}


# Define variables + parameters
# =============================================================================
xls = {}
data = {}
output = {}
label = {}
n = "slm_sv_counterfactual"  # name of the model
v = 13  # number of variables
r = 81  # estimation sample + forecast horizon +2
c = 5  # time/sample/lw.bound/median/up.bound

res = ["UFOR", "CFOR"]
res_bear = ["forecasts", "cond forecasts"]
colours = ["g", "b", "k", "m", "c", "r"]
palettes = ["YlGn", "PuBu", "bone_r", "RdPu", "BrBG", "YlOrRd"]
country = ["ID", "IN", "KR", "MY", "PH", "TH"]
country_long = ["indonesia", "india", "korea", "malaysia", "philippines", "thailand"]
country_global = ["SSR_actual_long", "SSR_long", "VIX_actual", "VIX"]

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
    DATA["DATA_"+str(country[i])] = os.path.join(WDPATH, "bvar_"+str(country[i]), "toolbox_5.0", "results", n)
    OUTPUTS["OUTPUTS_"+str(country[i])] = os.path.join(WDPATH, "bvar_"+str(country[i]), "toolbox_5.0", "results", n)
    os.chdir(DATA["DATA_"+str(country[i])])
    # data
    for g in range(len(country_global)):
        xls["xls_"+str(country[i])+"_"+str(country_global[g])] = pd.ExcelFile("results_"+str(country[i])+"_"+n+"_"+str(country_global[g])+".xlsx")
        for k, l in zip(res, res_bear):
            data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(country_global[g])] = pd.read_excel(xls["xls_"+str(country[i])+"_"+str(country_global[g])], l, index_col=0, header=0)
            data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(country_global[g])].dropna(axis=0, how="all", inplace=True)
            data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(country_global[g])].dropna(axis=1, how="all", inplace=True)
            for row in range(1):
                for col in range(v):
                    # reframe data considering different types of results
                    if pd.isna(data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(country_global[g])].iloc[r*row+1, c*col]):
                        data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(country_global[g])].iloc[r*row+1, c*col] = data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(country_global[g])].iloc[r*row, c*col]
                        data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(country_global[g])].iloc[r*row, c*col] = np.nan
            data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(country_global[g])].dropna(axis=0, how="all", inplace=True) 
            for row in range(1):
                for col in range(v):
                    # save outputs individually considering different types of results
                    output["output_"+k]["output_"+k+"_"+str(country[i])][k+"_"+str(country[i])+"_"+str(country_global[g])+"_"+str(col+1)] = \
                    pd.DataFrame(data=data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(country_global[g])].iloc[row*(r-1)+1:(r-1)*(row+1), col*c+1:c*(col+1)].values,
                                   index=data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(country_global[g])].iloc[row*(r-1)+1:(r-1)*(row+1), 0].values,
                                   columns=data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(country_global[g])].iloc[0, col*c+1:c*(col+1)].values)
                    label["label_"+k]["label_"+k+"_"+str(country[i])][k+"_"+str(country[i])+"_"+str(country_global[g])+"_"+str(col+1)] = data["data_"+k]["data_"+k+"_"+str(country[i])+"_"+str(country_global[g])].iloc[(r-1)*row, c*col]
    for col in range(v):
        # Counterfactuals for SSR (UFOR and CFOR)
        bar_l_0 = [i for i in range(len(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(country_global[0])+"_"+str(col+1)]["median"][51:]))]  # plot only from 2012 onwards
        tick_pos_0 = [i for i in bar_l_0]
        fig, ax = plt.subplots(1, figsize=(15, 7))
        ax.plot(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(country_global[0])+"_"+str(col+1)]["sample"][51:], color=colours[i], linewidth=5, alpha=0.8, label="sample")
        ax.plot(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(country_global[0])+"_"+str(col+1)]["median"][51:], color=colours[i], linewidth=5, linestyle="--", alpha=0.8, label="unconditional forecast")
        #ax.plot(output["output_"+res[1]]["output_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(country_global[0])+"_"+str(col+1)]["median"][51:], color=colours[i], linewidth=5, linestyle="-.", alpha=0.8, label="scenario of actual SSR path")        
        ax.plot(output["output_"+res[1]]["output_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(country_global[1])+"_"+str(col+1)]["median"][51:], color=colours[i], linewidth=5, linestyle=":", alpha=0.8, label="scenario of tighter Fed & BoJ MP")
        #ax.plot(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(col+1)]["lw. bound"], color=colours[i], linewidth=1, linestyle=(0, (5, 1)), alpha=0.6, label="16% credible set")
        #ax.plot(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(col+1)]["up. bound"], color=colours[i], linewidth=1, linestyle=(0, (5, 1)), alpha=0.6, label="84% credible set")
        #ax.fill_between(horizon, np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(col+1)]["lw. bound"], dtype=float), np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(col+1)]["up. bound"], dtype=float), color=colours[i], alpha=0.2)
        ax.tick_params(axis="both", labelsize=18)
        plt.xticks(np.arange(0, len(tick_pos_0) , 4), [output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(country_global[0])+"_"+str(col+1)].index[51+j][:-2] for j in np.arange(0, len(tick_pos_0), 4)], rotation='vertical')
        ax.patch.set_facecolor("white")
        ax.grid(color="k", alpha=0.2, linewidth=0.5, linestyle="--")
        ax.legend(loc="center left", fontsize="x-large", bbox_to_anchor=(1, 0.8), frameon=False)
        ax.set_ylabel("Forecast of "+label["label_"+res[1]]["label_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(country_global[0])+"_"+str(col+1)][23:], fontsize=18)
        #ax.set_ylabel("variable", fontsize=15)
        #ax.set_title("Counterfactual: "+label["label_"+res[1]]["label_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(country_global[0])+"_"+str(col+1)][23:], fontdict = {"fontsize" : 20})  # positions to isolate title from string in labels
        fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[0]+"_"+res[1]+"_"+str(country[i])+"_"+str(col+1)+"_"+"SSR"+".png"), dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[0]+"_"+res[1]+"_"+str(country[i])+"_"+str(col+1)+"_"+"SSR"+".pdf"), dpi=200, bbox_inches="tight")
    for col in range(v):
        # Counterfactuals for VIX (UFOR and CFOR)
        bar_l_1 = [i for i in range(len(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(country_global[2])+"_"+str(col+1)]["median"][51:]))]  # plot only from 2012 onwards
        tick_pos_1 = [i for i in bar_l_1]
        fig, ax = plt.subplots(1, figsize=(15, 7))
        ax.plot(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(country_global[2])+"_"+str(col+1)]["sample"][51:], color=colours[i], linewidth=5, alpha=0.8, label="sample")
        ax.plot(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(country_global[2])+"_"+str(col+1)]["median"][51:], color=colours[i], linewidth=5, linestyle="--", alpha=0.8, label="unconditional forecast")
        #ax.plot(output["output_"+res[1]]["output_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(country_global[2])+"_"+str(col+1)]["median"][51:], color=colours[i], linewidth=5, linestyle="-.", alpha=0.8, label="scenario of actual VIX path")        
        ax.plot(output["output_"+res[1]]["output_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(country_global[3])+"_"+str(col+1)]["median"][51:], color=colours[i], linewidth=5, linestyle=":", alpha=0.8, label="scenario of tighter VIX")
        #ax.plot(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(col+1)]["lw. bound"], color=colours[i], linewidth=1, linestyle=(0, (5, 1)), alpha=0.6, label="16% credible set")
        #ax.plot(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(col+1)]["up. bound"], color=colours[i], linewidth=1, linestyle=(0, (5, 1)), alpha=0.6, label="84% credible set")
        #ax.fill_between(horizon, np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(col+1)]["lw. bound"], dtype=float), np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(col+1)]["up. bound"], dtype=float), color=colours[i], alpha=0.2)
        ax.tick_params(axis="both", labelsize=18)
        plt.xticks(np.arange(0, len(tick_pos_1) , 4), [output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(country_global[2])+"_"+str(col+1)].index[51+j][:-2] for j in np.arange(0, len(tick_pos_1), 4)], rotation='vertical')
        ax.patch.set_facecolor("white")
        ax.grid(color="k", alpha=0.2, linewidth=0.5, linestyle="--")
        ax.legend(loc="center left", fontsize="x-large", bbox_to_anchor=(1, 0.8), frameon=False)
        ax.set_ylabel("Forecast of "+label["label_"+res[1]]["label_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(country_global[2])+"_"+str(col+1)][23:], fontsize=18)
        #ax.set_ylabel("variable", fontsize=15)
        #ax.set_title("Counterfactual: "+label["label_"+res[1]]["label_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(country_global[2])+"_"+str(col+1)][23:], fontdict = {"fontsize" : 20})  # positions to isolate title from string in labels
        fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[0]+"_"+res[1]+"_"+str(country[i])+"_"+str(col+1)+"_"+"VIX"+".png"), dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[0]+"_"+res[1]+"_"+str(country[i])+"_"+str(col+1)+"_"+"VIX"+".pdf"), dpi=200, bbox_inches="tight")
# end