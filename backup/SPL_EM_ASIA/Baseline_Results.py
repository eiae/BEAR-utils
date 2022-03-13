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

WDPATH = "C:\\Users\\eandr\\Documents\\1_UNIVERSITY\\A_MANNHEIM\\1_Notes\\A_TFM\\4_Model\\Model"
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
v_2 = 14  # extra variable since exogenous part in HD
r = 22  # h horizon +2
r_2 = 85  # t estimation sample +2
c = 4  # time/lw.bound/median/up.bound

res = ["IRF", "FEVD", "HD"]
res_bear = ["IRF", "FEVD", "hist decomposition"]
colours = ["g", "b", "k", "m", "c", "r"]
palettes = ["YlGn", "PuBu", "bone_r", "RdPu", "BrBG", "YlOrRd"]
country = ["ID", "IN", "KR", "MY", "PH", "TH"]
country_long = ["indonesia", "india", "korea", "malaysia", "philippines", "thailand"]

FEVD_median = {}
FEVD_total = {}
FEVD_rel_median = {}
FEVD_plot = {}

HD_median = {}
HD_total = {}
HD_plot = {}
HD_cumulated_data = {}
HD_cumulated_data_neg = {}
HD_row_mask = {}
HD_stacked_data = {}

bar_width = 0.8
alphas = []
for i in range(v):
    alphas.append(1.0-(1/v*i))  # alphas equal to number of variables
col_pal = {}
for i in range(len(country)):
    col_pal["palette"+"_"+country[i]] = cm.get_cmap(palettes[i], v)

def get_cumulated_array(data, **kwargs):
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    df = np.zeros(np.shape(data))
    df[1:] = cum[:-1]
    return df


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
    DATA["DATA_"+str(country[i])] = os.path.join(WDPATH, "bvar_"+str(country[i]), "toolbox_4.2", "results", n)
    OUTPUTS["OUTPUTS_"+str(country[i])] = os.path.join(WDPATH, "bvar_"+str(country[i]), "toolbox_4.2", "results", n)
    os.chdir(DATA["DATA_"+str(country[i])])
    # data
    xls["xls_"+str(country[i])] = pd.ExcelFile("results_"+str(country[i])+"_"+n+".xlsx")  
    for k, l in zip(res, res_bear):
        data["data_"+k]["data_"+k+"_"+str(country[i])] = pd.read_excel(xls["xls_"+str(country[i])], l, index_col=0, header=0)
        data["data_"+k]["data_"+k+"_"+str(country[i])].dropna(axis=0, how="all", inplace=True)
        data["data_"+k]["data_"+k+"_"+str(country[i])].dropna(axis=1, how="all", inplace=True)
        for row in range(v):
            for col_2 in range(v_2):
                for col in range(v):
                    # reframe data considering different types of results
                    if data["data_"+k] == data["data_"+res[2]]:
                        if pd.isna(data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[r_2*row+1, c*col_2]):
                            data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[r_2*row+1, c*col_2] = data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[r_2*row, c*col_2]
                            data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[r_2*row, c*col_2] = np.nan
                    else:
                        if pd.isna(data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[r*row+1, c*col]):
                            data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[r*row+1, c*col] = data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[r*row, c*col]
                            data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[r*row, c*col] = np.nan
        data["data_"+k]["data_"+k+"_"+str(country[i])].dropna(axis=0, how="all", inplace=True) 
        for row in range(v):
            for col_2 in range(v_2):
                for col in range(v):
                    # save outputs individually considering different types of results
                    if data["data_"+k] == data["data_"+res[2]]:
                        output["output_"+k]["output_"+k+"_"+str(country[i])][k+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col_2+1)] = pd.DataFrame(data=data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[row*(r_2-1)+1:(r_2-1)*(row+1), col_2*c+1:c*(col_2+1)].values,
                                     index=data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[row*(r_2-1)+1:(r_2-1)*(row+1), 0].values,
                                     columns=data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[0, (col_2*c+1):c*(col_2+1)].values)
                        label["label_"+k]["label_"+k+"_"+str(country[i])][k+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col_2+1)] = data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[(r_2-1)*row, c*col_2]  
                    else:
                        output["output_"+k]["output_"+k+"_"+str(country[i])][k+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)] = pd.DataFrame(data=data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[row*(r-1)+1:(r-1)*(row+1), col*c+1:c*(col+1)].values,
                                       index=data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[row*(r-1)+1:(r-1)*(row+1), 0].values,
                                       columns=data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[0, col*c+1:c*(col+1)].values)
                        label["label_"+k]["label_"+k+"_"+str(country[i])][k+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)] = data["data_"+k]["data_"+k+"_"+str(country[i])].iloc[(r-1)*row, c*col]
    for row in range(v):
        for col in range(v):
            # IRF
            horizon = np.arange(1, 21, dtype=float)
            bar_l_0 = [i+1 for i in range(len(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["median"]))]
            tick_pos_0 = [i for i in bar_l_0]
            fig, ax = plt.subplots(1, figsize=(15, 7))
            ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["median"], color=colours[i], linewidth=2, alpha=0.8, label="median")
            ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["lw. bound"], color=colours[i], linewidth=1, linestyle=(0, (5, 1)), alpha=0.6, label="16% credible set")
            ax.plot(bar_l_0, output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["up. bound"], color=colours[i], linewidth=1, linestyle=(0, (5, 1)), alpha=0.6, label="84% credible set")
            ax.fill_between(horizon, np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["lw. bound"], dtype=float), np.asarray(output["output_"+res[0]]["output_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["up. bound"], dtype=float), color=colours[i], alpha=0.2)
            ax.axhline(linewidth=2, color='k', alpha=0.6) ###
            ax.tick_params(axis="both", labelsize=18) ###
            plt.xticks(tick_pos_0, bar_l_0)
            ax.patch.set_facecolor("white")
            ax.grid(color="k", alpha=0.2, linewidth=0.5, linestyle="--")
            ax.set_ylabel("Response of "+label["label_"+res[0]]["label_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)][12:], fontsize=18)
            #ax.legend(loc="center left", fontsize="small", bbox_to_anchor=(1, 0.7), frameon=False)
            #ax.set_xlabel("horizon")
            #ax.set_ylabel("IRF")
            #ax.set_title(res[0]+": "+label["label_"+res[0]]["label_"+res[0]+"_"+str(country[i])][res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)])
            fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)+".png"), dpi=200, bbox_inches="tight")
            fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[0]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)+".pdf"), dpi=200, bbox_inches="tight")
    for row in range(v):
        # FEVD
        FEVD_median["FEVD_median"+"_"+str(row+1)] = []
        for col in range(v):
            bar_l_1 = [i+1 for i in range(len(output["output_"+res[1]]["output_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["median"]))]
            tick_pos_1 = [i for i in bar_l_1]
            FEVD_median["FEVD_median"+"_"+str(row+1)].append(output["output_"+res[1]]["output_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["median"])
        FEVD_total["FEVD_total"+"_"+str(row+1)] = list(map(sum, zip(*FEVD_median["FEVD_median"+"_"+str(row+1)])))
        for col in range(v):
            FEVD_rel_median["FEVD_rel_median"+"_"+str(row+1)+"_"+str(col+1)] = (FEVD_median["FEVD_median"+"_"+str(row+1)][col] / FEVD_total["FEVD_total"+"_"+str(row+1)])*100
        FEVD_plot["FEVD_plot"+"_"+str(row+1)] = np.zeros([v, r-2])
        for col in range(v):
            FEVD_plot["FEVD_plot"+"_"+str(row+1)][col, :] = FEVD_rel_median["FEVD_rel_median_"+str(row+1)+"_"+str(col+1)].values
        fig, ax = plt.subplots(1, figsize=(15, 7))
        for col in range(v):
            ax.bar(bar_l_1, FEVD_plot["FEVD_plot"+"_"+str(row+1)][col], bottom=np.sum(FEVD_plot["FEVD_plot"+"_"+str(row+1)][:col], axis=0),  width=bar_width, label="contribution of "+label["label_"+res[1]]["label_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)][36:], color=col_pal["palette"+"_"+str(country[i])](col), edgecolor="k", linewidth=0.3) # alpha=alphas[col],
        ax.tick_params(axis="both", labelsize=18) ###
        plt.xticks(tick_pos_1, bar_l_1)
        plt.xlim([min(tick_pos_1)-bar_width, max(tick_pos_1)+bar_width])
        plt.ylim(0, 100)
        ax.patch.set_facecolor("white")
        ax.legend(loc="center left", fontsize="x-large", bbox_to_anchor=(1, 0.6), frameon=False)
        ax.set_ylabel("Variability of "+label["label_"+res[1]]["label_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(1)][8:-34], fontsize=18)
        #ax.set_xlabel("horizon")
        #ax.set_ylabel("contribution in %")
        #ax.set_title(res[1]+": "+label["label_"+res[1]]["label_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(1)][8:-35])  # positions to isolate title from string in labels
        fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[1]+"_"+str(country[i])+"_"+str(row+1)+".png"), dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[1]+"_"+str(country[i])+"_"+str(row+1)+".pdf"), dpi=200, bbox_inches="tight")
    for row in range(v):
        # HD
        HD_median["HD_median"+"_"+str(row+1)] = []
        for col in range(v):
            bar_l_2 = [i for i in range(len(output["output_"+res[2]]["output_"+res[2]+"_"+str(country[i])][res[2]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["median"]))]
            tick_pos_2 = [i for i in bar_l_2]
            HD_median["HD_median"+"_"+str(row+1)].append(output["output_"+res[2]]["output_"+res[2]+"_"+str(country[i])][res[2]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["median"])
        HD_total["HD_total"+"_"+str(row+1)] = list(map(sum, zip(*HD_median["HD_median"+"_"+str(row+1)])))
        HD_plot["HD_plot"+"_"+str(row+1)] = np.zeros([v, r_2-2])
        for col in range(v):
            HD_plot["HD_plot"+"_"+str(row+1)][col, :] = HD_median["HD_median_"+str(row+1)][col].values
        HD_cumulated_data["HD_cumulated"+"_"+str(row+1)] = get_cumulated_array(HD_plot["HD_plot"+"_"+str(row+1)], min=0)
        HD_cumulated_data_neg["HD_cumulated_neg"+"_"+str(row+1)] = get_cumulated_array(HD_plot["HD_plot"+"_"+str(row+1)], max=0)
        HD_row_mask["HD_row_mask"+"_"+str(row+1)] = (HD_plot["HD_plot"+"_"+str(row+1)] < 0)
        HD_cumulated_data["HD_cumulated"+"_"+str(row+1)][HD_row_mask["HD_row_mask"+"_"+str(row+1)]] = HD_cumulated_data_neg["HD_cumulated_neg"+"_"+str(row+1)][HD_row_mask["HD_row_mask"+"_"+str(row+1)]]
        HD_stacked_data["HD_stacked_data"+"_"+str(row+1)] = HD_cumulated_data["HD_cumulated"+"_"+str(row+1)]
        fig, ax = plt.subplots(1, figsize=(15, 7))
        for col in range(v):
            ax.bar(bar_l_2, HD_plot["HD_plot"+"_"+str(row+1)][col], bottom=HD_stacked_data["HD_stacked_data"+"_"+str(row+1)][col],  width=bar_width, label=label["label_"+res[2]]["label_"+res[2]+"_"+str(country[i])][res[2]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)][:-24], color=col_pal["palette"+"_"+str(country[i])](col), edgecolor="k", linewidth=0.1) # alpha=alphas[col],
        ax.plot(bar_l_2, HD_total["HD_total"+"_"+str(row+1)], color="k", linewidth=5, label="actual")
        ax.tick_params(axis="both", labelsize=18) ###
        plt.xticks(np.arange(0, len(tick_pos_2), 4), [output["output_"+res[2]]["output_"+res[2]+"_"+str(country[i])][res[2]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)].index[j][:-2] for j in np.arange(0, len(tick_pos_2), 4)], rotation='vertical')
        plt.xlim([min(tick_pos_2)-bar_width, max(tick_pos_2)+bar_width])
        ax.patch.set_facecolor("white")
        ax.grid(color="k", alpha=0.2, linewidth=0.5, linestyle="--")
        ax.legend(loc="center left", fontsize="x-large", bbox_to_anchor=(1, 0.6), frameon=False)
        ax.set_ylabel("Fluctuations of "+label["label_"+res[2]]["label_"+res[2]+"_"+str(country[i])][res[2]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(1)][35:-11], fontsize=18)
        #ax.set_ylabel("contribution to fluctuation")
        #ax.set_title(res[2]+": "+label["label_"+res[2]]["label_"+res[2]+"_"+str(country[i])][res[2]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(1)][35:-11])  # positions to isolate title from string in labels
        fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[2]+"_"+str(country[i])+"_"+str(row+1)+".png"), dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[2]+"_"+str(country[i])+"_"+str(row+1)+".pdf"), dpi=200, bbox_inches="tight")
# end