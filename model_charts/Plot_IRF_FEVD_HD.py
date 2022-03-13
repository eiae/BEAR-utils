"""
@author: Erik Andres Escayola

@title: Create charts of model results

@description:
  1. Import data after model estimation (BEAR results)
  2. Reframe data to have coherent structure
  3. Generate charts and save in publication format
"""

# %% Preamble

# Import packages
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm

# Define placeholders
# =============================================================================
v = 3  # number of variables
h = 40 + 2  # h horizon +2
c = 4  # time/lw.bound/median/up.bound

vv = v + 2  # extra initial condition and constant
cv = 2  # contrib./value
s = 80  # full sample
p = 4  # p number of lags
t = s - p + 1  # t estimation sample +1

res = ["IRF", "FEVD", "HD"]
res_bear = ["IRF", "FEVD", "hist decomp"]
cc = ["BR", "MX"]
country_long = ["Brazil", "Mexico"]
colors = ["g", "b"]  #, "k", "m", "c", "r", "y", "k"]
palettes = ["YlGn", "PuBu"]  #, "bone_r", "RdPu", "BrBG", "YlOrRd"]

xls = {}
data = {}
output = {}
label = {}
for i in res:
    data[i] = {}
    output[i] = {}
    label[i] = {}
    for j in cc:
        output[i][j] = {}
        label[i][j] = {}

# Define paths
# =============================================================================
WDPATH = "C:\\Users\\eandr\\Documents\\1_PROJECTS\\BEAR-utils\\model_charts"
DATA = {}
OUTPUTS = {}

for i in cc:
    DATA[i] = os.path.join(WDPATH, "example", "bvar_"+i, "results")
    OUTPUTS[i] = os.path.join(WDPATH, "example", "bvar_"+i, "charts")


# %% Data

# Import and wrangle data
# =============================================================================
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
                    output[j][i][j+"_"+str(row+1)+"_"+str(col+1)] = df
                    label[j][i][j+"_"+str(row+1)+"_"+str(col+1)] = lbl
        else:
            for row in range(v):
                for col in range(v):
                    df = pd.DataFrame(data=data[j][i].iloc[row*(h-1)+1:(h-1)*(row+1), col*c+1:c*(col+1)].values,
                                      index=data[j][i].iloc[row*(h-1)+1:(h-1)*(row+1), 0].values,
                                      columns=data[j][i].iloc[0, col*c+1:c*(col+1)].values)
                    lbl = data[j][i].iloc[(h-1)*row, c*col]
                    output[j][i][j+"_"+str(row+1)+"_"+str(col+1)] = df
                    label[j][i][j+"_"+str(row+1)+"_"+str(col+1)] = lbl 


# %% IRF

# Plot IRF
# =============================================================================
for i, k in zip(cc, colors):
    for row in range(v):
        for col in range(v):
            horizon = np.arange(1, h-1, dtype=float)
            bar0 = [q+1 for q in range(0,h-2)]
            #tick0 = [q+1 for q in range(0,h-2)]
            median = output["IRF"][i]["IRF_"+str(row+1)+"_"+str(col+1)]["median"]
            lw = output["IRF"][i]["IRF_"+str(row+1)+"_"+str(col+1)]["lw. bound"]
            up = output["IRF"][i]["IRF_"+str(row+1)+"_"+str(col+1)]["up. bound"]
            lbl = label["IRF"][i]["IRF_"+str(row+1)+"_"+str(col+1)]
            
            fig, ax = plt.subplots(1, figsize=(15, 7))
            ax.plot(bar0, median, color=k, linewidth=2, alpha=0.8) #label="median"
            ax.plot(bar0, lw, color=k, linewidth=1, linestyle=(0, (5, 1)), alpha=0.6) #label="16% credible set"
            ax.plot(bar0, up, color=k, linewidth=1, linestyle=(0, (5, 1)), alpha=0.6) #label="84% credible set"
            ax.fill_between(horizon, np.asarray(lw, dtype=float), np.asarray(up, dtype=float), color=k, alpha=0.2)
            ax.axhline(linewidth=2, color='k', alpha=0.6) 
            
            ax.tick_params(axis="both", labelsize=18) 
            #plt.xticks(tick0, tick0)
            ax.patch.set_facecolor("white")
            ax.grid(color="k", alpha=0.2, linewidth=0.5, linestyle="--")
            
            ax.set_title("IRF: "+lbl, fontsize=20)
            #ax.set_xlabel("horizon")
            #ax.set_ylabel("IRF")
            #ax.legend(loc="center left", fontsize=18, bbox_to_anchor=(1, 0.7), frameon=False)
            
            fig.savefig(os.path.join(OUTPUTS[i], "IRF_"+i+"_"+str(row+1)+"_"+str(col+1)+".png"), dpi=200, bbox_inches="tight")


# %% FEVD

# Plot FEVD
# =============================================================================
FEVD_median = {}
FEVD_total = {}
FEVD_rel_median = {}
FEVD_plot = {}

alphas = []
col_pal = {}

bar_width = 0.8
delim = 32  # adapt values to isolate correct string from label

for i, k in zip(cc, palettes):
    col_pal[i] = cm.get_cmap(k, v)
    for row in range(v):
        alphas.append(1.0-(1/v*row))  # alphas equal to number of variables
        FEVD_median[str(row+1)] = []
        for col in range(v):
            median = output["FEVD"][i]["FEVD_"+str(row+1)+"_"+str(col+1)]["median"]
            lw = output["FEVD"][i]["FEVD_"+str(row+1)+"_"+str(col+1)]["lw. bound"]
            up = output["FEVD"][i]["FEVD_"+str(row+1)+"_"+str(col+1)]["up. bound"]
            bar1 = [i+1 for i in range(len(median))]
            tick1 = [i for i in bar1]
            FEVD_median[str(row+1)].append(median)
        FEVD_total[str(row+1)] = list(map(sum, zip(*FEVD_median[str(row+1)])))
        for col in range(v):
            FEVD_rel_median[str(row+1)+"_"+str(col+1)] = (FEVD_median[str(row+1)][col] / FEVD_total[str(row+1)])*100
        FEVD_plot[str(row+1)] = np.zeros([v, h-2])
        for col in range(v):
            FEVD_plot[str(row+1)][col, :] = FEVD_rel_median[str(row+1)+"_"+str(col+1)].values
       
        fig, ax = plt.subplots(1, figsize=(15, 7))
        for col in range(v):
            lbl = label["FEVD"][i]["FEVD_"+str(row+1)+"_"+str(col+1)]
            ax.bar(bar1, FEVD_plot[str(row+1)][col],
                   bottom=np.sum(FEVD_plot[str(row+1)][:col], axis=0),
                   width=bar_width, label=lbl[delim:], color=col_pal[i](col),  # delim
                   edgecolor="k", linewidth=0.3) 

        ax.tick_params(axis="both", labelsize=18)
        plt.xticks(tick1, bar1)
        plt.xlim([min(tick1)-bar_width, max(tick1)+bar_width])
        plt.ylim(0, 100)
        ax.patch.set_facecolor("white")
        
        ax.set_title("FEVD: " + lbl[8:-delim], fontsize=20)  # delim
        ax.set_ylabel("contribution in %", fontsize=18)
        # ax.set_xlabel("horizon")
        ax.legend(loc="center left", fontsize="x-large", bbox_to_anchor=(1, 0.6), frameon=False)

        fig.savefig(os.path.join(OUTPUTS[i], "FEVD_"+i+"_"+str(row+1)+".png"), dpi=200, bbox_inches="tight")


# %% HD

# Plot HD
# =============================================================================


#-----------------------------------------------------------------------------
# appendix
#-----------------------------------------------------------------------------

# preamble countries
#--------------------
# colours = ["g", "b", "k", "m", "c", "r"]
# palettes = ["YlGn", "PuBu", "bone_r", "RdPu", "BrBG", "YlOrRd"]
# country = ["ID", "IN", "KR", "MY", "PH", "TH"]
# country_long = ["indonesia", "india", "korea", "malaysia", "philippines", "thailand"]

# preamble FEVD and HD
#----------------------
# FEVD_median = {}
# FEVD_total = {}
# FEVD_rel_median = {}
# FEVD_plot = {}

# HD_median = {}
# HD_total = {}
# HD_plot = {}
# HD_cumulated_data = {}
# HD_cumulated_data_neg = {}
# HD_row_mask = {}
# HD_stacked_data = {}

# bar_width = 0.8
# alphas = []
# for i in range(v):
#     alphas.append(1.0-(1/v*i))  # alphas equal to number of variables
# col_pal = {}
# for i in range(len(country)):
#     col_pal["palette"+"_"+country[i]] = cm.get_cmap(palettes[i], v)

# def get_cumulated_array(data, **kwargs):
#     cum = data.clip(**kwargs)
#     cum = np.cumsum(cum, axis=0)
#     df = np.zeros(np.shape(data))
#     df[1:] = cum[:-1]
#     return df

# plotting FEVD and HD
#----------------------
# for row in range(v):
    #     # FEVD
    #     FEVD_median["FEVD_median"+"_"+str(row+1)] = []
    #     for col in range(v):
    #         bar_l_1 = [i+1 for i in range(len(output["output_"+res[1]]["output_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["median"]))]
    #         tick_pos_1 = [i for i in bar_l_1]
    #         FEVD_median["FEVD_median"+"_"+str(row+1)].append(output["output_"+res[1]]["output_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["median"])
    #     FEVD_total["FEVD_total"+"_"+str(row+1)] = list(map(sum, zip(*FEVD_median["FEVD_median"+"_"+str(row+1)])))
    #     for col in range(v):
    #         FEVD_rel_median["FEVD_rel_median"+"_"+str(row+1)+"_"+str(col+1)] = (FEVD_median["FEVD_median"+"_"+str(row+1)][col] / FEVD_total["FEVD_total"+"_"+str(row+1)])*100
    #     FEVD_plot["FEVD_plot"+"_"+str(row+1)] = np.zeros([v, r-2])
    #     for col in range(v):
    #         FEVD_plot["FEVD_plot"+"_"+str(row+1)][col, :] = FEVD_rel_median["FEVD_rel_median_"+str(row+1)+"_"+str(col+1)].values
    #     fig, ax = plt.subplots(1, figsize=(15, 7))
    #     for col in range(v):
    #         ax.bar(bar_l_1, FEVD_plot["FEVD_plot"+"_"+str(row+1)][col], bottom=np.sum(FEVD_plot["FEVD_plot"+"_"+str(row+1)][:col], axis=0),  width=bar_width, label="contribution of "+label["label_"+res[1]]["label_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)][36:], color=col_pal["palette"+"_"+str(country[i])](col), edgecolor="k", linewidth=0.3) # alpha=alphas[col],
    #     ax.tick_params(axis="both", labelsize=18) ###
    #     plt.xticks(tick_pos_1, bar_l_1)
    #     plt.xlim([min(tick_pos_1)-bar_width, max(tick_pos_1)+bar_width])
    #     plt.ylim(0, 100)
    #     ax.patch.set_facecolor("white")
    #     ax.legend(loc="center left", fontsize="x-large", bbox_to_anchor=(1, 0.6), frameon=False)
    #     ax.set_ylabel("Variability of "+label["label_"+res[1]]["label_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(1)][8:-34], fontsize=18)
    #     #ax.set_xlabel("horizon")
    #     #ax.set_ylabel("contribution in %")
    #     #ax.set_title(res[1]+": "+label["label_"+res[1]]["label_"+res[1]+"_"+str(country[i])][res[1]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(1)][8:-35])  # positions to isolate title from string in labels
    #     fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[1]+"_"+str(country[i])+"_"+str(row+1)+".png"), dpi=200, bbox_inches="tight")
    #     fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[1]+"_"+str(country[i])+"_"+str(row+1)+".pdf"), dpi=200, bbox_inches="tight")
    # for row in range(v):
    #     # HD
    #     HD_median["HD_median"+"_"+str(row+1)] = []
    #     for col in range(v):
    #         bar_l_2 = [i for i in range(len(output["output_"+res[2]]["output_"+res[2]+"_"+str(country[i])][res[2]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["median"]))]
    #         tick_pos_2 = [i for i in bar_l_2]
    #         HD_median["HD_median"+"_"+str(row+1)].append(output["output_"+res[2]]["output_"+res[2]+"_"+str(country[i])][res[2]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)]["median"])
    #     HD_total["HD_total"+"_"+str(row+1)] = list(map(sum, zip(*HD_median["HD_median"+"_"+str(row+1)])))
    #     HD_plot["HD_plot"+"_"+str(row+1)] = np.zeros([v, r_2-2])
    #     for col in range(v):
    #         HD_plot["HD_plot"+"_"+str(row+1)][col, :] = HD_median["HD_median_"+str(row+1)][col].values
    #     HD_cumulated_data["HD_cumulated"+"_"+str(row+1)] = get_cumulated_array(HD_plot["HD_plot"+"_"+str(row+1)], min=0)
    #     HD_cumulated_data_neg["HD_cumulated_neg"+"_"+str(row+1)] = get_cumulated_array(HD_plot["HD_plot"+"_"+str(row+1)], max=0)
    #     HD_row_mask["HD_row_mask"+"_"+str(row+1)] = (HD_plot["HD_plot"+"_"+str(row+1)] < 0)
    #     HD_cumulated_data["HD_cumulated"+"_"+str(row+1)][HD_row_mask["HD_row_mask"+"_"+str(row+1)]] = HD_cumulated_data_neg["HD_cumulated_neg"+"_"+str(row+1)][HD_row_mask["HD_row_mask"+"_"+str(row+1)]]
    #     HD_stacked_data["HD_stacked_data"+"_"+str(row+1)] = HD_cumulated_data["HD_cumulated"+"_"+str(row+1)]
    #     fig, ax = plt.subplots(1, figsize=(15, 7))
    #     for col in range(v):
    #         ax.bar(bar_l_2, HD_plot["HD_plot"+"_"+str(row+1)][col], bottom=HD_stacked_data["HD_stacked_data"+"_"+str(row+1)][col],  width=bar_width, label=label["label_"+res[2]]["label_"+res[2]+"_"+str(country[i])][res[2]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)][:-24], color=col_pal["palette"+"_"+str(country[i])](col), edgecolor="k", linewidth=0.1) # alpha=alphas[col],
    #     ax.plot(bar_l_2, HD_total["HD_total"+"_"+str(row+1)], color="k", linewidth=5, label="actual")
    #     ax.tick_params(axis="both", labelsize=18) ###
    #     plt.xticks(np.arange(0, len(tick_pos_2), 4), [output["output_"+res[2]]["output_"+res[2]+"_"+str(country[i])][res[2]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(col+1)].index[j][:-2] for j in np.arange(0, len(tick_pos_2), 4)], rotation='vertical')
    #     plt.xlim([min(tick_pos_2)-bar_width, max(tick_pos_2)+bar_width])
    #     ax.patch.set_facecolor("white")
    #     ax.grid(color="k", alpha=0.2, linewidth=0.5, linestyle="--")
    #     ax.legend(loc="center left", fontsize="x-large", bbox_to_anchor=(1, 0.6), frameon=False)
    #     ax.set_ylabel("Fluctuations of "+label["label_"+res[2]]["label_"+res[2]+"_"+str(country[i])][res[2]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(1)][35:-11], fontsize=18)
    #     #ax.set_ylabel("contribution to fluctuation")
    #     #ax.set_title(res[2]+": "+label["label_"+res[2]]["label_"+res[2]+"_"+str(country[i])][res[2]+"_"+str(country[i])+"_"+str(row+1)+"_"+str(1)][35:-11])  # positions to isolate title from string in labels
    #     fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[2]+"_"+str(country[i])+"_"+str(row+1)+".png"), dpi=200, bbox_inches="tight")
    #     fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], res[2]+"_"+str(country[i])+"_"+str(row+1)+".pdf"), dpi=200, bbox_inches="tight")