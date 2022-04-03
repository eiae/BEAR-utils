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
colors = ["g", "b"]
palettes = ["Greens", "Blues"]

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


# %% IRF

# Plot IRF
# =============================================================================
for i, k in zip(cc, colors):
    for row in range(v):
        for col in range(v):
            horizon = np.arange(1, h-1, dtype=float)
            bar0 = [q+1 for q in range(0, h-2)]
            #tick0 = [q+1 for q in range(0, h-2)]
            srow = str(row+1)
            scol = str(col+1)
            median = output["IRF"][i]["IRF_"+srow+"_"+scol]["median"]
            lw = output["IRF"][i]["IRF_"+srow+"_"+scol]["lw. bound"]
            up = output["IRF"][i]["IRF_"+srow+"_"+scol]["up. bound"]
            lbl = label["IRF"][i]["IRF_"+srow+"_"+scol]

            fig, ax = plt.subplots(1, figsize=(15, 7))
            ax.plot(bar0, median, color=k, linewidth=2, alpha=0.8)  #label="median"
            ax.plot(bar0, lw, color=k, linewidth=1, linestyle=(0, (5, 1)), alpha=0.6)  #label="16% credible set"
            ax.plot(bar0, up, color=k, linewidth=1, linestyle=(0, (5, 1)), alpha=0.6)  #label="84% credible set"
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

            fig.savefig(os.path.join(OUTPUTS[i], "IRF_"+i+"_"+srow+"_"+scol+".png"), dpi=200, bbox_inches="tight")


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
    col_pal[i] = cm.get_cmap(k, v+1)  # avoid white in cmap
    for row in range(v):
        alphas.append(1.0-(1/v*row))  # alphas equal to number of variables
        srow = str(row+1)
        FEVD_median[srow] = []
        for col in range(v):
            scol = str(col+1)
            median = output["FEVD"][i]["FEVD_"+srow+"_"+scol]["median"]
            lw = output["FEVD"][i]["FEVD_"+srow+"_"+scol]["lw. bound"]
            up = output["FEVD"][i]["FEVD_"+srow+"_"+scol]["up. bound"]
            bar1 = [i+1 for i in range(len(median))]
            tick1 = [i for i in bar1]
            FEVD_median[srow].append(median)
        FEVD_total[srow] = list(map(sum, zip(*FEVD_median[srow])))
        for col in range(v):
            scol = str(col+1)
            FEVD_rel_median[srow+"_"+scol] = (FEVD_median[srow][col] / FEVD_total[srow])*100
        FEVD_plot[srow] = np.zeros([v, h-2])
        for col in range(v):
            scol = str(col+1)
            FEVD_plot[srow][col, :] = FEVD_rel_median[srow+"_"+scol].values

        fig, ax = plt.subplots(1, figsize=(15, 7))
        for col in range(v):
            lbl = label["FEVD"][i]["FEVD_"+srow+"_"+scol]
            ax.bar(bar1, FEVD_plot[srow][col],
                   bottom=np.sum(FEVD_plot[srow][:col], axis=0),
                   width=bar_width, label=lbl[delim:], color=col_pal[i](col+1),  # delim
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

        fig.savefig(os.path.join(OUTPUTS[i], "FEVD_"+i+"_"+srow+".png"), dpi=200, bbox_inches="tight")


# %% HD

# Plot HD
# =============================================================================
HD_median = {}
HD_total = {}
HD_plot = {}
HD_cumulated_data = {}
HD_cumulated_data_neg = {}
HD_row_mask = {}
HD_stacked_data = {}

col_pal = {}

bar_width = 0.8


def get_cumulated_array(data, **kwargs):
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    df = np.zeros(np.shape(data))
    df[1:] = cum[:-1]
    return df


for i, k in zip(cc, palettes):
    col_pal[i] = cm.get_cmap(k, v+1)  # avoid white in cmap
    for row in range(v):
        srow = str(row+1)
        HD_median[srow] = []
        for col in range(v):
            scol = str(col+1)
            median = output["HD"][i]["HD_"+srow+"_"+scol]["median"]
            bar2 = [i for i in range(len(median))]
            tick2 = [i for i in bar2]
            HD_median[srow].append(median)
        HD_total[srow] = list(map(sum, zip(*HD_median[srow])))
        HD_plot[srow] = np.zeros([v, t-1])
        for col in range(v):
            HD_plot[srow][col, :] = HD_median[srow][col].values
        HD_cumulated_data[srow] = get_cumulated_array(HD_plot[srow], min=0)
        HD_cumulated_data_neg[srow] = get_cumulated_array(HD_plot[srow], max=0)
        HD_row_mask[srow] = (HD_plot[srow] < 0)
        HD_cumulated_data[srow][HD_row_mask[srow]] = HD_cumulated_data_neg[srow][HD_row_mask[srow]]
        HD_stacked_data[srow] = HD_cumulated_data[srow]

        fig, ax = plt.subplots(1, figsize=(15, 7))
        for col in range(v):
            lbl = label["HD"][i]["HD_"+srow+"_"+scol]
            ax.bar(bar2, HD_plot[srow][col],
                   bottom=HD_stacked_data[srow][col],
                   width=bar_width, label=lbl[16:-20],  # adapt values to isolate correct string from label
                   color=col_pal[i](col+1), edgecolor="k", linewidth=0.1)
        ax.plot(bar2, HD_total[srow], color="k", linewidth=5, label="actual")

        ax.tick_params(axis="both", labelsize=18)
        plt.xticks(np.arange(0, len(tick2), 4), [median.index[j][:-2] for j in np.arange(0, len(tick2), 4)], rotation='vertical')
        plt.xlim([min(tick2)-bar_width, max(tick2)+bar_width])
        ax.patch.set_facecolor("white")
        ax.grid(color="k", alpha=0.2, linewidth=0.5, linestyle="--")

        ax.set_title("HD:" + lbl[32:-12], fontsize=20)  # adapt values to isolate correct string from label
        ax.set_ylabel("contribution to fluctuation", fontsize=18)
        # ax.set_xlabel("sample")
        ax.legend(loc="center left", fontsize="x-large", bbox_to_anchor=(1, 0.6), frameon=False)

        fig.savefig(os.path.join(OUTPUTS[i], "HD_"+i+"_"+srow+".png"), dpi=200, bbox_inches="tight")
