# %% FEVD

# Import packages
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

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

def plotFEVD(cc, palettes, v, h, output, label, OUTPUTS):
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
                scol = str(col+1)
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
