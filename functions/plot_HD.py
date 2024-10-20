# %% HD

# Import packages
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

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

def plotHD(cc, palettes, v, t, output, label, OUTPUTS):
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
                scol = str(col+1)
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
