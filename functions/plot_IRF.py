# %% IRF

# Import packages
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm


# Plot IRF
# =============================================================================
def plotIRF(cc, colors, v, h, output, label, OUTPUTS):

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

