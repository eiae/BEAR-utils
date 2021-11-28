"""
@author: Erik Andres Escayola

@title: Data analysis of model variables

@description:
  1. Import data to be used in model estimation
  2. Reframe data to have coherent structure
  3. Inspect data through line-plots and summary statistics
  4. Compute correlation matrices and bilateral line-fitted relations
  5. Test for unit roots (stationarity) and Granger causality (block exog.)
  6. Save plottings
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import grangercausalitytests

WDPATH = "C:\\Users\\eandr\\Documents\\1_PROJECTS\\ST_LATAM"
DATA = {}
OUTPUTS = {}


# Define variables + parameters
# =============================================================================
xls = {}
data = {}
summary = {}
adf_test = {}
kpss_test = {}
ur_adf = {}
ur_kpss = {}
gc_test = {}
pred_caus_gc = {}
signif = [0.05]
colours = ["g", "b", "r", "y", "m", "k"]
palettes = ["YlGn", "PuBu", "OrRd", "YlOrBr", "PiYG", "RdGy"]
country = ["BR", "MX", "CL", "CO", "PE", "AG"]
country_long = ["brazil", "mexico", "chile", "colombia", "peru", "argentina"]
country_title = ["Brazil", "Mexico", "Chile", "Colombia", "Peru", "Argentina"]


# Import data + plot + summary stats + save
# =============================================================================
for i in range(len(country)):
    # paths
    DATA["DATA_"+str(country[i])] = os.path.join(WDPATH, "bvar_"+str(country_long[i]))
    OUTPUTS["OUTPUTS_"+str(country[i])] = os.path.join(WDPATH, "bvar_"+str(country_long[i]), "outputs")
    os.chdir(DATA["DATA_"+str(country[i])])
    # data
    xls["xls_"+str(country[i])] = pd.ExcelFile("bvar_"+str(country_long[i])+"_data_ST.xlsx")
    data["data_"+str(country[i])] = pd.read_excel(xls["xls_"+str(country[i])], "Summary_stats", index_col=1, header=1)
    data["data_"+str(country[i])].dropna(axis=1, how="all", inplace=True)
    # summary stats
    summary["summary_"+str(country[i])] = data["data_"+str(country[i])].describe()
    # heatmaps
    #sns.set(font_scale=1.6)
    fig = plt.figure(figsize=(15, 7)) # figsize=(30,20)
    sns.heatmap(data["data_"+str(country[i])].corr(), cmap=palettes[i], linecolor='white', linewidths=1, annot=True).get_figure()  # annot_kws={"size": 14}
    plt.title(country_title[i], pad=20)
    fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "heatmap_"+str(country[i])+".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "heatmap_"+str(country[i])+".pdf"), dpi=200, bbox_inches="tight")
    # regressions (extendible for more variables)
    # basic macro-financial relations
    fig_1 = sns.jointplot(x=data["data_"+str(country[i])]["sti"+str(country[i])],
                         y=data["data_"+str(country[i])]["gdp"+str(country[i])],
                         kind='reg', color=colours[i]) #, height=10
    fig_2 = sns.jointplot(x=data["data_"+str(country[i])]["sti"+str(country[i])],
                         y=data["data_"+str(country[i])]["fxr"+str(country[i])],
                         kind='reg', color=colours[i]) # , height=10
    fig_3 = sns.jointplot(x=data["data_"+str(country[i])]["sti"+str(country[i])],
                         y=data["data_"+str(country[i])]["smkt"+str(country[i])],
                         kind='reg', color=colours[i]) # , height=10
    fig_4 = sns.jointplot(x=data["data_"+str(country[i])]["sti"+str(country[i])],
                         y=data["data_"+str(country[i])]["vix"],
                         kind='reg', color=colours[i]) # , height=10
    fig_5 = sns.jointplot(x=data["data_"+str(country[i])]["sti"+str(country[i])],
                         y=data["data_"+str(country[i])]["pcf"+str(country[i])],
                         kind='reg', color=colours[i]) # , height=10.
 
    fig_1.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "regplot_sti_gdp_"+str(country[i])+".png"), dpi=200)
    fig_1.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "regplot_sti_gdp_"+str(country[i])+".pdf"), dpi=200)
    fig_2.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "regplot_sti_fxr_"+str(country[i])+".png"), dpi=200)
    fig_2.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "regplot_sti_fxr_"+str(country[i])+".pdf"), dpi=200)
    fig_3.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "regplot_sti_smkt_"+str(country[i])+".png"), dpi=200)
    fig_3.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "regplot_sti_smkt_"+str(country[i])+".pdf"), dpi=200)
    fig_4.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "regplot_sti_vix_"+str(country[i])+".png"), dpi=200)
    fig_4.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "regplot_sti_vix_"+str(country[i])+".pdf"), dpi=200)    
    fig_5.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "regplot_sti_pcf_"+str(country[i])+".png"), dpi=200)
    fig_5.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "regplot_sti_pcf_"+str(country[i])+".pdf"), dpi=200)
    for j in range(data["data_"+str(country[i])].shape[1]):
        # lineplots
        fig, ax = plt.subplots(1, figsize=(15, 7))
        ax.plot(data["data_"+str(country[i])].iloc[:, j], color=colours[i], linewidth=2)
        ax.patch.set_facecolor("white")
        ax.grid(color="k", alpha=0.2, linewidth=0.5, linestyle="--")
        ax.set_title(data["data_"+str(country[i])].columns[j])
        fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "lineplot_"+str(data["data_"+str(country[i])].columns[j])+".png"), dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(OUTPUTS["OUTPUTS_"+str(country[i])], "lineplot_"+str(data["data_"+str(country[i])].columns[j])+".pdf"), dpi=200, bbox_inches="tight")
    
    # ADAPT CODE FROM HERE https://rishi-a.github.io/2020/05/25/granger-causality.html
    for j in data["data_"+str(country[i])].columns:
        # UR tests - ADF null UR / KPSS null not UR
        adf_test["adf_"+j] = adfuller(data["data_"+str(country[i])][j], regression="c", autolag="AIC", regresults=True)[1] # lag length criteria Schwarz 1978
        kpss_test["kpss_"+j] = kpss(data["data_"+str(country[i])][j], regression="c")[1] # lag length criteria Schwert 1989
        for k in range(len(signif)):
            # keep only UR results
            ur_adf["adf_"+j+"_"+str(signif[k])] = adf_test["adf_"+j] <= signif[k]
            ur_kpss["kpss_"+j+"_"+str(signif[k])] = kpss_test["kpss_"+j] > signif[k]
            if ur_adf["adf_"+j+"_"+str(signif[k])]:
                del ur_adf["adf_"+j+"_"+str(signif[k])]
            if ur_kpss["kpss_"+j+"_"+str(signif[k])]:
                del ur_kpss["kpss_"+j+"_"+str(signif[k])]
    gc_test["gc_"+str(country[i])] = {}
    for j in range(1, 3): # adapt loop according to lag length
        # GC tests - last variable not GC first variable (need to optimize with loops)
        gc_test["gc_"+str(country[i])]["lag_"+str(j)] = {}     
        gc_test["gc_"+str(country[i])]["lag_"+str(j)]["gdp_"+str(country[i])+"_"+"sti_"+str(country[i])] = grangercausalitytests(data["data_"+str(country[i])][["gdp"+str(country[i]), "sti"+str(country[i])]], maxlag=2, verbose=False)[j][0]["params_ftest"][1]
        gc_test["gc_"+str(country[i])]["lag_"+str(j)]["inf_"+str(country[i])+"_"+"sti_"+str(country[i])] = grangercausalitytests(data["data_"+str(country[i])][["inf"+str(country[i]), "sti"+str(country[i])]], maxlag=2, verbose=False)[j][0]["params_ftest"][1]
        gc_test["gc_"+str(country[i])]["lag_"+str(j)]["fxr_"+str(country[i])+"_"+"sti_"+str(country[i])] = grangercausalitytests(data["data_"+str(country[i])][["fxr"+str(country[i]), "sti"+str(country[i])]], maxlag=2, verbose=False)[j][0]["params_ftest"][1]
        gc_test["gc_"+str(country[i])]["lag_"+str(j)]["int_"+str(country[i])+"_"+"sti_"+str(country[i])] = grangercausalitytests(data["data_"+str(country[i])][["int"+str(country[i]), "sti"+str(country[i])]], maxlag=2, verbose=False)[j][0]["params_ftest"][1]
        gc_test["gc_"+str(country[i])]["lag_"+str(j)]["smkt_"+str(country[i])+"_"+"sti_"+str(country[i])] = grangercausalitytests(data["data_"+str(country[i])][["smkt"+str(country[i]), "sti"+str(country[i])]], maxlag=2, verbose=False)[j][0]["params_ftest"][1]
        gc_test["gc_"+str(country[i])]["lag_"+str(j)]["vix"+"_"+"sti_"+str(country[i])] = grangercausalitytests(data["data_"+str(country[i])][["vix", "sti"+str(country[i])]], maxlag=2, verbose=False)[j][0]["params_ftest"][1]
        gc_test["gc_"+str(country[i])]["lag_"+str(j)]["pcf_"+str(country[i])+"_"+"sti_"+str(country[i])] = grangercausalitytests(data["data_"+str(country[i])][["pcf"+str(country[i]), "sti"+str(country[i])]], maxlag=2, verbose=False)[j][0]["params_ftest"][1]
        for k in range(len(signif)):
            # keep only GC results
            pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"gdp_"+str(country[i])+"_"+"sti_"+str(country[i])] = gc_test["gc_"+str(country[i])]["lag_"+str(j)]["gdp_"+str(country[i])+"_"+"sti_"+str(country[i])] <= signif[k]
            pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"inf_"+str(country[i])+"_"+"sti_"+str(country[i])] = gc_test["gc_"+str(country[i])]["lag_"+str(j)]["int_"+str(country[i])+"_"+"sti_"+str(country[i])] <= signif[k]
            pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"fxr_"+str(country[i])+"_"+"sti_"+str(country[i])] = gc_test["gc_"+str(country[i])]["lag_"+str(j)]["fxr_"+str(country[i])+"_"+"sti_"+str(country[i])] <= signif[k]
            pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"int_"+str(country[i])+"_"+"sti_"+str(country[i])] = gc_test["gc_"+str(country[i])]["lag_"+str(j)]["int_"+str(country[i])+"_"+"sti_"+str(country[i])] <= signif[k]
            pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"smkt_"+str(country[i])+"_"+"sti_"+str(country[i])] = gc_test["gc_"+str(country[i])]["lag_"+str(j)]["smkt_"+str(country[i])+"_"+"sti_"+str(country[i])] <= signif[k]
            pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"vix"+"_"+"sti_"+str(country[i])] = gc_test["gc_"+str(country[i])]["lag_"+str(j)]["vix"+"_"+"sti_"+str(country[i])] <= signif[k]
            pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"pcf_"+str(country[i])+"_"+"sti_"+str(country[i])] = gc_test["gc_"+str(country[i])]["lag_"+str(j)]["pcf_"+str(country[i])+"_"+"sti_"+str(country[i])] <= signif[k]
            if pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"gdp_"+str(country[i])+"_"+"sti_"+str(country[i])] == False:
                del pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"gdp_"+str(country[i])+"_"+"sti_"+str(country[i])] 
            if pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"inf_"+str(country[i])+"_"+"sti_"+str(country[i])] == False:
                del pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"inf_"+str(country[i])+"_"+"sti_"+str(country[i])] 
            if pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"fxr_"+str(country[i])+"_"+"sti_"+str(country[i])] == False:
                del pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"fxr_"+str(country[i])+"_"+"sti_"+str(country[i])] 
            if pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"int_"+str(country[i])+"_"+"sti_"+str(country[i])] == False:
                del pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"int_"+str(country[i])+"_"+"sti_"+str(country[i])] 
            if pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"smkt_"+str(country[i])+"_"+"sti_"+str(country[i])] == False:
                del pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"smkt_"+str(country[i])+"_"+"sti_"+str(country[i])] 
            if pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"vix"+"_"+"sti_"+str(country[i])] == False:
                del pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"vix"+"_"+"sti_"+str(country[i])] 
            if pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"pcf_"+str(country[i])+"_"+"sti_"+str(country[i])] == False:
                del pred_caus_gc["gc_"+str(country[i])+"_"+"lag_"+str(j)+"_"+"pcf_"+str(country[i])+"_"+"sti_"+str(country[i])] 