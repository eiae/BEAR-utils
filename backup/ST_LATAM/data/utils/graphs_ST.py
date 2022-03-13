import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

summary = {}
figname = []


def plot_data(output_dir, colours, palettes, data,
              country_name, country_name_Long, varname, ref):
    for i in range(len(country_name)):
        # shortcuts
        str_country = str(country_name[i])
        save_path = output_dir["OUTPUTS_"+str_country]
        save_file = str_country+".pdf"
        data_country = data["data_"+str_country]
        # summary stats
        summary["summary_"+str_country] = data_country.describe()
        # heatmaps
        fig = plt.figure(figsize=(15, 7))
        sns.heatmap(data_country.corr(),
                    cmap=palettes[i], linecolor='white',
                    linewidths=1, annot=True).get_figure()
        plt.title(country_name_Long[i], pad=20)
        fig.savefig(os.path.join(save_path, "heatmap"+save_file),
                    dpi=200, bbox_inches="tight")
        # scatterplots
        for n in range(1, len(varname)):
            figname.append("scatterplot_"+ref+"_"+varname[n])
            fig = sns.jointplot(x=data_country[ref+str_country],
                                y=data_country[varname[n]+str_country],
                                kind='reg', color=colours[i])
            fig.annotate(stats.pearsonr)
            fig.savefig(os.path.join(save_path, figname[n-1]+save_file),
                        dpi=200)
        # lineplots
        for j in range(data_country.shape[1]):
            save_file2 = str(data_country.columns[j])+".pdf"
            fig, ax = plt.subplots(1, figsize=(15, 7))
            ax.plot(data_country.iloc[:, j], color=colours[i], linewidth=2)
            ax.patch.set_facecolor("white")
            ax.grid(color="k", alpha=0.2, linewidth=0.5, linestyle="--")
            ax.set_title(data_country.columns[j])
            fig.savefig(os.path.join(save_path, "lineplot_"+save_file2),
                        dpi=200, bbox_inches="tight")

    return summary
