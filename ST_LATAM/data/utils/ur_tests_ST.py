from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


adf_test = {}
kpss_test = {}
non_stat_adf = {}
non_stat_kpss = {}


def ur_test_data(data, country_name, signif):
    for i in range(len(country_name)):
        # shortcuts
        str_country = str(country_name[i])
        data_country = data["data_"+str_country]
        # UR tests - ADF null UR / KPSS null not UR
        # ADF lag length criteria Schwarz 1978
        # KPSS lag length criteria Schwert 1989
        for j in data_country.columns:
            # shortcuts
            str_adf = "adf_"+j
            str_kpss = "kpss_"+j
            adf_test[str_adf] = adfuller(data_country[j],
                                         regression="c",
                                         autolag="AIC",
                                         regresults=True)[1]
            kpss_test[str_kpss] = kpss(data_country[j],
                                       regression="c")[1]
            # keep only UR results
            for k in range(len(signif)):
                str_signif = str(signif[k])
                non_stat_adf[str_adf+"_"+str_signif] = adf_test[str_adf] \
                    <= signif[k]
                non_stat_kpss[str_kpss+"_"+str_signif] = kpss_test[str_kpss] \
                    > signif[k]
                if non_stat_adf[str_adf+"_"+str_signif]:
                    del non_stat_adf[str_adf+"_"+str_signif]
                if non_stat_kpss[str_kpss+"_"+str_signif]:
                    del non_stat_kpss[str_kpss+"_"+str_signif]

    return non_stat_adf, non_stat_kpss, adf_test, kpss_test
