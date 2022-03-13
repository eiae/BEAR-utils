import os
import pandas as pd

data_dir = {}
outputs_dir = {}
xls = {}
data = {}


def read_data(working_dir, prefix, suffix, sheet,
              country_name, country_name_long):
    for i in range(len(country_name)):
        # shortcuts
        str_country = str(country_name[i])
        str_country_long = str(country_name_long[i])
        # paths
        data_dir["DATA_"+str_country] =\
            os.path.join(working_dir, prefix+str_country_long)
        outputs_dir["OUTPUTS_"+str_country] = \
            os.path.join(working_dir, prefix+str_country_long, "outputs")
        os.chdir(data_dir["DATA_"+str_country])
        # data
        xls["xls_"+str_country] = \
            pd.ExcelFile(prefix+str_country_long+suffix+".xlsx")
        data["data_"+str_country] = \
            pd.read_excel(xls["xls_"+str_country],
                          sheet, index_col=1, header=1)
        data["data_"+str_country].dropna(axis=1, how="all", inplace=True)

    return data_dir, outputs_dir, data
