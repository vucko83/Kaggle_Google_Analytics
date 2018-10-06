import os
idir = "Data/"
for x in os.listdir(idir):
        f = idir + x
        s = os.stat(f)
        num_lines = sum(1 for line in open(f))
        print(x + ":" + str(round(s.st_size / (1024 * 1024)) ) + " MB : " + str(num_lines) + " lines")
        
# Read+CSV+Into+Dataframe
import numpy as np 
import pandas as pd 
ifile = idir + "train.csv"


import json
from pandas.io.json import json_normalize



def ld_dn_df(csv_file, json_cols, rows_to_load = 100 ): 

    # Apply converter to convert JSON format data into JSON object
    json_conv = {col: json.loads for col in (json_cols)}

    # Read the CSV with the new converter
    df = pd.read_csv(csv_file, 
        dtype={'fullVisitorId': 'object'},
        converters=json_conv, 
        nrows=rows_to_load, 
        low_memory = False
        )
    
    for jcol in json_cols: 
        tdf = pd.concat([pd.DataFrame(json_normalize(x)) for x in df[jcol]], ignore_index=True) #json_normalize(df[jcol])
        tdf.columns = [jcol + "_" + col for col in tdf.columns]
        df = df.merge(tdf, left_index=True, right_index=True)
        
    df = df.drop(json_cols, axis = 1)
    return df


rows_to_load = 1000
json_cols = ["totals", "device", "geoNetwork", "trafficSource"]
train_df =  ld_dn_df("Data/train.csv", json_cols, rows_to_load)

train_df.head()