# =============================================================================
import pandas as pd
import os
os.chdir("C:\\Users\\MONSTER\\Google Drive\\ders\\Data Mining\\project")
# =============================================================================
data = pd.read_csv("data.csv") 

columns = list(data.columns)
columns[3] = "Area"
columns[4] = "Population Density"
columns[5] = "Coastline"
columns[7] = "Infant Mortality"
columns[8] = "GDP per Capita"
columns[9] = "Literacy"
columns[10] = "Phones"
columns[11] = "Arable"
columns[12] = "Crops"
columns[13] = "Other"

data.columns = columns
data.iloc[:,0] = data.iloc[:,0].str.strip()
data.iloc[:,0] = data.iloc[:,0].astype('category')
data.iloc[:,1] = data.iloc[:,1].astype('category')
data.iloc[:,4] = data.iloc[:,4].str.replace(",",".").astype(float)
data.iloc[:,5] = data.iloc[:,5].str.replace(",",".").astype(float)
data.iloc[:,6] = data.iloc[:,6].str.replace(",",".").astype(float)
data.iloc[:,7] = data.iloc[:,7].str.replace(",",".").astype(float)
data.iloc[:,9] = data.iloc[:,9].str.replace(",",".").astype(float)
data.iloc[:,10] = data.iloc[:,10].str.replace(",",".").astype(float)
data.iloc[:,11] = data.iloc[:,11].str.replace(",",".").astype(float)
data.iloc[:,12] = data.iloc[:,12].str.replace(",",".").astype(float)
data.iloc[:,13] = data.iloc[:,13].str.replace(",",".").astype(float)
data.iloc[:,14] = data.iloc[:,14].str.replace(",",".").astype(float)
data.iloc[:,15] = data.iloc[:,15].str.replace(",",".").astype(float)
data.iloc[:,16] = data.iloc[:,16].str.replace(",",".").astype(float)
data.iloc[:,17] = data.iloc[:,17].str.replace(",",".").astype(float)
data.iloc[:,18] = data.iloc[:,18].str.replace(",",".").astype(float)
data.iloc[:,19] = data.iloc[:,19].str.replace(",",".").astype(float)
# =============================================================================
life = pd.read_csv("life.csv", encoding='latin-1') 
life.iloc[:,0] = life.iloc[:,0].astype('category')
data = pd.merge(data, life, how='left', on="Country") 
del life
# =============================================================================
for col in data.columns.values:
    if data[col].isnull().sum() == 0:
        continue
    if col == "Climate":
        value = data.groupby("Region")["Climate"].apply(lambda x: x.mode().max())
    elif col == "Life":
        value = data.groupby("Region")["Life"].apply(lambda x: x.mode().max())
    else:
        value = data.groupby("Region")[col].median()
    for region in data["Region"].unique():
        data[col].loc[(data[col].isnull())&(data["Region"]==region)] = value[region]
        
data.drop("Country", axis=1, inplace=True)            

data.to_csv("data2.csv",index=False)
# =============================================================================
del region, value, data, col, columns