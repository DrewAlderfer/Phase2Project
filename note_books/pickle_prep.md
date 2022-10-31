```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from statsmodels.api import stats
import matplotlib.pyplot as plt
import seaborn as sns
```


```
# Reading the clean datasets into the flow
house_data = pd.read_pickle('../00-final-dsc-phase-2-project-v2-3/data/house_pickle.pkl')
parcel_data = pd.read_pickle('../00-final-dsc-phase-2-project-v2-3/data/census_col.pkl')
```


```
# In this section the constants are set up to form the grid for binning
base_lat_max = house_data['lat'].max()
base_lon_max = house_data['long'].max()
base_lat_min = house_data['lat'].min()
base_lon_min = house_data['long'].min()
```


```
# Lat/Long Domains calculated
lat_span = base_lat_max - base_lat_min
long_span = base_lon_max - base_lon_min
```


```
# The base 'downsampling' for grid. This will set the number of vertical bins as 200
downsample_height = 200
downsample_width = (long_span * downsample_height)/ lat_span
```


```
# Binning Function for Lat/Long Coordinates
def bin_coord(lat, long):
    b_lat = int(round(((lat - base_lat_min) / lat_span) * downsample_height))
    b_long = int(round(((long - base_lon_min) / long_span) * downsample_width))
    return (b_lat, b_long)
```


```
# Binning the Coordinates and inserting them into new rows
house_data['bins'] = house_data.apply(lambda x : bin_coord(x['lat'], x['long']), axis=1)
parcel_data['bins'] = parcel_data.apply(lambda x : bin_coord(x['latitude'], x['longitude']), axis=1)
```


```
# Splitting Categrocial and Numerical Data and then Grouping by Bin
parcel_cat_group = parcel_data[['PREUSE_DESC', 'bins']].groupby('bins')['PREUSE_DESC'].apply(list)
parcel_grouped = parcel_data.drop('PREUSE_DESC', axis=1).groupby('bins').mean()
parcel_grouped_m = parcel_grouped.merge(parcel_cat_group, how='inner', on='bins')
```


```
# Merging the two Datasets into one
comb_parcel_house = house_data.merge(parcel_grouped_m, how='left', on='bins')
comb_parcel_house_dna = comb_parcel_house.dropna(axis=0, how='any')
comb_parcel_house_dna = pd.read_pickle('./test_train_set02.pkl')
data01_corr = comb_parcel_house_dna.corr()
```


```
fig, ax = plt.subplots(figsize=(40,40))
sns.heatmap(data01_corr, annot=False)
plt.show();
```
