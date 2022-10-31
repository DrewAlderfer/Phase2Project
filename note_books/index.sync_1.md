```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
```

## Data-Understanding
First thing to do is set up my imports. I have already done a lot of EDA in other notebooks, so
here my main focus is going to be on modeling and seeing what works.

**Now** I am going to pull in the datasets I'll use for my analysis.


```python
house_data = pd.read_pickle('../00-final-dsc-phase-2-project-v2-3/data/house_pickle.pkl')
parcel_data = pd.read_pickle('../00-final-dsc-phase-2-project-v2-3/data/census_col.pkl')
house_data.info(), parcel_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 12 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             21597 non-null  int64  
     1   price          21597 non-null  float64
     2   sqft_living    21597 non-null  int64  
     3   sqft_lot       21597 non-null  int64  
     4   view           21534 non-null  object 
     5   grade          21597 non-null  float64
     6   sqft_above     21597 non-null  int64  
     7   yr_built       21597 non-null  int64  
     8   lat            21597 non-null  float64
     9   long           21597 non-null  float64
     10  sqft_living15  21597 non-null  int64  
     11  sqft_lot15     21597 non-null  int64  
    dtypes: float64(4), int64(7), object(1)
    memory usage: 2.0+ MB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 621954 entries, 0 to 621953
    Data columns (total 11 columns):
     #   Column               Non-Null Count   Dtype  
    ---  ------               --------------   -----  
     0   POC_pct              621238 non-null  float64
     1   median_income        621219 non-null  float64
     2   PREUSE_DESC          614265 non-null  object 
     3   income_3rd           621219 non-null  float64
     4   LifeExpectancy       618976 non-null  float64
     5   TREE_PCT             621238 non-null  float64
     6   osdist_mean          621260 non-null  float64
     7   os_per_person_pctle  462424 non-null  float64
     8   longitude            621954 non-null  float64
     9   latitude             621954 non-null  float64
     10  Shape_Area           621954 non-null  float64
    dtypes: float64(10), object(1)
    memory usage: 52.2+ MB
    




    (None, None)



## Train / Test Split
*Here I am going* to perform the train test split on both sets of data. I don't know if this is the
right move yet but I am going to go ahead with it while attempting to keep things organized in such
a ways as to make them easy to change later.

## Data Merging
I need to merge the two datasets. The way I am going to merge is by binning the latitude and longitude
and then merging on that column. Parcel data will be averaged by bin and the house sale data will
remain unchanged.


```python
def hex_bin(LAT, LONG, base_h):
    span = (LAT.max() - LAT.min(), LONG.max() - LONG.min())
    base_w = round((span[1] - span[0]) * base_h)
    for (lat, lon) in zip(LAT, LONG):
        lat_bins =  int(round((LAT.max() - lat) * base_h))
        long_bins = int(round((LONG.max() - lon) * base_w))
        yield (lat_bins, long_bins)
```


```python
base_lat_max = house_data['lat'].max()
base_lon_max = house_data['long'].max()
base_lat_min = house_data['lat'].min()
base_lon_min = house_data['long'].min()
```


```python
lat_span = base_lat_max - base_lat_min
long_span = base_lon_max - base_lon_min
```


```python
downsample_height = 200
downsample_width = (long_span * downsample_height)/ lat_span
```


```python
def bin_coord(lat, long):
    b_lat = int(round(((lat - base_lat_min) / lat_span) * downsample_height))
    b_long = int(round(((long - base_lon_min) / long_span) * downsample_width))
    return (b_lat, b_long)
```


```python
house_data['bins'] = house_data.apply(lambda x : bin_coord(x['lat'], x['long']), axis=1)
house_data_wb = house_data['bins']
```


```python
house_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 13 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             21597 non-null  int64  
     1   price          21597 non-null  float64
     2   sqft_living    21597 non-null  int64  
     3   sqft_lot       21597 non-null  int64  
     4   view           21534 non-null  object 
     5   grade          21597 non-null  float64
     6   sqft_above     21597 non-null  int64  
     7   yr_built       21597 non-null  int64  
     8   lat            21597 non-null  float64
     9   long           21597 non-null  float64
     10  sqft_living15  21597 non-null  int64  
     11  sqft_lot15     21597 non-null  int64  
     12  bins           21597 non-null  object 
    dtypes: float64(4), int64(7), object(2)
    memory usage: 2.1+ MB
    


```python
points = [[a for (a, b) in house_data_wb],
        [b for (a, b) in house_data_wb]]

sns.scatterplot(x=points[1], y=points[0], s=2)
plt.show()
```


![png](output_12_0.png)



```python
parcel_data['bins'] = parcel_data.apply(lambda x : bin_coord(x['latitude'], x['longitude']), axis=1)
```


```python
parcel_data['bins'].value_counts().head(20)
```




    (172, 57)    311
    (174, 55)    278
    (125, 48)    277
    (166, 45)    247
    (149, 71)    240
    (167, 45)    224
    (173, 57)    223
    (166, 44)    221
    (167, 43)    219
    (142, 71)    217
    (141, 72)    214
    (132, 48)    207
    (118, 52)    206
    (144, 67)    206
    (143, 71)    200
    (172, 69)    200
    (170, 46)    199
    (164, 49)    199
    (163, 54)    198
    (141, 71)    196
    Name: bins, dtype: int64




```python
parcel_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 621954 entries, 0 to 621953
    Data columns (total 12 columns):
     #   Column               Non-Null Count   Dtype  
    ---  ------               --------------   -----  
     0   POC_pct              621238 non-null  float64
     1   median_income        621219 non-null  float64
     2   PREUSE_DESC          614265 non-null  object 
     3   income_3rd           621219 non-null  float64
     4   LifeExpectancy       618976 non-null  float64
     5   TREE_PCT             621238 non-null  float64
     6   osdist_mean          621260 non-null  float64
     7   os_per_person_pctle  462424 non-null  float64
     8   longitude            621954 non-null  float64
     9   latitude             621954 non-null  float64
     10  Shape_Area           621954 non-null  float64
     11  bins                 621954 non-null  object 
    dtypes: float64(10), object(2)
    memory usage: 56.9+ MB
    

## Grouping the Secondary Table
*Now that I have bins* designated for both dataframes I am going to groupby bins on the secondary
frame, taking the mean of each group. Then I will merge the two frames, and start the test process.


```python
parcel_cat_group = parcel_data[['PREUSE_DESC', 'bins']].groupby('bins')['PREUSE_DESC'].apply(list)
```


```python
parcel_grouped = parcel_data.drop('PREUSE_DESC', axis=1).groupby('bins').mean()
```


```python
parcel_grouped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>POC_pct</th>
      <th>median_income</th>
      <th>income_3rd</th>
      <th>LifeExpectancy</th>
      <th>TREE_PCT</th>
      <th>osdist_mean</th>
      <th>os_per_person_pctle</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>Shape_Area</th>
    </tr>
    <tr>
      <th>bins</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(-22, 345)</th>
      <td>0.167722</td>
      <td>92717.0</td>
      <td>2.0</td>
      <td>77.4</td>
      <td>86.771175</td>
      <td>0.0</td>
      <td>0.998113</td>
      <td>-121.446184</td>
      <td>47.086427</td>
      <td>7.515090e+04</td>
    </tr>
    <tr>
      <th>(-22, 347)</th>
      <td>0.167722</td>
      <td>92717.0</td>
      <td>2.0</td>
      <td>77.4</td>
      <td>86.771175</td>
      <td>0.0</td>
      <td>0.998113</td>
      <td>-121.441819</td>
      <td>47.088245</td>
      <td>2.776598e+07</td>
    </tr>
    <tr>
      <th>(-22, 366)</th>
      <td>0.167722</td>
      <td>92717.0</td>
      <td>2.0</td>
      <td>77.4</td>
      <td>86.771175</td>
      <td>0.0</td>
      <td>0.998113</td>
      <td>-121.380911</td>
      <td>47.088922</td>
      <td>1.293861e+06</td>
    </tr>
    <tr>
      <th>(-21, 356)</th>
      <td>0.167722</td>
      <td>92717.0</td>
      <td>2.0</td>
      <td>77.4</td>
      <td>86.771175</td>
      <td>0.0</td>
      <td>0.998113</td>
      <td>-121.413650</td>
      <td>47.091408</td>
      <td>2.740976e+07</td>
    </tr>
    <tr>
      <th>(-21, 366)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-121.379857</td>
      <td>47.090316</td>
      <td>2.621410e+07</td>
    </tr>
  </tbody>
</table>
</div>




```python
parcel_cat_group.head(20)
```




    bins
    (-22, 345)                                               [None]
    (-22, 347)    [Vacant(Single-family)                        ...
    (-22, 366)    [Vacant(Single-family)                        ...
    (-21, 356)    [Vacant(Single-family)                        ...
    (-21, 366)                                               [None]
    (-21, 367)    [Vacant(Single-family)                        ...
    (-20, 341)    [None, None, Vacant(Single-family)            ...
    (-20, 362)                                               [None]
    (-20, 366)                                               [None]
    (-17, 359)                                               [None]
    (-17, 360)                                         [None, None]
    (-16, 335)    [None, Vacant(Single-family)                  ...
    (-16, 340)    [Vacant(Single-family)                        ...
    (-15, 349)    [Vacant(Single-family)                        ...
    (-15, 357)    [Vacant(Single-family)                        ...
    (-14, 358)                                               [None]
    (-13, 332)    [None, Vacant(Single-family)                  ...
    (-12, 301)                                               [None]
    (-12, 328)    [None, Vacant(Single-family)                  ...
    (-12, 329)                                               [None]
    Name: PREUSE_DESC, dtype: object




```python
parcel_grouped_m = parcel_grouped.merge(parcel_cat_group, how='inner', on='bins')
```


```python
color = sns.color_palette(palette='crest', as_cmap=True)
```


```python
sns.scatterplot(data=parcel_grouped, x='longitude', y='latitude', hue='LifeExpectancy', palette=color, s=2)
plt.savefig('./life_expc.png')
plt.show()
```


![png](output_23_0.png)



```python
comb_parcel_house = house_data.merge(parcel_grouped_m, how='left', on='bins')
comb_parcel_house_dna = comb_parcel_house.dropna(axis=0, how='any')
comb_parcel_house_dna.to_pickle('./test_train_set.pkl')
comb_parcel_house_dna.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 18256 entries, 0 to 21595
    Data columns (total 24 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   id                   18256 non-null  int64  
     1   price                18256 non-null  float64
     2   sqft_living          18256 non-null  int64  
     3   sqft_lot             18256 non-null  int64  
     4   view                 18256 non-null  object 
     5   grade                18256 non-null  float64
     6   sqft_above           18256 non-null  int64  
     7   yr_built             18256 non-null  int64  
     8   lat                  18256 non-null  float64
     9   long                 18256 non-null  float64
     10  sqft_living15        18256 non-null  int64  
     11  sqft_lot15           18256 non-null  int64  
     12  bins                 18256 non-null  object 
     13  POC_pct              18256 non-null  float64
     14  median_income        18256 non-null  float64
     15  income_3rd           18256 non-null  float64
     16  LifeExpectancy       18256 non-null  float64
     17  TREE_PCT             18256 non-null  float64
     18  osdist_mean          18256 non-null  float64
     19  os_per_person_pctle  18256 non-null  float64
     20  longitude            18256 non-null  float64
     21  latitude             18256 non-null  float64
     22  Shape_Area           18256 non-null  float64
     23  PREUSE_DESC          18256 non-null  object 
    dtypes: float64(14), int64(7), object(3)
    memory usage: 3.5+ MB
    


```python
# Set up target variables
X = comb_parcel_house_dna.drop('grade', axis=1)
y = comb_parcel_house_dna['grade']
# parcel = parcel_data.drop('median_income')
# income = parcel_data['median_income']
# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=133)
# parcel_train, parcel_test, median_income_train, median_income_test = train_test_split(parcel, income, test_size=.33, random_state=133)
```


```python
X_num_train = X_train.select_dtypes([np.float64, np.int64])
X_obj_train = X_train.select_dtypes(np.object)
```


```python
y_train.value_counts().sum()
```




    12231




```python
X_num_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 12231 entries, 10068 to 20580
    Data columns (total 20 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   id                   12231 non-null  int64  
     1   price                12231 non-null  float64
     2   sqft_living          12231 non-null  int64  
     3   sqft_lot             12231 non-null  int64  
     4   sqft_above           12231 non-null  int64  
     5   yr_built             12231 non-null  int64  
     6   lat                  12231 non-null  float64
     7   long                 12231 non-null  float64
     8   sqft_living15        12231 non-null  int64  
     9   sqft_lot15           12231 non-null  int64  
     10  POC_pct              12231 non-null  float64
     11  median_income        12231 non-null  float64
     12  income_3rd           12231 non-null  float64
     13  LifeExpectancy       12231 non-null  float64
     14  TREE_PCT             12231 non-null  float64
     15  osdist_mean          12231 non-null  float64
     16  os_per_person_pctle  12231 non-null  float64
     17  longitude            12231 non-null  float64
     18  latitude             12231 non-null  float64
     19  Shape_Area           12231 non-null  float64
    dtypes: float64(13), int64(7)
    memory usage: 2.0 MB
    


```python
scaler = StandardScaler()
scaler.fit(X_num_train)
X_train_scaled = scaler.transform(X_num_train)
# X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_num_train.columns, index=X_num_train.index)
X_train_scaled.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>POC_pct</th>
      <th>median_income</th>
      <th>income_3rd</th>
      <th>LifeExpectancy</th>
      <th>TREE_PCT</th>
      <th>osdist_mean</th>
      <th>os_per_person_pctle</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>Shape_Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10068</th>
      <td>-0.730177</td>
      <td>1.093213</td>
      <td>1.933771</td>
      <td>0.590941</td>
      <td>2.490933</td>
      <td>0.568767</td>
      <td>1.240088</td>
      <td>0.537891</td>
      <td>2.529542</td>
      <td>0.781417</td>
      <td>0.623907</td>
      <td>2.212576</td>
      <td>0.996759</td>
      <td>0.982719</td>
      <td>-0.505323</td>
      <td>1.069484</td>
      <td>1.564231</td>
      <td>0.529244</td>
      <td>1.229711</td>
      <td>1.268832</td>
    </tr>
    <tr>
      <th>2617</th>
      <td>-0.991833</td>
      <td>-0.702776</td>
      <td>-1.337473</td>
      <td>-0.286422</td>
      <td>-1.127478</td>
      <td>-1.371809</td>
      <td>-0.543179</td>
      <td>0.039185</td>
      <td>-0.872997</td>
      <td>-0.287577</td>
      <td>1.037163</td>
      <td>-1.473056</td>
      <td>-1.690792</td>
      <td>0.221115</td>
      <td>-1.564861</td>
      <td>-0.907843</td>
      <td>-0.326349</td>
      <td>0.031593</td>
      <td>-0.552198</td>
      <td>-0.201072</td>
    </tr>
    <tr>
      <th>12413</th>
      <td>-1.274353</td>
      <td>1.958332</td>
      <td>0.084342</td>
      <td>-0.109690</td>
      <td>-0.311562</td>
      <td>-0.520679</td>
      <td>0.781554</td>
      <td>-1.380741</td>
      <td>0.256418</td>
      <td>-0.136893</td>
      <td>-0.566508</td>
      <td>0.620760</td>
      <td>0.996759</td>
      <td>1.172317</td>
      <td>-0.020831</td>
      <td>-0.854972</td>
      <td>1.725286</td>
      <td>-1.386621</td>
      <td>0.786196</td>
      <td>0.187872</td>
    </tr>
    <tr>
      <th>20404</th>
      <td>1.367413</td>
      <td>0.118955</td>
      <td>-0.289820</td>
      <td>-0.329054</td>
      <td>-0.134189</td>
      <td>1.419897</td>
      <td>-0.210742</td>
      <td>0.946553</td>
      <td>0.828273</td>
      <td>-0.370022</td>
      <td>0.178696</td>
      <td>0.878433</td>
      <td>0.996759</td>
      <td>0.951120</td>
      <td>3.075231</td>
      <td>-1.038892</td>
      <td>1.567732</td>
      <td>0.943735</td>
      <td>-0.210437</td>
      <td>-0.310662</td>
    </tr>
    <tr>
      <th>11110</th>
      <td>0.898146</td>
      <td>0.208129</td>
      <td>0.565407</td>
      <td>-0.210700</td>
      <td>0.977349</td>
      <td>1.079445</td>
      <td>0.883292</td>
      <td>1.188979</td>
      <td>0.899755</td>
      <td>-0.296277</td>
      <td>0.746420</td>
      <td>1.248619</td>
      <td>0.996759</td>
      <td>0.003131</td>
      <td>0.667881</td>
      <td>-0.807881</td>
      <td>-1.153207</td>
      <td>1.195639</td>
      <td>0.878992</td>
      <td>-0.125343</td>
    </tr>
    <tr>
      <th>17798</th>
      <td>1.367502</td>
      <td>0.101652</td>
      <td>0.062961</td>
      <td>-0.183814</td>
      <td>-0.524410</td>
      <td>-0.112137</td>
      <td>0.499269</td>
      <td>0.808024</td>
      <td>0.013379</td>
      <td>-0.191131</td>
      <td>0.205718</td>
      <td>0.854260</td>
      <td>0.996759</td>
      <td>0.710962</td>
      <td>0.182502</td>
      <td>-0.299792</td>
      <td>0.099135</td>
      <td>0.806024</td>
      <td>0.494292</td>
      <td>-0.197000</td>
    </tr>
    <tr>
      <th>16703</th>
      <td>-1.127991</td>
      <td>-0.448031</td>
      <td>0.747143</td>
      <td>-0.176293</td>
      <td>-0.240613</td>
      <td>0.262360</td>
      <td>-0.829046</td>
      <td>0.593303</td>
      <td>0.556642</td>
      <td>-0.202822</td>
      <td>0.190644</td>
      <td>-0.358026</td>
      <td>-0.347016</td>
      <td>-0.312865</td>
      <td>0.216737</td>
      <td>-0.518713</td>
      <td>0.962027</td>
      <td>0.590507</td>
      <td>-0.820141</td>
      <td>-0.208810</td>
    </tr>
    <tr>
      <th>5844</th>
      <td>-0.349279</td>
      <td>2.597190</td>
      <td>2.788998</td>
      <td>0.116067</td>
      <td>3.436922</td>
      <td>-0.010001</td>
      <td>1.430666</td>
      <td>-1.152167</td>
      <td>1.428721</td>
      <td>0.218577</td>
      <td>-0.885080</td>
      <td>0.028989</td>
      <td>0.996759</td>
      <td>0.066330</td>
      <td>-0.429301</td>
      <td>-0.721310</td>
      <td>0.706441</td>
      <td>-1.152598</td>
      <td>1.431784</td>
      <td>0.007117</td>
    </tr>
    <tr>
      <th>5762</th>
      <td>1.600463</td>
      <td>0.228093</td>
      <td>0.212626</td>
      <td>0.371060</td>
      <td>0.587128</td>
      <td>0.432586</td>
      <td>1.518074</td>
      <td>0.607156</td>
      <td>0.999829</td>
      <td>0.763745</td>
      <td>-0.813799</td>
      <td>1.858054</td>
      <td>0.996759</td>
      <td>0.635123</td>
      <td>1.295016</td>
      <td>1.587060</td>
      <td>-1.821415</td>
      <td>0.612336</td>
      <td>1.518772</td>
      <td>0.282871</td>
    </tr>
    <tr>
      <th>2748</th>
      <td>-0.755368</td>
      <td>-0.538536</td>
      <td>-0.918412</td>
      <td>0.210120</td>
      <td>-0.971390</td>
      <td>0.841129</td>
      <td>-1.703843</td>
      <td>-0.563417</td>
      <td>-0.215363</td>
      <td>0.388564</td>
      <td>0.052191</td>
      <td>-0.702805</td>
      <td>-0.347016</td>
      <td>-1.387252</td>
      <td>0.594224</td>
      <td>0.326725</td>
      <td>-0.911107</td>
      <td>-0.568309</td>
      <td>-1.709309</td>
      <td>0.118792</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.linear_model import LinearRegression
```


```python
model_sk = LinearRegression()
```


```python
model_sk.fit(X_train_scaled, y_train)
```




    LinearRegression()




```python
model_sk.score(X_train_scaled, y_train)
```




    0.7393317953638282




```python
train_preds = model_sk.predict(X_train_scaled)

r2_score(y_train, train_preds)
```




    0.7393317953638282




```python
dict(zip(X_train.columns, model_sk.coef_))
```




    {'id': 0.019877009072816282,
     'price': 0.31231914758324103,
     'sqft_living': 0.18480895427840743,
     'sqft_lot': 0.009261569231549532,
     'view': 0.25950951023762914,
     'sqft_above': 0.31509094564064966,
     'yr_built': 1.4797592615993318,
     'lat': -1.1113158362991518,
     'long': 0.2458783807259897,
     'sqft_living15': -0.012428540227437005,
     'sqft_lot15': -0.015781777247554692,
     'bins': 0.03582411319515027,
     'POC_pct': -0.0036553335319543905,
     'median_income': 0.053346260762527443,
     'income_3rd': 0.05228512762651232,
     'LifeExpectancy': -0.01818672668895527,
     'TREE_PCT': 0.01322167885174718,
     'osdist_mean': 0.9539820381643114,
     'os_per_person_pctle': -1.4762396809390237,
     'longitude': 0.0010138308524793923}


