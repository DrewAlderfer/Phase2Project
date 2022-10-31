```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
```

POC_pct
median_income
PREUSE_DESC
income_3rd
LifeExpectancy
TREE_PCT
osdist_mean
os_per_person_pctle
longitude
latitude
Shape_Area


```python
df = pd.read_pickle('./data/census_lci.pkl')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 621954 entries, 0 to 621953
    Data columns (total 49 columns):
     #   Column                    Non-Null Count   Dtype  
    ---  ------                    --------------   -----  
     0   OBJECTID                  621954 non-null  int64  
     1   PIN                       621954 non-null  object 
     2   ADDR_FULL                 565135 non-null  object 
     3   ZIP5                      621260 non-null  float64
     4   CTYNAME                   518843 non-null  object 
     5   PREUSE_DESC               614265 non-null  object 
     6   KCA_ACRES                 614390 non-null  float64
     7   GEO_ID_GRP                621260 non-null  float64
     8   GEO_ID_TRT                621260 non-null  float64
     9   median_income             621219 non-null  float64
     10  income_pctle              621219 non-null  float64
     11  income_3rd                621219 non-null  float64
     12  below200FedPov_pct        621238 non-null  float64
     13  below200FedPov_pctle      621238 non-null  float64
     14  limitedEng_pct            621238 non-null  float64
     15  limitedEng_pctle          621238 non-null  float64
     16  POC_pct                   621238 non-null  float64
     17  POC_pctle                 621238 non-null  float64
     18  underAge5_pct             621238 non-null  float64
     19  underAge5_pctle           621238 non-null  float64
     20  disabled_pct              621238 non-null  float64
     21  disabled_pctle            621238 non-null  float64
     22  disabled_uninsured_pct    621238 non-null  float64
     23  disabled_uninsured_pctle  621238 non-null  float64
     24  foodstamp_pct             621238 non-null  float64
     25  foodstamp_pctle           621238 non-null  float64
     26  LifeExpectancy            618976 non-null  float64
     27  life_exp_pctle            618976 non-null  float64
     28  os_per_person_pctle       462424 non-null  float64
     29  TREE_PCT                  621238 non-null  float64
     30  tree_pctle                621238 non-null  float64
     31  displacement_risk         621211 non-null  object 
     32  hospZ_pctle               620779 non-null  float64
     33  hospZ_3rd                 620779 non-null  float64
     34  SchoolName_OSPI           620918 non-null  object 
     35  PctFreeReduced            619227 non-null  float64
     36  PctFreeReduced_pctle      619227 non-null  float64
     37  UGASIDE                   620104 non-null  object 
     38  exclude                   430 non-null     float64
     39  osdist_mean               621260 non-null  float64
     40  resdist_mean              621260 non-null  float64
     41  OpenSpaceAccessYN         621260 non-null  object 
     42  ResAccessYN               621260 non-null  object 
     43  CriteriaIncHospYN         621221 non-null  object 
     44  CriteriaAllYN             621259 non-null  object 
     45  Shape_Length              621954 non-null  float64
     46  Shape_Area                621954 non-null  float64
     47  longitude                 621954 non-null  float64
     48  latitude                  621954 non-null  float64
    dtypes: float64(37), int64(1), object(11)
    memory usage: 232.5+ MB
    


```python
df_c = df[['POC_pct', 'median_income', 'PREUSE_DESC','income_3rd', 'LifeExpectancy', 'TREE_PCT', 'osdist_mean', 'os_per_person_pctle', 'longitude', 'latitude', 'Shape_Area']]
df_c.to_pickle('./data/census_col.pkl')
df_c.info()
```

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
    


```python
df_c['PREUSE_DESC'].value_counts().head(20)
```




    Single Family(Res Use/Zone)                           472355
    Vacant(Single-family)                                  37894
    Townhouse Plat                                         28396
                                                            9501
    Mobile Home                                             6518
    Apartment                                               6441
    Duplex                                                  6303
    Single Family(C/I Zone)                                 3844
    Office Building                                         3221
    Vacant(Commercial)                                      3160
    Condominium(Residential)                                2825
    Retail Store                                            2690
    Warehouse                                               2515
    4-Plex                                                  2170
    Triplex                                                 1751
    Vacant(Multi-family)                                    1521
    Vacant(Industrial)                                      1432
    Apartment(Mixed Use)                                    1365
    Parking(Assoc)                                          1210
    Church/Welfare/Relig Srvc                               1183
    Name: PREUSE_DESC, dtype: int64



These are the columns I want to test. Now I am going to bin the deg coordinates.


```python
max_lat = 47.7776
min_lat = 47.1559
span_lat = 0.621699999999997
max_long = -121.315
min_long = -122.519
span_long = 1.2040000000000077
df_f = df_c[((df_c['latitude'] < max_lat) & (df_c['latitude'] > min_lat)) & ((df_c['longitude'] < max_long) & (df_c['longitude'] > min_long))]
df_f.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 621051 entries, 0 to 621953
    Data columns (total 11 columns):
     #   Column               Non-Null Count   Dtype  
    ---  ------               --------------   -----  
     0   POC_pct              620412 non-null  float64
     1   median_income        620393 non-null  float64
     2   PREUSE_DESC          613482 non-null  object 
     3   income_3rd           620393 non-null  float64
     4   LifeExpectancy       618154 non-null  float64
     5   TREE_PCT             620412 non-null  float64
     6   osdist_mean          620434 non-null  float64
     7   os_per_person_pctle  461614 non-null  float64
     8   longitude            621051 non-null  float64
     9   latitude             621051 non-null  float64
     10  Shape_Area           621051 non-null  float64
    dtypes: float64(10), object(1)
    memory usage: 56.9+ MB
    


```python
map_height = 150 
map_width = round((span_long/span_lat) * map_height)
print(f"map height {map_height} times map_width {map_width} = {map_height * map_width}")
```

    map height 150 times map_width 290 = 43500
    


```python
x_bin = lambda x : int(round(((x-min_long)/span_long)*map_width))
y_bin = lambda y : int(round(((y-min_lat)/span_lat)*map_height))
df_f['coord_bin'] = df_f[['latitude','longitude']].apply(lambda x : str(y_bin(x['latitude'])) + str(x_bin(x['longitude'])), axis=1)
```

    <ipython-input-90-5ddefada08a8>:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_f['coord_bin'] = df_f[['latitude','longitude']].apply(lambda x : str(y_bin(x['latitude'])) + str(x_bin(x['longitude'])), axis=1)
    


```python
df_f.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 621051 entries, 0 to 621953
    Data columns (total 12 columns):
     #   Column               Non-Null Count   Dtype  
    ---  ------               --------------   -----  
     0   POC_pct              620412 non-null  float64
     1   median_income        620393 non-null  float64
     2   PREUSE_DESC          613482 non-null  object 
     3   income_3rd           620393 non-null  float64
     4   LifeExpectancy       618154 non-null  float64
     5   TREE_PCT             620412 non-null  float64
     6   osdist_mean          620434 non-null  float64
     7   os_per_person_pctle  461614 non-null  float64
     8   longitude            621051 non-null  float64
     9   latitude             621051 non-null  float64
     10  Shape_Area           621051 non-null  float64
     11  coord_bin            621051 non-null  object 
    dtypes: float64(10), object(2)
    memory usage: 81.6+ MB
    


```python
data_bin = df_f.drop('PREUSE_DESC', axis=1).groupby('coord_bin').mean()
data_bin.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 17337 entries, 0133 to 9999
    Data columns (total 10 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   POC_pct              17276 non-null  float64
     1   median_income        17273 non-null  float64
     2   income_3rd           17273 non-null  float64
     3   LifeExpectancy       17247 non-null  float64
     4   TREE_PCT             17276 non-null  float64
     5   osdist_mean          17286 non-null  float64
     6   os_per_person_pctle  16394 non-null  float64
     7   longitude            17337 non-null  float64
     8   latitude             17337 non-null  float64
     9   Shape_Area           17337 non-null  float64
    dtypes: float64(10)
    memory usage: 1.5+ MB
    


```python
fig, ax = plt.subplots(figsize=(20,12.5))
sns.scatterplot(data=data_bin, x='longitude', y='latitude', hue='median_income', markers='markers', s=65, alpha=.3)
plt.show();
```


![png](output_11_0.png)



```python
fig, ax = plt.subplots(figsize=(20,12.5))
data_bin['markers'] = ["s" for num in  range(data_bin.shape[0])]
color = sns.color_palette('light:#385', as_cmap=True)
sns.scatterplot(data=data_bin, x='longitude', y='latitude', hue='TREE_PCT', markers='markers', s=65, alpha=.3, palette=color)
plt.show();
```


![png](output_12_0.png)

