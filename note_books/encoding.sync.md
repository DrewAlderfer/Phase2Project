```python
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


```python
data = pd.read_pickle('./test_train_set.pkl')
data.info()
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
cat_data = data.select_dtypes(np.object).join(data[['grade', 'yr_built']])
cat_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 18256 entries, 0 to 21595
    Data columns (total 5 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   view         18256 non-null  object 
     1   bins         18256 non-null  object 
     2   PREUSE_DESC  18256 non-null  object 
     3   grade        18256 non-null  float64
     4   yr_built     18256 non-null  int64  
    dtypes: float64(1), int64(1), object(3)
    memory usage: 1.5+ MB
    


```python
# def dict_to_columns(df, col_name):
cat_data['PREUSE_DESC'][134]
```




    ['Condominium(Residential)                          ',
     'Condominium(Residential)                          ',
     'Condominium(Residential)                          ',
     'Single Family(Res Use/Zone)                       ',
     '4-Plex                                            ',
     'Apartment                                         ',
     'Apartment                                         ',
     'Apartment                                         ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     None,
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Triplex                                           ',
     'Triplex                                           ',
     'Apartment                                         ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Post Office/Post Service                          ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(C/I Zone)                           ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Apartment                                         ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(Res Use/Zone)                       ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     '4-Plex                                            ',
     'Single Family(Res Use/Zone)                       ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Duplex                                            ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Triplex                                           ',
     'Single Family(Res Use/Zone)                       ',
     'Triplex                                           ',
     'Apartment                                         ',
     'Duplex                                            ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(C/I Zone)                           ',
     'Apartment                                         ',
     'Apartment                                         ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Duplex                                            ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(Res Use/Zone)                       ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Duplex                                            ',
     'Single Family(Res Use/Zone)                       ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(Res Use/Zone)                       ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(C/I Zone)                           ',
     'Single Family(Res Use/Zone)                       ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(Res Use/Zone)                       ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Apartment                                         ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(Res Use/Zone)                       ',
     'Triplex                                           ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Duplex                                            ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Duplex                                            ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Church/Welfare/Relig Srvc                         ',
     '4-Plex                                            ',
     'Apartment                                         ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(Res Use/Zone)                       ',
     'Duplex                                            ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Apartment                                         ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Townhouse Plat                                    ',
     'Single Family(Res Use/Zone)                       ',
     'Single Family(Res Use/Zone)                       ',
     'Townhouse Plat                                    ',
     'Condominium(Residential)                          ',
     'Condominium(Residential)                          ',
     'Condominium(Residential)                          ']




```python

def counter(a_list):
    result = {}
    unique = list(set(a_list))
    for item in unique:
        count = a_list.count(item)
        try:
            id = item.rstrip()
        except AttributeError:
            continue
        if id == "":
            continue
        result.update({f'{id}': count})
    return result

def key_counter(count_dict):
    for key, val in count_dict.items():
        try:
            master_count[key] = master_count[key] + val
        except KeyError:
            master_count.update({key: val})
```


```python
master_count = {}
new_col_names = []
cat_data['use_counts'] = cat_data['PREUSE_DESC'].map(counter)
cat_data['use_counts'].map(key_counter)
master_list = list(master_count.items())
master_list.sort(reverse=True, key=lambda x : x[1])
for item, count in master_list:
    if count > 1000:
        new_col_names.append(item)
        print(f"{item}: {count}")
```

    Single Family(Res Use/Zone): 950089
    Townhouse Plat: 120831
    Vacant(Single-family): 34433
    Duplex: 17419
    Apartment: 14000
    Single Family(C/I Zone): 6335
    Triplex: 6117
    Condominium(Residential): 5292
    4-Plex: 4273
    Mobile Home: 3159
    Retail Store: 2804
    Vacant(Multi-family): 2657
    Office Building: 2380
    Apartment(Mixed Use): 2208
    Vacant(Commercial): 2052
    Church/Welfare/Relig Srvc: 1905
    Parking(Assoc): 1320
    Park, Public(Zoo/Arbor): 1152
    


```python
def handler(col_name, col_dict):
    if col_name in col_dict.keys():
        return col_dict[col_name]
    return 0

def dict_to_col(df, name_list):
    for name in name_list:
        df[name] = df.apply(lambda x : handler(name, x['use_counts']), axis=1)

dict_to_col(cat_data, new_col_names)
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 18256 entries, 0 to 21595
    Data columns (total 24 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   view                         18256 non-null  object 
     1   bins                         18256 non-null  object 
     2   PREUSE_DESC                  18256 non-null  object 
     3   grade                        18256 non-null  float64
     4   yr_built                     18256 non-null  int64  
     5   use_counts                   18256 non-null  object 
     6   Single Family(Res Use/Zone)  18256 non-null  int64  
     7   Townhouse Plat               18256 non-null  int64  
     8   Vacant(Single-family)        18256 non-null  int64  
     9   Duplex                       18256 non-null  int64  
     10  Apartment                    18256 non-null  int64  
     11  Single Family(C/I Zone)      18256 non-null  int64  
     12  Triplex                      18256 non-null  int64  
     13  Condominium(Residential)     18256 non-null  int64  
     14  4-Plex                       18256 non-null  int64  
     15  Mobile Home                  18256 non-null  int64  
     16  Retail Store                 18256 non-null  int64  
     17  Vacant(Multi-family)         18256 non-null  int64  
     18  Office Building              18256 non-null  int64  
     19  Apartment(Mixed Use)         18256 non-null  int64  
     20  Vacant(Commercial)           18256 non-null  int64  
     21  Church/Welfare/Relig Srvc    18256 non-null  int64  
     22  Parking(Assoc)               18256 non-null  int64  
     23  Park, Public(Zoo/Arbor)      18256 non-null  int64  
    dtypes: float64(1), int64(19), object(4)
    memory usage: 4.1+ MB
    


```python
out_data = data.join(cat_data.drop('yr_built', axis=1).select_dtypes(np.int64))
out_data.to_pickle('./test_train_set02.pkl')
```


```python
out_data['price'].hist()
plt.show()
```


![png](output_8_0.png)



```python

```
