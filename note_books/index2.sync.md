```python
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
house_df = pd.read_csv('./data/kc_house_data.csv')
house_df.head()
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
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NONE</td>
      <td>...</td>
      <td>7 Average</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>NO</td>
      <td>NONE</td>
      <td>...</td>
      <td>7 Average</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>NO</td>
      <td>NONE</td>
      <td>...</td>
      <td>6 Low Average</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>NO</td>
      <td>NONE</td>
      <td>...</td>
      <td>7 Average</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>NO</td>
      <td>NONE</td>
      <td>...</td>
      <td>8 Good</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
house_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             21597 non-null  int64  
     1   date           21597 non-null  object 
     2   price          21597 non-null  float64
     3   bedrooms       21597 non-null  int64  
     4   bathrooms      21597 non-null  float64
     5   sqft_living    21597 non-null  int64  
     6   sqft_lot       21597 non-null  int64  
     7   floors         21597 non-null  float64
     8   waterfront     19221 non-null  object 
     9   view           21534 non-null  object 
     10  condition      21597 non-null  object 
     11  grade          21597 non-null  object 
     12  sqft_above     21597 non-null  int64  
     13  sqft_basement  21597 non-null  object 
     14  yr_built       21597 non-null  int64  
     15  yr_renovated   17755 non-null  float64
     16  zipcode        21597 non-null  int64  
     17  lat            21597 non-null  float64
     18  long           21597 non-null  float64
     19  sqft_living15  21597 non-null  int64  
     20  sqft_lot15     21597 non-null  int64  
    dtypes: float64(6), int64(9), object(6)
    memory usage: 3.5+ MB
    


```python
house_df['date'].value_counts()
```




    6/23/2014     142
    6/26/2014     131
    6/25/2014     131
    7/8/2014      127
    4/27/2015     126
                 ... 
    1/31/2015       1
    1/17/2015       1
    11/30/2014      1
    8/30/2014       1
    11/2/2014       1
    Name: date, Length: 372, dtype: int64




```python
house_df['date'] = pd.to_datetime(house_df['date'])
house_df['grade'] = house_df['grade'].map(lambda x : x.split()[0])
house_df['grade'] = house_df['grade'].astype(np.float64)
```


```python
house_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype         
    ---  ------         --------------  -----         
     0   id             21597 non-null  int64         
     1   date           21597 non-null  datetime64[ns]
     2   price          21597 non-null  float64       
     3   bedrooms       21597 non-null  int64         
     4   bathrooms      21597 non-null  float64       
     5   sqft_living    21597 non-null  int64         
     6   sqft_lot       21597 non-null  int64         
     7   floors         21597 non-null  float64       
     8   waterfront     19221 non-null  object        
     9   view           21534 non-null  object        
     10  condition      21597 non-null  object        
     11  grade          21597 non-null  float64       
     12  sqft_above     21597 non-null  int64         
     13  sqft_basement  21597 non-null  object        
     14  yr_built       21597 non-null  int64         
     15  yr_renovated   17755 non-null  float64       
     16  zipcode        21597 non-null  int64         
     17  lat            21597 non-null  float64       
     18  long           21597 non-null  float64       
     19  sqft_living15  21597 non-null  int64         
     20  sqft_lot15     21597 non-null  int64         
    dtypes: datetime64[ns](1), float64(7), int64(9), object(4)
    memory usage: 3.5+ MB
    

Looking at the data it is clear that many of the columns are not going to e useful to me.

I think I should drop the following:
  - I am going to drop `date` because the set only contains a single year. Not enough time to use
    date for anything interesting imo.
  - `waterfront` I think waterfront is probably heavily coorelated to price, but I'm trying to avoid 
     predicting for price. There are also only a hand full of records for it. So, it's probably not
     worth using.
  - `floors` floors seem like their going to be colinear with other values like sqft.
  - `zipcode` is useful for making a nice graph of the lat and long but I think it's
    probably a category that is very colinear with many other categories.
  - `grade` and `condition` may also be irrelevant to what I am looking to compare or
    predict. Maybe they would be interesting values to try and predict.
  - `bedrooms` and `bathrooms` also both seem like they will be colinear to sqft, so I am going to
     drop them.


```python
house_df.to_pickle('./data/house_pickle.pkl')
```


```python
house_df['waterfront'].value_counts()
```




    NO     19075
    YES      146
    Name: waterfront, dtype: int64




```python
house_df['date']
```




    0       2014-10-13
    1       2014-12-09
    2       2015-02-25
    3       2014-12-09
    4       2015-02-18
               ...    
    21592   2014-05-21
    21593   2015-02-23
    21594   2014-06-23
    21595   2015-01-16
    21596   2014-10-15
    Name: date, Length: 21597, dtype: datetime64[ns]




```python
house_df_dropped_num = house_df_dropped.select_dtypes(include=[np.int64, np.float64])
house_df_dropped_num['date'] = house_df['date']
house_df_dropped_cat = house_df_dropped.select_dtypes(include=[np.object])

house_df_dropped_num['living_above_diff'] = house_df_dropped_num.apply(lambda x : x['sqft_living'] - x['sqft_above'], axis=1)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In [11], line 1
    ----> 1 house_df_dropped_num = house_df_dropped.select_dtypes(include=[np.int64, np.float64])
          2 house_df_dropped_num['date'] = house_df['date']
          3 house_df_dropped_cat = house_df_dropped.select_dtypes(include=[np.object])
    

    NameError: name 'house_df_dropped' is not defined



```python
house_df_dropped_num.info()
```


```python
house_df_dropped_num.head()
```


```python
house_df_dropped_cat.info()
```


```python
sns.pairplot(house_df_dropped_num[['living_above_diff', 'yr_built', 'price']])
plt.show();
```


```python
house_df_dropped_num.corr()
```


```python
max_lat = house_df_dropped_num['lat'].max()
min_lat = house_df_dropped_num['lat'].min()
span_lat = abs(max_lat - min_lat)
max_long = house_df_dropped_num['long'].max()
min_long = house_df_dropped_num['long'].min()
span_long = abs(max_long - min_long)
print(f'max_lat: {max_lat}')
print(f'min_lat: {min_lat}')
print(f'span_lat: {span_lat}')
print(f'max_long: {max_long}')
print(f'min_long: {min_long}')
print(f'span_long: {span_long}')
```


```python
map_height = 250 
map_width = round((span_long/span_lat) * map_height)

x_bin = lambda x : round(((x-min_long)/span_long)*map_width)
y_bin = lambda y : round(((y-min_lat)/span_lat)*map_height)
house_df_dropped_num['long_bin'] = house_df_dropped_num['long'].map(x_bin)
house_df_dropped_num['lat_bin'] = house_df_dropped_num['lat'].map(y_bin)
house_df_dropped_num.head()
```


```python
sns.histplot(house_df_dropped_num['long_bin'])
plt.show();
```


```python
sns.histplot(house_df_dropped_num['lat_bin'])
plt.show();
```


```python
lat_bin = house_df_dropped_num['lat_bin']
long_bin = house_df_dropped_num['long_bin']
price = house_df_dropped_num['price']
sns.scatterplot(x=long_bin, y=lat_bin, hue=price, s=1)
plt.ylim(150/850 * map_height, 885/850 * map_height)
plt.xlim(0, 1250/850 * map_height)
plt.show();
```


```python
house_df_dropped_num['bin_cat'] = house_df_dropped_num[['lat_bin', 'long_bin']].apply(lambda x : str(x['lat_bin'])+str(x['long_bin']), axis=1) 
```


```python
house_df_dropped_num.info()
```


```python
binned_coords = house_df_dropped_num.groupby('bin_cat').mean()
binned_coords.info()
```


```python
sns.scatterplot(data=binned_coords, x='long_bin', y='lat_bin', hue='sqft_living', s=10)
plt.ylim(150/850 * map_height, 885/850 * map_height)
plt.xlim(0, 1250/850 * map_height)
plt.show();
```


```python
sns.histplot(binned_coords['price'])
plt.show();
```


```python
def plot_time(yr_range):
    for yr in range(yr_range):
        yr = yr + 10
        sns.scatterplot(data=house_df_dropped_num[(house_df_dropped_num['date'] > pd.to_datetime(f'01/01/20{yr:02}')) & (house_df_dropped_num['date'] < pd.to_datetime(f'01/01/20{yr+1:02}'))], x='long_bin', y='lat_bin', hue='price', s=10)
        plt.ylim(150/850 * map_height, 885/850 * map_height)
        plt.xlim(0, 1250/850 * map_height)
        plt.show();

                
plot_time(8)
```


```python
house_df_dropped_num['date'].min()
```
