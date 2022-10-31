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
data = pd.read_pickle('./test_train_set02.pkl')
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 18256 entries, 0 to 21595
    Data columns (total 42 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   id                           18256 non-null  int64  
     1   price                        18256 non-null  float64
     2   sqft_living                  18256 non-null  int64  
     3   sqft_lot                     18256 non-null  int64  
     4   view                         18256 non-null  object 
     5   grade                        18256 non-null  float64
     6   sqft_above                   18256 non-null  int64  
     7   yr_built                     18256 non-null  int64  
     8   lat                          18256 non-null  float64
     9   long                         18256 non-null  float64
     10  sqft_living15                18256 non-null  int64  
     11  sqft_lot15                   18256 non-null  int64  
     12  bins                         18256 non-null  object 
     13  POC_pct                      18256 non-null  float64
     14  median_income                18256 non-null  float64
     15  income_3rd                   18256 non-null  float64
     16  LifeExpectancy               18256 non-null  float64
     17  TREE_PCT                     18256 non-null  float64
     18  osdist_mean                  18256 non-null  float64
     19  os_per_person_pctle          18256 non-null  float64
     20  longitude                    18256 non-null  float64
     21  latitude                     18256 non-null  float64
     22  Shape_Area                   18256 non-null  float64
     23  PREUSE_DESC                  18256 non-null  object 
     24  Single Family(Res Use/Zone)  18256 non-null  int64  
     25  Townhouse Plat               18256 non-null  int64  
     26  Vacant(Single-family)        18256 non-null  int64  
     27  Duplex                       18256 non-null  int64  
     28  Apartment                    18256 non-null  int64  
     29  Single Family(C/I Zone)      18256 non-null  int64  
     30  Triplex                      18256 non-null  int64  
     31  Condominium(Residential)     18256 non-null  int64  
     32  4-Plex                       18256 non-null  int64  
     33  Mobile Home                  18256 non-null  int64  
     34  Retail Store                 18256 non-null  int64  
     35  Vacant(Multi-family)         18256 non-null  int64  
     36  Office Building              18256 non-null  int64  
     37  Apartment(Mixed Use)         18256 non-null  int64  
     38  Vacant(Commercial)           18256 non-null  int64  
     39  Church/Welfare/Relig Srvc    18256 non-null  int64  
     40  Parking(Assoc)               18256 non-null  int64  
     41  Park, Public(Zoo/Arbor)      18256 non-null  int64  
    dtypes: float64(14), int64(25), object(3)
    memory usage: 6.0+ MB
    


```python
col_names = list(data.columns)
col_names
```




    ['id',
     'price',
     'sqft_living',
     'sqft_lot',
     'view',
     'grade',
     'sqft_above',
     'yr_built',
     'lat',
     'long',
     'sqft_living15',
     'sqft_lot15',
     'bins',
     'POC_pct',
     'median_income',
     'income_3rd',
     'LifeExpectancy',
     'TREE_PCT',
     'osdist_mean',
     'os_per_person_pctle',
     'longitude',
     'latitude',
     'Shape_Area',
     'PREUSE_DESC',
     'Single Family(Res Use/Zone)',
     'Townhouse Plat',
     'Vacant(Single-family)',
     'Duplex',
     'Apartment',
     'Single Family(C/I Zone)',
     'Triplex',
     'Condominium(Residential)',
     '4-Plex',
     'Mobile Home',
     'Retail Store',
     'Vacant(Multi-family)',
     'Office Building',
     'Apartment(Mixed Use)',
     'Vacant(Commercial)',
     'Church/Welfare/Relig Srvc',
     'Parking(Assoc)',
     'Park, Public(Zoo/Arbor)']




```python
for index, col in enumerate(col_names):
    data[col].plot(title=col, figsize=(3,3), legend=False).hist(x=data[col], bins=16)

plt.show()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In [9], line 2
          1 for index, col in enumerate(col_names):
    ----> 2     data[col].plot(title=col, figsize=(3,3), legend=False).hist(x=data[col], bins=16)
          4 plt.show()
    

    File C:\tools\Anaconda3\envs\learn-env\lib\site-packages\pandas\plotting\_core.py:949, in PlotAccessor.__call__(self, *args, **kwargs)
        946             label_name = label_kw or data.columns
        947             data.columns = label_name
    --> 949 return plot_backend.plot(data, kind=kind, **kwargs)
    

    File C:\tools\Anaconda3\envs\learn-env\lib\site-packages\pandas\plotting\_matplotlib\__init__.py:61, in plot(data, kind, **kwargs)
         59         kwargs["ax"] = getattr(ax, "left_ax", ax)
         60 plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    ---> 61 plot_obj.generate()
         62 plot_obj.draw()
         63 return plot_obj.result
    

    File C:\tools\Anaconda3\envs\learn-env\lib\site-packages\pandas\plotting\_matplotlib\core.py:269, in MPLPlot.generate(self)
        267 def generate(self):
        268     self._args_adjust()
    --> 269     self._compute_plot_data()
        270     self._setup_subplots()
        271     self._make_plot()
    

    File C:\tools\Anaconda3\envs\learn-env\lib\site-packages\pandas\plotting\_matplotlib\core.py:418, in MPLPlot._compute_plot_data(self)
        416 # no non-numeric frames or series allowed
        417 if is_empty:
    --> 418     raise TypeError("no numeric data to plot")
        420 # GH25587: cast ExtensionArray of pandas (IntegerArray, etc.) to
        421 # np.ndarray before plot.
        422 numeric_data = numeric_data.copy()
    

    TypeError: no numeric data to plot



```python
def outlier_filter(df):
    col_names = list(df.select_dtypes([np.int64, np.float64]).columns)
    new_df = df
    for col in col_names:
        num = 8
        std = df[col].std()
        med = df[col].median()
        new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    return new_df
```


```python
# std = data['price'].std()
# med = data['price'].median()
# data[data['price'] < med+(2.5*std)].info()
data.select_dtypes([np.int64, np.float64]).describe()
```


```python
f_data = data.select_dtypes([np.int64, np.float64])
filtered_data = outlier_filter(f_data)
filtered_data.describe()
```

    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-11-c5ffd8170272>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    




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
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>...</th>
      <th>4-Plex</th>
      <th>Mobile Home</th>
      <th>Retail Store</th>
      <th>Vacant(Multi-family)</th>
      <th>Office Building</th>
      <th>Apartment(Mixed Use)</th>
      <th>Vacant(Commercial)</th>
      <th>Church/Welfare/Relig Srvc</th>
      <th>Parking(Assoc)</th>
      <th>Park, Public(Zoo/Arbor)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.746900e+04</td>
      <td>1.746900e+04</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.00000</td>
      <td>...</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
      <td>17469.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.631458e+09</td>
      <td>5.485812e+05</td>
      <td>2112.319938</td>
      <td>14283.407522</td>
      <td>7.693056</td>
      <td>1813.188448</td>
      <td>1971.391494</td>
      <td>47.558899</td>
      <td>-122.208144</td>
      <td>2020.89862</td>
      <td>...</td>
      <td>0.186273</td>
      <td>0.142996</td>
      <td>0.106646</td>
      <td>0.101780</td>
      <td>0.087984</td>
      <td>0.081516</td>
      <td>0.080371</td>
      <td>0.090503</td>
      <td>0.045395</td>
      <td>0.046024</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.900559e+09</td>
      <td>3.538351e+05</td>
      <td>910.825489</td>
      <td>28062.556563</td>
      <td>1.186380</td>
      <td>826.470767</td>
      <td>28.748668</td>
      <td>0.139715</td>
      <td>0.143330</td>
      <td>693.14488</td>
      <td>...</td>
      <td>0.735981</td>
      <td>0.635778</td>
      <td>0.512737</td>
      <td>0.503198</td>
      <td>0.407296</td>
      <td>0.415792</td>
      <td>0.421891</td>
      <td>0.336328</td>
      <td>0.265465</td>
      <td>0.234552</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.200019e+06</td>
      <td>8.000000e+04</td>
      <td>370.000000</td>
      <td>520.000000</td>
      <td>3.000000</td>
      <td>370.000000</td>
      <td>1900.000000</td>
      <td>47.159300</td>
      <td>-122.515000</td>
      <td>620.00000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.126059e+09</td>
      <td>3.250000e+05</td>
      <td>1450.000000</td>
      <td>5210.000000</td>
      <td>7.000000</td>
      <td>1210.000000</td>
      <td>1952.000000</td>
      <td>47.469400</td>
      <td>-122.323000</td>
      <td>1510.00000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.999200e+09</td>
      <td>4.589500e+05</td>
      <td>1950.000000</td>
      <td>7800.000000</td>
      <td>7.000000</td>
      <td>1590.000000</td>
      <td>1975.000000</td>
      <td>47.570000</td>
      <td>-122.219000</td>
      <td>1880.00000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.430500e+09</td>
      <td>6.599500e+05</td>
      <td>2600.000000</td>
      <td>11063.000000</td>
      <td>8.000000</td>
      <td>2250.000000</td>
      <td>1996.000000</td>
      <td>47.677700</td>
      <td>-122.119000</td>
      <td>2410.00000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.900000e+09</td>
      <td>3.420000e+06</td>
      <td>8670.000000</td>
      <td>327135.000000</td>
      <td>13.000000</td>
      <td>8020.000000</td>
      <td>2015.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.00000</td>
      <td>...</td>
      <td>7.000000</td>
      <td>9.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 39 columns</p>
</div>




```python
filtered_data.to_pickle('./test_train_set03.pkl')
```




```python
print(filtered_data['price'].head().to_markdown())tablefrmt="grid"
```

    |    |         price |
    |---:|--------------:|
    |  0 | 221900        |
    |  1 | 538000        |
    |  2 | 180000        |
    |  3 | 604000        |
    |  5 |      1.23e+06 |
    


```python
targ = filtered_data.select_dtypes([np.int64, np.float64])
columns = [col for col in targ]
divmod(len(columns), 3)


```


```python
def make_array(num, div, opt=None, verbose=False):
    def v_print(msg_string):
        if verbose is True:
            return print(msg_string)
        pass
    
    rows, rem = divmod(num, div)
    if rem == 0:
        for row in range(rows):
            row_out = ""
            for col in range(div):
                row_out += f" {col} "
            v_print(row_out)    
    else:
        v_print(f"{div} does not go into {num} and equal amount of times:")
        v_print(f"{num}/{div} returns {rows} with {rem} remaining.")
        div_up, div_down = (div+1, div-1)
        for step in range(div):
            v_print(f"trying {num}/{div_up}")
            if divmod(num, div_up)[1] == 0:
                v_print(f"success! recursing with {num} and {div_up}")
                make_array(num, div_up, verbose=verbose)
                break
        
            v_print(f"trying {num}/{div_down}")
            if divmod(num, div_down)[1] == 0:
                v_print(f"success! recursing with {num} and {div_down}")
                make_array(num, div_down, verbose=verbose)
                break
            div_up += 1
            div_down -= 1
make_array(3, 7)
```


```python
filtered_data.plot()
for col in columns:
    filtered_data[col]plot.hist(bins=16)
    plt.show()
```


```python

```
