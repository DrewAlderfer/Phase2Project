```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from statsmodels.api import categorical
from statsmodels.api import stats as sms
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tabulate import tabulate
```

## Re-processing the Datasets
I'm going to jump back with the original full datasets now and see if I can't bring back in some
of the original columns to impove accuracy in my models.


```python
# Reading the clean datasets into the flow
house_data = pd.read_pickle('./00-final-dsc-phase-2-project-v2-3/data/house_pickle.pkl')
parcel_data = pd.read_pickle('./00-final-dsc-phase-2-project-v2-3/data/census_col.pkl')
```


```python
house_data.describe()
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
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>grade</th>
      <th>sqft_above</th>
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
      <th>count</th>
      <td>2.159700e+04</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>17755.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.580474e+09</td>
      <td>5.402966e+05</td>
      <td>3.373200</td>
      <td>2.115826</td>
      <td>2080.321850</td>
      <td>1.509941e+04</td>
      <td>1.494096</td>
      <td>7.657915</td>
      <td>1788.596842</td>
      <td>1970.999676</td>
      <td>83.636778</td>
      <td>98077.951845</td>
      <td>47.560093</td>
      <td>-122.213982</td>
      <td>1986.620318</td>
      <td>12758.283512</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.876736e+09</td>
      <td>3.673681e+05</td>
      <td>0.926299</td>
      <td>0.768984</td>
      <td>918.106125</td>
      <td>4.141264e+04</td>
      <td>0.539683</td>
      <td>1.173200</td>
      <td>827.759761</td>
      <td>29.375234</td>
      <td>399.946414</td>
      <td>53.513072</td>
      <td>0.138552</td>
      <td>0.140724</td>
      <td>685.230472</td>
      <td>27274.441950</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000102e+06</td>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>370.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.123049e+09</td>
      <td>3.220000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>1190.000000</td>
      <td>1951.000000</td>
      <td>0.000000</td>
      <td>98033.000000</td>
      <td>47.471100</td>
      <td>-122.328000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.904930e+09</td>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.618000e+03</td>
      <td>1.500000</td>
      <td>7.000000</td>
      <td>1560.000000</td>
      <td>1975.000000</td>
      <td>0.000000</td>
      <td>98065.000000</td>
      <td>47.571800</td>
      <td>-122.231000</td>
      <td>1840.000000</td>
      <td>7620.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.308900e+09</td>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.068500e+04</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>2210.000000</td>
      <td>1997.000000</td>
      <td>0.000000</td>
      <td>98118.000000</td>
      <td>47.678000</td>
      <td>-122.125000</td>
      <td>2360.000000</td>
      <td>10083.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.900000e+09</td>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>13.000000</td>
      <td>9410.000000</td>
      <td>2015.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# house_data.info()
```

### Splitting Numerical and Categorical Data
I am going to split the data as is and see if I need to change or process any of the columns.


```python
house_num = house_data.select_dtypes([np.int64, np.float64])
house_cat = house_data.select_dtypes([np.object])
```


```python
colors = list(mcolors.TABLEAU_COLORS.keys())
while len(colors) < num_of_columns:
    rand_index = np.random.randint(0, len(colors))
    colors.append(colors[rand_index])
```


```python
from graph_tool import make_array
```


```python
house_num_col = list(house_num.columns)
column_name = iter(house_num_col)
color = iter(colors)
num_of_columns = len(house_num_col)
graph_shape = make_array(num_of_columns, 4)
fig, ax = plt.subplots(len(graph_shape), len(graph_shape[0]), figsize=(12, 12))
ax[0, 0].hist(x=house_num['price'])
fig.set_edgecolor(color='black')
fig.set_tight_layout(tight=True)
for row, obj in enumerate(graph_shape):
    for col in obj:
        name = next(column_name)
        ax[row, col].hist(x=house_num[name], bins=16, color=next(color), alpha=.4)
        ax[row, col].set_title(name)
        # print(row, col, next(column_gen), next(color))
plt.show()
```


![png](output_9_0.png)



```python
data= house_num['grade'].value_counts()
x = list(data.index)
y = list(data)
sns.barplot(x=x, y=y)
plt.show()
```


![png](output_10_0.png)



```python
house_data[['grade', 'floors', 'yr_built', 'yr_renovated', 'zipcode']] = house_data[['grade', 'floors', 'yr_built', 'yr_renovated','zipcode']].astype('category')
house_data.info()
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
     7   floors         21597 non-null  category      
     8   waterfront     19221 non-null  object        
     9   view           21534 non-null  object        
     10  condition      21597 non-null  object        
     11  grade          21597 non-null  category      
     12  sqft_above     21597 non-null  int64         
     13  sqft_basement  21597 non-null  object        
     14  yr_built       21597 non-null  category      
     15  yr_renovated   17755 non-null  category      
     16  zipcode        21597 non-null  category      
     17  lat            21597 non-null  float64       
     18  long           21597 non-null  float64       
     19  sqft_living15  21597 non-null  int64         
     20  sqft_lot15     21597 non-null  int64         
    dtypes: category(5), datetime64[ns](1), float64(4), int64(7), object(4)
    memory usage: 2.8+ MB
    


```python
def outlier_filter(df, col_list):
    col_names = col_list
    new_df = df
    for col in col_names:
        num = 3
        std = df[col].std()
        med = df[col].mean()
        new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    return new_df
```


```python
# Grab the new numerical subset column names
target_columns = list(house_data.drop(['id', 'lat', 'long'], axis=1).select_dtypes([np.int64, np.float64]).columns)
# Filter the Dataframe and capture it in a new DataFrame
house_data_f = outlier_filter(house_data, target_columns)
print(target_columns, len(target_columns))
```

    ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_living15', 'sqft_lot15'] 8
    

    <ipython-input-100-6a00ff548aa8>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-100-6a00ff548aa8>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-100-6a00ff548aa8>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-100-6a00ff548aa8>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-100-6a00ff548aa8>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-100-6a00ff548aa8>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    <ipython-input-100-6a00ff548aa8>:8: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      new_df = new_df[(df[col] > med-(num*std)) & (df[col] < med+(num*std))]
    


```python
# reset the columns to exclude the catergorical data
column_name = iter(target_columns)
color = iter(colors)
num_of_columns = len(target_columns)

# Rebuild the color list
colors = list(mcolors.TABLEAU_COLORS.keys())
while len(colors) < num_of_columns:
    rand_index = np.random.randint(0, len(colors))
    colors.append(colors[rand_index])

# Regraph the distributions
graph_shape = make_array(num_of_columns, 4)
# print(tabulate(graph_shape, tablefmt="grid"))
rows, cols = np.shape(graph_shape)
fig, ax = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
fig.set_edgecolor(color='black')
fig.set_tight_layout(tight=True)
for row, obj in enumerate(graph_shape):
    for col in obj:
        name = next(column_name)
        ax[row, col].hist(x=house_data_f[name], bins=16, color=next(color), alpha=.4)
        ax[row, col].set_title(name)
plt.show()
```


![png](output_14_0.png)



```python
house_data_f.describe()
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
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>sqft_above</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.036800e+04</td>
      <td>2.036800e+04</td>
      <td>20368.000000</td>
      <td>20368.000000</td>
      <td>20368.000000</td>
      <td>20368.000000</td>
      <td>20368.000000</td>
      <td>20368.000000</td>
      <td>20368.000000</td>
      <td>20368.000000</td>
      <td>20368.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.650272e+09</td>
      <td>4.960311e+05</td>
      <td>3.328260</td>
      <td>2.052853</td>
      <td>1977.809652</td>
      <td>10119.530636</td>
      <td>1702.040259</td>
      <td>47.560389</td>
      <td>-122.219091</td>
      <td>1925.103496</td>
      <td>9305.417076</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.876436e+09</td>
      <td>2.481898e+05</td>
      <td>0.859081</td>
      <td>0.696666</td>
      <td>760.870054</td>
      <td>11839.085451</td>
      <td>705.626311</td>
      <td>0.138826</td>
      <td>0.137919</td>
      <td>609.111025</td>
      <td>9244.011093</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000102e+06</td>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>520.000000</td>
      <td>370.000000</td>
      <td>47.155900</td>
      <td>-122.512000</td>
      <td>460.000000</td>
      <td>651.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.195700e+09</td>
      <td>3.150000e+05</td>
      <td>3.000000</td>
      <td>1.500000</td>
      <td>1400.000000</td>
      <td>5000.000000</td>
      <td>1180.000000</td>
      <td>47.470800</td>
      <td>-122.331000</td>
      <td>1470.000000</td>
      <td>5000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.017355e+09</td>
      <td>4.400000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1860.000000</td>
      <td>7432.500000</td>
      <td>1520.000000</td>
      <td>47.571200</td>
      <td>-122.241000</td>
      <td>1800.000000</td>
      <td>7500.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.399301e+09</td>
      <td>6.150000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2442.750000</td>
      <td>10072.500000</td>
      <td>2100.000000</td>
      <td>47.679600</td>
      <td>-122.133000</td>
      <td>2290.000000</td>
      <td>9713.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.900000e+09</td>
      <td>1.640000e+06</td>
      <td>6.000000</td>
      <td>4.250000</td>
      <td>4790.000000</td>
      <td>137214.000000</td>
      <td>4270.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>4042.000000</td>
      <td>93825.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
house_data_f.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 20368 entries, 0 to 21596
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype         
    ---  ------         --------------  -----         
     0   id             20368 non-null  int64         
     1   date           20368 non-null  datetime64[ns]
     2   price          20368 non-null  float64       
     3   bedrooms       20368 non-null  int64         
     4   bathrooms      20368 non-null  float64       
     5   sqft_living    20368 non-null  int64         
     6   sqft_lot       20368 non-null  int64         
     7   floors         20368 non-null  category      
     8   waterfront     18124 non-null  object        
     9   view           20309 non-null  object        
     10  condition      20368 non-null  object        
     11  grade          20368 non-null  category      
     12  sqft_above     20368 non-null  int64         
     13  sqft_basement  20368 non-null  object        
     14  yr_built       20368 non-null  category      
     15  yr_renovated   16725 non-null  category      
     16  zipcode        20368 non-null  category      
     17  lat            20368 non-null  float64       
     18  long           20368 non-null  float64       
     19  sqft_living15  20368 non-null  int64         
     20  sqft_lot15     20368 non-null  int64         
    dtypes: category(5), datetime64[ns](1), float64(4), int64(7), object(4)
    memory usage: 2.8+ MB
    


```python
print(12758 + (3 * 27274))
print(house_data[['sqft_lot', 'sqft_lot15']].describe())
df = house_data[['sqft_lot', 'sqft_lot15']]
test = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
test.describe()
```

    94580
               sqft_lot     sqft_lot15
    count  2.159700e+04   21597.000000
    mean   1.509941e+04   12758.283512
    std    4.141264e+04   27274.441950
    min    5.200000e+02     651.000000
    25%    5.040000e+03    5100.000000
    50%    7.618000e+03    7620.000000
    75%    1.068500e+04   10083.000000
    max    1.651359e+06  871200.000000
    




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
      <th>sqft_lot</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>21110.000000</td>
      <td>21110.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>10434.632686</td>
      <td>9568.159308</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12169.375729</td>
      <td>9550.237231</td>
    </tr>
    <tr>
      <th>min</th>
      <td>520.000000</td>
      <td>651.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5000.000000</td>
      <td>5061.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7503.500000</td>
      <td>7546.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10300.000000</td>
      <td>9880.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>137214.000000</td>
      <td>94403.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fig, ax = plt.subplots(1, 2, figsize=(6, 3))
sqft_dist = test['sqft_lot'].value_counts()
sq_ft_itm = list(sqft_dist.index)
itm_count = list(sqft_dist) 
data_zip = list(zip(sq_ft_itm, itm_count))
data_zip.sort(key=lambda x : x[0])
x_s, y_s = [[x for (x, y) in data_zip],
            [y for (x, y) in data_zip]]
x_domain = max(x_s)-min(x_s)
bin_space = np.linspace(start=min(x_s), stop=max(x_s), num=60)
bin_results = list(range(len(bin_space)))
for point, count in data_zip:
    for index, bin in enumerate(bin_space):
        if point < bin:
            continue
        bin_results[index] += count
print(tabulate(list(zip(bin_space, bin_results)), tablefmt="grid"))
        
# sqft15_dist = test['sqft_lot15'].value_counts()
# x = list(sqft15_dist.index)
# y = list(sqft15_dist)
# # ax[0].hist(x=test['sqft_lot'].value_counts(), bins=50, color='red', alpha=.3)
# # ax[1].hist(x=test['sqft_lot15'].value_counts(), bins=50, color='blue', alpha=.3)
# # fig.set_tight_layout(tight=True)
# # sns.histplot(sqft_dist, x=x, y=y, alpha=.3)
# sns.scatterplot(data=data_zip,x=x_s, y=y_s, s=1, color='red', alpha=.3)
# plt.xlim(0, 20000)
# plt.show()
```

    +-----------+-------+
    |    520    | 21110 |
    +-----------+-------+
    |   2836.85 | 19395 |
    +-----------+-------+
    |   5153.69 | 15420 |
    +-----------+-------+
    |   7470.54 | 10747 |
    +-----------+-------+
    |   9787.39 |  6012 |
    +-----------+-------+
    |  12104.2  |  3875 |
    +-----------+-------+
    |  14421.1  |  2906 |
    +-----------+-------+
    |  16737.9  |  2314 |
    +-----------+-------+
    |  19054.8  |  1952 |
    +-----------+-------+
    |  21371.6  |  1684 |
    +-----------+-------+
    |  23688.5  |  1500 |
    +-----------+-------+
    |  26005.3  |  1384 |
    +-----------+-------+
    |  28322.2  |  1265 |
    +-----------+-------+
    |  30639    |  1196 |
    +-----------+-------+
    |  32955.9  |  1116 |
    +-----------+-------+
    |  35272.7  |   973 |
    +-----------+-------+
    |  37589.6  |   841 |
    +-----------+-------+
    |  39906.4  |   755 |
    +-----------+-------+
    |  42223.3  |   665 |
    +-----------+-------+
    |  44540.1  |   575 |
    +-----------+-------+
    |  46856.9  |   520 |
    +-----------+-------+
    |  49173.8  |   465 |
    +-----------+-------+
    |  51490.6  |   417 |
    +-----------+-------+
    |  53807.5  |   375 |
    +-----------+-------+
    |  56124.3  |   339 |
    +-----------+-------+
    |  58441.2  |   309 |
    +-----------+-------+
    |  60758    |   288 |
    +-----------+-------+
    |  63074.9  |   271 |
    +-----------+-------+
    |  65391.7  |   249 |
    +-----------+-------+
    |  67708.6  |   234 |
    +-----------+-------+
    |  70025.4  |   221 |
    +-----------+-------+
    |  72342.3  |   210 |
    +-----------+-------+
    |  74659.1  |   200 |
    +-----------+-------+
    |  76976    |   191 |
    +-----------+-------+
    |  79292.8  |   174 |
    +-----------+-------+
    |  81609.7  |   165 |
    +-----------+-------+
    |  83926.5  |   154 |
    +-----------+-------+
    |  86243.4  |   147 |
    +-----------+-------+
    |  88560.2  |   140 |
    +-----------+-------+
    |  90877.1  |   133 |
    +-----------+-------+
    |  93193.9  |   124 |
    +-----------+-------+
    |  95510.7  |   115 |
    +-----------+-------+
    |  97827.6  |   110 |
    +-----------+-------+
    | 100144    |   105 |
    +-----------+-------+
    | 102461    |    97 |
    +-----------+-------+
    | 104778    |    93 |
    +-----------+-------+
    | 107095    |    91 |
    +-----------+-------+
    | 109412    |    83 |
    +-----------+-------+
    | 111729    |    82 |
    +-----------+-------+
    | 114046    |    79 |
    +-----------+-------+
    | 116362    |    77 |
    +-----------+-------+
    | 118679    |    71 |
    +-----------+-------+
    | 120996    |    70 |
    +-----------+-------+
    | 123313    |    69 |
    +-----------+-------+
    | 125630    |    68 |
    +-----------+-------+
    | 127947    |    68 |
    +-----------+-------+
    | 130263    |    65 |
    +-----------+-------+
    | 132580    |    64 |
    +-----------+-------+
    | 134897    |    62 |
    +-----------+-------+
    | 137214    |    60 |
    +-----------+-------+
    
