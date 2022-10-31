```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
```


```python
foodest_df = pd.read_csv('./data/Food_Establishment_Inspection_Data.csv', dtype={'Zipcode':str, 'Grade':str}, parse_dates=['Inspection Date'])
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
      <th>Name</th>
      <th>Program Identifier</th>
      <th>Inspection Date</th>
      <th>Description</th>
      <th>Address</th>
      <th>City</th>
      <th>Zip Code</th>
      <th>Phone</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>...</th>
      <th>Inspection Score</th>
      <th>Inspection Result</th>
      <th>Inspection Closed Business</th>
      <th>Violation Type</th>
      <th>Violation Description</th>
      <th>Violation Points</th>
      <th>Business_ID</th>
      <th>Inspection_Serial_Num</th>
      <th>Violation_Record_ID</th>
      <th>Grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#807 TUTTA BELLA</td>
      <td>#807 TUTTA BELLA</td>
      <td>2022-08-31</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>2746 NE 45TH ST</td>
      <td>SEATTLE</td>
      <td>98105</td>
      <td>(206) 722-6400</td>
      <td>-122.296415</td>
      <td>47.662311</td>
      <td>...</td>
      <td>10.0</td>
      <td>Unsatisfactory</td>
      <td>False</td>
      <td>BLUE</td>
      <td>3200 - Insects, rodents, animals not present; ...</td>
      <td>5</td>
      <td>PR0089260</td>
      <td>DAEEWQC0L</td>
      <td>IVQ7QYW2V</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#807 TUTTA BELLA</td>
      <td>#807 TUTTA BELLA</td>
      <td>2022-08-31</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>2746 NE 45TH ST</td>
      <td>SEATTLE</td>
      <td>98105</td>
      <td>(206) 722-6400</td>
      <td>-122.296415</td>
      <td>47.662311</td>
      <td>...</td>
      <td>10.0</td>
      <td>Unsatisfactory</td>
      <td>False</td>
      <td>RED</td>
      <td>0200 - Food Worker Cards current for all food ...</td>
      <td>5</td>
      <td>PR0089260</td>
      <td>DAEEWQC0L</td>
      <td>IV0J437H6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#807 TUTTA BELLA</td>
      <td>#807 TUTTA BELLA</td>
      <td>2022-01-13</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>2746 NE 45TH ST</td>
      <td>SEATTLE</td>
      <td>98105</td>
      <td>(206) 722-6400</td>
      <td>-122.296415</td>
      <td>47.662311</td>
      <td>...</td>
      <td>0.0</td>
      <td>Satisfactory</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>PR0089260</td>
      <td>DAWWGK08K</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#807 TUTTA BELLA</td>
      <td>#807 TUTTA BELLA</td>
      <td>2021-01-06</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>2746 NE 45TH ST</td>
      <td>SEATTLE</td>
      <td>98105</td>
      <td>(206) 722-6400</td>
      <td>-122.296415</td>
      <td>47.662311</td>
      <td>...</td>
      <td>0.0</td>
      <td>Satisfactory</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>PR0089260</td>
      <td>DAUHM2FT8</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>+MAS CAFE</td>
      <td>+MAS CAFE</td>
      <td>2022-07-13</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>1906 N 34TH ST</td>
      <td>SEATTLE</td>
      <td>98103</td>
      <td>(206) 491-4694</td>
      <td>-122.334587</td>
      <td>47.648180</td>
      <td>...</td>
      <td>0.0</td>
      <td>Satisfactory</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>PR0046367</td>
      <td>DATSWIPUS</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
foodest_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 250176 entries, 0 to 250175
    Data columns (total 22 columns):
     #   Column                      Non-Null Count   Dtype  
    ---  ------                      --------------   -----  
     0   Name                        250176 non-null  object 
     1   Program Identifier          250176 non-null  object 
     2   Inspection Date             249389 non-null  object 
     3   Description                 250176 non-null  object 
     4   Address                     250176 non-null  object 
     5   City                        250176 non-null  object 
     6   Zip Code                    250176 non-null  object 
     7   Phone                       178918 non-null  object 
     8   Longitude                   249892 non-null  float64
     9   Latitude                    249892 non-null  float64
     10  Inspection Business Name    249389 non-null  object 
     11  Inspection Type             249389 non-null  object 
     12  Inspection Score            249347 non-null  float64
     13  Inspection Result           249389 non-null  object 
     14  Inspection Closed Business  249389 non-null  object 
     15  Violation Type              138708 non-null  object 
     16  Violation Description       138708 non-null  object 
     17  Violation Points            250176 non-null  int64  
     18  Business_ID                 250176 non-null  object 
     19  Inspection_Serial_Num       249389 non-null  object 
     20  Violation_Record_ID         138708 non-null  object 
     21  Grade                       190643 non-null  object 
    dtypes: float64(3), int64(1), object(18)
    memory usage: 42.0+ MB
    


```python
foodest_df.head()
```

    47.604292424700006 -122.29980006790001
    


```python
long_avg
lat_avg
print(lat_avg, long_avg)
```

    0.12887626722373807 0.1139288909623745
    


```python
small_df.head()
print(lat_std, long_std)
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
      <th>Name</th>
      <th>Inspection Date</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Description</th>
      <th>Grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#807 TUTTA BELLA</td>
      <td>2022-08-31</td>
      <td>-122.296415</td>
      <td>47.662311</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#807 TUTTA BELLA</td>
      <td>2022-08-31</td>
      <td>-122.296415</td>
      <td>47.662311</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#807 TUTTA BELLA</td>
      <td>2022-01-13</td>
      <td>-122.296415</td>
      <td>47.662311</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#807 TUTTA BELLA</td>
      <td>2021-01-06</td>
      <td>-122.296415</td>
      <td>47.662311</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>+MAS CAFE</td>
      <td>2022-07-13</td>
      <td>-122.334587</td>
      <td>47.648180</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
scatterplot(data=small_df, x='Longitude', y='Latitude', legend=False)
plt.show();
```


![png](output_6_0.png)



```python
sns.boxplot(data=small_df, y='Longitude')
plt.show();
```


```python
sns.boxplot(data=small_df, y='Latitude')
plt.show();
```


```python
long_avg = small_df['Longitude'].median()
lat_avg = small_df['Latitude'].median()
print(lat_avg, long_avg)
```

    47.604292424700006 -122.29980006790001
    


```python
long_std = np.std(small_df['Longitude'])
lat_std = np.std(small_df['Latitude'])
print(lat_std, long_std)
```


```python
zoom = 4
df_filtered = small_df[(abs(long_avg - small_df['Longitude']) < (abs(long_std) * zoom)) & (abs(lat_avg - small_df['Latitude']) < (abs(lat_std) * zoom)) ]
df_filtered.head()
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
      <th>Name</th>
      <th>Inspection Date</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Description</th>
      <th>Grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#807 TUTTA BELLA</td>
      <td>2022-08-31</td>
      <td>-122.296415</td>
      <td>47.662311</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#807 TUTTA BELLA</td>
      <td>2022-08-31</td>
      <td>-122.296415</td>
      <td>47.662311</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#807 TUTTA BELLA</td>
      <td>2022-01-13</td>
      <td>-122.296415</td>
      <td>47.662311</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#807 TUTTA BELLA</td>
      <td>2021-01-06</td>
      <td>-122.296415</td>
      <td>47.662311</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>+MAS CAFE</td>
      <td>2022-07-13</td>
      <td>-122.334587</td>
      <td>47.648180</td>
      <td>Seating 0-12 - Risk Category III</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
lat_max = df_filtered['Latitude'].min()
lat_min = df_filtered['Latitude'].max()
print(lat_max, lat_min)
```

    47.2231063708 47.926919
    


```python
sns.scatterplot(data=df_filtered, x='Longitude', y='Latitude', s=.8, legend=False)
plt.ylim(47.48, 47.65)
plt.xlim(-122.45, -122.18)
plt.show();
```


![png](output_13_0.png)



```python

```
