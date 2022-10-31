```python
import pandas as pd
import sqlite3
```


```python
df = pd.read_pickle('./data/census_lci.pkl')
```


```python
with sqlite3.connect('./data/king_lci_census.db') as db_con:
    df.to_sql(name='king_county_lci', con=db_con)
```
