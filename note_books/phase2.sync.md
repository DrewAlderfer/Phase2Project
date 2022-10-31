```python
run imports_book.py 
```


```python
run -i pickle_prep.py
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
local_col = comb_parcel_house_dna['PREUSE_DESC']
```


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
```


```python
comb_parcel_house_dna['use_counts'] = local_col.map(counter)
```

    C:\tools\Anaconda3\envs\learn-env\lib\site-packages\pandas\core\indexing.py:1745: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      isetter(ilocs[0], value)
    


```python
print(comb_parcel_house_dna['use_counts'][134])
```

    {'Single Family(Res Use/Zone)': 38, 'Townhouse Plat': 68, 'Vacant(Single-family)': 3}
    


```python
def tokenizer(v_dict):
    result = ""
    for tup in v_dict.items():
        if isinstance(tup[0], str):
            words = tup[0].split()
            for word in words:
                if word[0].isdigit():
                    letters = word.split()
                    length = 0
                    for letter in word:
                        if letter.isalpha() and length < 3:
                            result += letter
                            length += 1
                elif word[0].isalpha() and len(words) == 1:
                    result += word[0:3] 
                elif word[0].isalpha():
                    result += word[0].upper()
                else:
                    result += f"?{word.upper()}?"
            result += f"{tup[1]}"
        else:
            continue
    return result
```


```python
print(tokenizer(comb_parcel_house_dna['use_counts'][134]))
```

    TP171Tri5SFU38SFZ3POS1Dup7Apa11CS1Con6Ple3
    


```python
master_count = {}
```


```python
def key_counter(count_dict):
    for key, val in count_dict.items():
        try:
            master_count['key'] = master_count['key'] + val
        except KeyError:
            master_count.update({key: val})
```


```python
comb_parcel_house_dna['use_counts'].map(key_counter)
master_list = list(master_count.items())
master_list.sort(reverse=True, key=lambda x : x[1])
for item, count in master_list:
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
    Warehouse: 885
    Service Building: 855
    Single Family(C/I Use): 846
    Restaurant/Lounge: 763
    Utility, Public: 696
    School(Public): 694
    Easement: 670
    Medical/Dental Office: 629
    Apartment(Subsidized): 467
    Right of Way/Utility, Road: 443
    Governmental Service: 422
    Group Home: 399
    Retail(Line/Strip): 375
    Condominium(Mixed Use): 367
    School(Private): 337
    Conv Store with Gas: 312
    Parking(Commercial Lot): 312
    Industrial(Lignt): 304
    Retirement Facility: 285
    Restaurant(Fast Food): 275
    Vacant(Industrial): 264
    Daycare Center: 228
    Bank: 219
    Rooming House: 188
    Industrial(Gen Purpose): 187
    Tavern/Lounge: 159
    Club: 159
    Grocery Store: 148
    Golf Course: 125
    Houseboat: 121
    Vet/Animal Control Srvc: 117
    Shell Structure: 105
    Mini Warehouse: 103
    Sport Facility: 98
    Service Station: 90
    Art Gallery/Museum/Soc Srvc: 80
    Apartment(Co-op): 79
    Auto Showroom and Lot: 79
    Mobile Home Park: 78
    Parking(Garage): 74
    Conv Store without Gas: 71
    Shopping Ctr(Nghbrhood): 69
    Nursing Home: 69
    Park, Private(Amuse Ctr): 68
    Mortuary/Cemetery/Crematory: 67
    Hotel/Motel: 65
    Residence Hall/Dorm: 62
    Health Club: 57
    Industrial(Heavy): 51
    Terminal(Rail): 48
    Open Space Tmbr Land/Greenbelt: 46
    Post Office/Post Service: 44
    Marina: 44
    Car Wash: 42
    Utility, Private(Radio/T.V.): 41
    Retail(Discount): 40
    Open Space(Curr Use-RCW 84.34): 40
    Condominium(Office): 33
    Auditorium//Assembly Bldg: 30
    Retail(Big Box): 28
    Tideland, 1st Class: 27
    Movie Theater: 26
    Resort/Lodge/Retreat: 23
    Mini Lube: 22
    Historic Prop(Misc): 18
    Industrial Park: 18
    Hospital: 18
    River/Creek/Stream: 17
    Fraternity/Sorority House: 17
    Shopping Ctr(Community): 17
    Greenhse/Nrsry/Hort Srvc: 16
    Rehabilitation Center: 15
    Open Space(Agric-RCW 84.34): 14
    Historic Prop(Residence): 13
    Mining/Quarry/Ore Processing: 12
    Reserve/Wilderness Area: 12
    Bowling Alley: 11
    Gas Station: 11
    Farm: 9
    Historic Prop(Office): 8
    Air Terminal and Hangers: 7
    High Tech/High Flex: 7
    Historic Prop(Rec/Entertain): 7
    Shopping Ctr(Maj Retail): 6
    Tideland, 2nd Class: 5
    Condominium(M Home Pk): 3
    Office Park: 3
    Terminal(Marine): 2
    Historic Prop(Retail): 2
    Shopping Ctr(Specialty): 1
    Terminal(Auto/Bus/Other): 1
    Bed & Breakfast: 1
    
