```python
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
```

### TO DO:
- Check the effect on model prediction if you downsample the coordinate bins more
  - It would be nice to do this programtically
- Go back and run some models on the orginal dataset to compare models
- Look into a more rational binning algorithm and whether or not you can test the merge results
  - Maybe look into matching the secondary dataset bins to the ones captured in the first set, or
    at least compare with some hueristic on how your combining them in the merge.
- Try an get a more regular grid for the bins so that displaying them in easier. 
  - I still really would like a binning algorithm or maybe a post processor that creates a hex 
    grid for displaying the data.


```python
run imports_book.py
```


```python
run -i pickle_prep.py
```


![png](output_3_0.png)



```python
run -i model_01.py
```

    id                      : -4879.948635
    sqft_living             : 151686.937695
    sqft_lot                : 7293.590851
    view                    : 109802.224208
    grade                   : 20175.412241
    sqft_above              : -58182.788212
    yr_built                : -171616.202317
    lat                     : -501013.786540
    long                    : 13111.815278
    sqft_living15           : -1503.820840
    sqft_lot15              : -15972.695640
    bins                    : 55992.024801
    POC_pct                 : -13022.025996
    median_income           : 54244.241114
    income_3rd              : -61196.850391
    LifeExpectancy          : -10541.386761
    TREE_PCT                : 13574.651323
    osdist_mean             : 487812.494173
    os_per_person_pctle     : 213103.456189
    longitude               : 11673.383222
    latitude                : -7670.913437
    Shape_Area              : -165.509178
    PREUSE_DESC             : -340.797301
    Single Family(Res Use/Zone): 1136.603725
    Townhouse Plat          : -1049.672925
    Vacant(Single-family)   : -747.302664
    Duplex                  : -1738.233156
    Apartment               : 10935.780594
    Single Family(C/I Zone) : -600.013849
    Triplex                 : 3617.324426
    Condominium(Residential): -1129.166875
    4-Plex                  : 2906.372115
    Mobile Home             : 2687.710991
    Retail Store            : -446.806135
    Vacant(Multi-family)    : -825.402082
    Office Building         : -2941.636287
    Apartment(Mixed Use)    : 459.031637
    Vacant(Commercial)      : 2994.596586
    ----------------------------------------
    Model intercept         : 554312.200065
    Model r_sq score        : 0.704917
    Mean Absolute Error     : 129179.355142
    Root Mean Sq Error      : 217176.778224
    


```python
run -i model02.py
```

    id                      :  -0.011527  |
    price                   :   0.754479  |
    sqft_living             :  -0.133988  |
    sqft_lot                :  -0.062994  |
    view                    :   0.197484  |
    grade                   :  -0.349309  |
    sqft_above              :  -0.194811  |
    yr_built                :   6.578058  |
    lat                     :   3.017798  |
    long                    :   0.105476  |
    sqft_living15           :  -0.070847  |
    sqft_lot15              :  -0.338865  |
    bins                    :   0.770171  |
    POC_pct                 :   0.510359  |
    median_income           :  -0.147622  |
    income_3rd              :  -0.120870  |
    TREE_PCT                :   0.095158  |
    osdist_mean             :  -3.288904  |
    os_per_person_pctle     :  -5.995679  |
    longitude               :  -0.012387  |
    latitude                :   0.203180  |
    Shape_Area              :  -0.012500  |
    PREUSE_DESC             :   0.008999  |
    Single Family(Res Use/Zone):   0.095918  |
    Townhouse Plat          :   0.031978  |
    Vacant(Single-family)   :  -0.034414  |
    Duplex                  :   0.122646  |
    Apartment               :   0.055695  |
    Single Family(C/I Zone) :  -0.030057  |
    Triplex                 :  -0.148633  |
    Condominium(Residential):   0.028704  |
    4-Plex                  :   0.058441  |
    Mobile Home             :   0.004119  |
    Retail Store            :   0.220048  |
    Vacant(Multi-family)    :   0.016817  |
    Office Building         :   0.009173  |
    Apartment(Mixed Use)    :   0.025365  |
    Vacant(Commercial)      :  -0.024724  |
    ----------------------------------------
    Model intercept         : 82.190092
    Model r_sq score        : 0.512778
    Mean Absolute Error     : 1.731843
    Root Mean Sq Error      : 2.272448
    


```python
run -i model03.py
```

    id                      :  -0.004668  |
    price                   :   0.739497  |
    sqft_living             :  -0.361522  |
    sqft_lot                :  -0.057240  |
    grade                   :   0.142520  |
    yr_built                :  -0.218601  |
    lat                     :   0.627254  |
    sqft_living15           :   0.085145  |
    sqft_lot15              :  -0.074418  |
    bins                    :  -0.420045  |
    POC_pct                 :   1.153714  |
    median_income           :  -0.132929  |
    TREE_PCT                :  -0.132692  |
    osdist_mean             :   0.100125  |
    os_per_person_pctle     :  -0.276935  |
    longitude               :  -0.019307  |
    Shape_Area              :   0.229707  |
    PREUSE_DESC             :   0.001986  |
    Single Family(Res Use/Zone):   0.016150  |
    Townhouse Plat          :   0.104244  |
    Vacant(Single-family)   :   0.020079  |
    Duplex                  :  -0.036180  |
    Apartment               :   0.123427  |
    Single Family(C/I Zone) :   0.062854  |
    Triplex                 :  -0.036127  |
    Condominium(Residential):  -0.157482  |
    4-Plex                  :   0.024960  |
    Mobile Home             :   0.052849  |
    Retail Store            :   0.020947  |
    Vacant(Multi-family)    :   0.224054  |
    Office Building         :   0.016554  |
    Apartment(Mixed Use)    :   0.003203  |
    Vacant(Commercial)      :   0.017612  |
    Church/Welfare/Relig Srvc:  -0.029823  |
    ----------------------------------------
    Model intercept         : 82.190092
    Model r_sq score        : 0.503081
    Mean Absolute Error     : 1.751840
    Root Mean Sq Error      : 2.296161
    


```python
sns.kdeplot(train_pred_le2, alpha=.3)
sns.kdeplot(y_pred_le2, fill=True, color='red', alpha=.3)
plt.show()
```


![png](output_7_0.png)



```python
resid_le2 = y_test_le2 - y_pred_le2
# print(len(y_test_le2), len(y_pred_le2), len(resid_le2))
fig, ax = plt.subplots()
ax.scatter(x=range(y_pred_le2.shape[0]),y=resid_le2, alpha=0.1);
plt.show()
```


![png](output_8_0.png)



```python
run -i model4.py
```

    sqft_living             : 219099.462945  |
    sqft_lot                : 9290.234770  |
    yr_built                : -27848.709663  |
    sqft_living15           : 44866.791050  |
    sqft_lot15              : -2724.809073  |
    bins                    : -16101.298067  |
    POC_pct                 : 55905.246611  |
    median_income           : 59952.290974  |
    LifeExpectancy          : -60328.386610  |
    TREE_PCT                : -11441.176855  |
    osdist_mean             : 17103.463778  |
    os_per_person_pctle     : -26768.619268  |
    longitude               : 42813.682268  |
    latitude                : 13428.901004  |
    Shape_Area              : -7538.631673  |
    PREUSE_DESC             : 3663.985004  |
    Single Family(Res Use/Zone): -1808.745239  |
    Townhouse Plat          : 3530.299180  |
    Vacant(Single-family)   : 2327.015372  |
    Duplex                  : -1926.164536  |
    Apartment               : -2101.830571  |
    Single Family(C/I Zone) : 14062.112455  |
    Triplex                 : -2142.561942  |
    Condominium(Residential): 3002.451949  |
    4-Plex                  : -801.557623  |
    Mobile Home             : 2585.485665  |
    Retail Store            : 1444.180222  |
    Vacant(Multi-family)    : 780.718206  |
    Office Building         : 287.184493  |
    Apartment(Mixed Use)    : -2973.087712  |
    Vacant(Commercial)      : 2083.298803  |
    Church/Welfare/Relig Srvc: 3981.009697  |
    ----------------------------------------
    Model trian score       : 0.677517
    Model test score        : 0.675415
    Model intercept         : 554312.200065
    Model r_sq score        : 0.677517
    Mean Absolute Error     : 136620.885085
    Mean Sq Error Train     : 213334.256463
    Mean Sq Error Test      : 223146.486575
    Mean Sq Error Diff      : 9812.230112
    


```python
sns.kdeplot(train_pred_le3, alpha=.3)
sns.kdeplot(y_pred_le3, fill=True, color='red', alpha=.3)
plt.show()
```


![png](output_10_0.png)



```python
resid_le3 = y_test_le3 - y_pred_le2
# print(len(y_test_le3), len(y_pred_le3), len(resid_le3))
fig, ax = plt.subplots()
ax.scatter(x=range(y_pred_le3.shape[0]),y=resid_le3, color='lightgreen',  alpha=0.1);
plt.show()
```


![png](output_11_0.png)



```python
run -i model06.py
```

    price                   :  -3.745926  |
    sqft_living             :   1.642800  |
    sqft_lot                :   0.681374  |
    yr_built                :   1.196240  |
    sqft_living15           :   1.013943  |
    sqft_lot15              :   1.786108  |
    bins                    :  -2.032365  |
    POC_pct                 :   0.866011  |
    median_income           :  -0.596829  |
    LifeExpectancy          :   0.595129  |
    osdist_mean             :   4.099845  |
    os_per_person_pctle     :   4.925862  |
    longitude               :   2.432153  |
    latitude                :  -0.055143  |
    Shape_Area              :  -2.580435  |
    PREUSE_DESC             :  -0.412106  |
    Single Family(Res Use/Zone):   0.327542  |
    Townhouse Plat          :  -0.285114  |
    Vacant(Single-family)   :  -0.131323  |
    Duplex                  :  -0.683825  |
    Apartment               :   0.355417  |
    Single Family(C/I Zone) :  -0.599532  |
    Triplex                 :  -0.696121  |
    Condominium(Residential):   0.099225  |
    4-Plex                  :  -0.250894  |
    Mobile Home             :   0.095317  |
    Retail Store            :  -0.390760  |
    Vacant(Multi-family)    :   0.180432  |
    Office Building         :  -0.528248  |
    Apartment(Mixed Use)    :  -0.158147  |
    Vacant(Commercial)      :  -0.494629  |
    Church/Welfare/Relig Srvc:  -0.206831  |
    ----------------------------------------
    Model trian score       : 0.550543
    Model test score        : 0.539444
    Model intercept         : 33.420800
    Model r_sq score        : 0.550543
    Mean Absolute Error     : 554075.437474
    Mean Sq Error Train     : 11.220377
    Mean Sq Error Test      : 10.882046
    Mean Sq Error Diff      : 0.338331
    


```python
resid_t01 = y_test_t01 - y_test_pred_t01
# print(len(y_test_t01), len(y_pred_t01), len(resid_t01))
fig, ax = plt.subplots()
ax.scatter(x=range(y_test_pred_t01.shape[0]),y=resid_t01, color='coral',  alpha=0.1);
plt.show()
```


![png](output_13_0.png)



```python
run -i model_life_ex_01.py
```

    price                   :   0.822755  |
    sqft_living             :  -0.350384  |
    sqft_lot                :  -0.136434  |
    bins                    :  -0.466828  |
    POC_pct                 :   1.208713  |
    median_income           :  -0.227466  |
    TREE_PCT                :  -0.169396  |
    osdist_mean             :   0.061875  |
    os_per_person_pctle     :  -0.395556  |
    longitude               :   0.672996  |
    latitude                :   0.006453  |
    Vacant(Single-family)   :   0.026493  |
    Apartment               :   0.053237  |
    Condominium(Residential):  -0.005184  |
    4-Plex                  :   0.071272  |
    Vacant(Multi-family)    :  -0.001428  |
    Office Building         :   0.233913  |
    ----------------------------------------
    Model trian score       : 0.488695
    Model test score        : 0.469140
    Model intercept         : 82.190092
    Model r_sq score        : 0.488695
    Mean Absolute Error     : 554026.663594
    Mean Sq Error Train     : 2.262861
    Mean Sq Error Test      : 2.312613
    Mean Sq Error Diff      : 0.049751
    


```python
run -i model_life_ex02.py
```

    price                   :   0.833719  |
    sqft_living             :  -0.360626  |
    sqft_lot                :  -0.135397  |
    bins                    :  -0.463764  |
    POC_pct                 :   1.195224  |
    median_income           :  -0.232253  |
    TREE_PCT                :  -0.175662  |
    osdist_mean             :   0.054945  |
    os_per_person_pctle     :  -0.399031  |
    longitude               :   0.683447  |
    latitude                :   0.006327  |
    Vacant(Single-family)   :   0.018216  |
    Office Building         :   0.254087  |
    ----------------------------------------
    Model trian score       : 0.487691
    Model test score        : 0.466680
    Model intercept         : 82.190092
    Model r_sq score        : 0.487691
    Mean Absolute Error     : 554026.663594
    Mean Sq Error Train     : 2.265083
    Mean Sq Error Test      : 2.317966
    Mean Sq Error Diff      : 0.052883
    

## Third Version of the model
Even with all the variables in it's still only getting toward a %50 explination value for the 
variance in my training data.


```python
run -i model_life_ex03.py
```

    price                   :   1.139433  |
    sqft_living             :  -0.284965  |
    sqft_lot                :  -0.020515  |
    grade                   :  -0.006116  |
    sqft_above              :  -0.322965  |
    sqft_living15           :   0.085369  |
    sqft_lot15              :  -0.138030  |
    POC_pct                 :  -0.460402  |
    median_income           :   1.063076  |
    TREE_PCT                :  -0.053785  |
    osdist_mean             :  -0.019653  |
    os_per_person_pctle     :   0.052579  |
    longitude               :  -0.152452  |
    latitude                :   0.238275  |
    Shape_Area              :   0.093486  |
    Single Family(Res Use/Zone):   0.197502  |
    Townhouse Plat          :  -0.099342  |
    Vacant(Single-family)   :   0.000788  |
    Duplex                  :   0.136813  |
    Apartment               :   0.105723  |
    Single Family(C/I Zone) :   0.012633  |
    Triplex                 :   0.004708  |
    Condominium(Residential):  -0.007958  |
    4-Plex                  :  -0.001327  |
    Mobile Home             :  -0.090309  |
    Retail Store            :  -0.049221  |
    Vacant(Multi-family)    :  -0.004983  |
    Office Building         :  -0.052206  |
    Apartment(Mixed Use)    :   0.095634  |
    Vacant(Commercial)      :   0.080392  |
    Church/Welfare/Relig Srvc:   0.000000  |
    Parking(Assoc)          :   0.008222  |
    Park, Public(Zoo/Arbor) :   0.000000  |
    ----------------------------------------
    Model trian score       : 0.528839
    Model test score        : 0.513588
    Model intercept         : 81.876177
    Model r_sq score        : 0.528839
    Mean Absolute Error     : 1.584966
    Mean Sq Error Train     : 2.036806
    Mean Sq Error Test      : 2.066766
    Mean Sq Error Diff      : 0.029960
    
