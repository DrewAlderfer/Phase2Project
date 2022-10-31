```
# Starting Model Section
X_t01 = comb_parcel_house_dna.drop(['id','TREE_PCT', 'grade', 'income_3rd', 'long', 'lat','sqft_above', 'view'], axis=1)
y_t01 = comb_parcel_house_dna['TREE_PCT']
```


```
X_train_t01, X_test_t01, y_train_t01, y_test_t01 = train_test_split(X_t01, y_t01, test_size=.33, random_state=133)
X_num_train_t01 = X_train_t01.select_dtypes([np.float64, np.int64])
X_obj_train_t01 = X_train_t01.select_dtypes(np.object)
X_test_obj_t01 = X_test_t01.select_dtypes(np.object)
X_test_num_t01 = X_test_t01.select_dtypes([np.float64, np.int64])
```


```
scaler = StandardScaler()
scaler.fit(X_num_train_t01)
X_train_scaled_t01 = scaler.transform(X_num_train_t01)
X_train_scaled_t01 = pd.DataFrame(X_train_scaled_t01, columns=X_num_train_t01.columns, index=X_num_train_t01.index)
```


```
X_test_scaled_t01 = scaler.transform(X_test_num_t01)
X_test_scaled_t01 = pd.DataFrame(X_test_scaled_t01, columns=X_test_num_t01.columns, index=X_test_num_t01.index)
```


```
model_sk = LinearRegression()
model_sk.fit(X_train_scaled_t01, y_train_t01)
score_train_t01 = model_sk.score(X_train_scaled_t01, y_train_t01)
score_test_t01 = model_sk.score(X_test_scaled_t01, y_test_t01)
```


```
y_train_pred_t01 = model_sk.predict(X_train_scaled_t01)
y_test_pred_t01 = model_sk.predict(X_test_scaled_t01)
```


```
coef_t01 = dict(zip(X_train_t01.columns, model_sk.coef_))
```


```
for item, val in coef_t01.items():
    a = item
    a_s = " " * (24 - len(a))
    b = val
    print(f"{a}{a_s}: {b:10.6f}  |")
y_intrcpt_t01 = model_sk.intercept_
r_score_t01 = r2_score(y_train_t01, y_train_pred_t01)
m_a_err_t01 = mean_absolute_error(y_test_pred_02, y_test_t01)
rms_err_train_t01 = np.sqrt(mean_squared_error(y_train_t01, y_train_pred_t01))
rms_err_test_t01 = np.sqrt(mean_squared_error(y_test_t01, y_test_pred_t01))
rms_err_diff_t01 = abs(rms_err_train_t01 - rms_err_test_t01)
print("-"*40)
print(f"Model trian score       : {score_train_t01:06f}")
print(f"Model test score        : {score_test_t01:06f}")
print(f"Model intercept         : {y_intrcpt_t01:06f}")
print(f"Model r_sq score        : {r_score_t01:06f}")
print(f"Mean Absolute Error     : {m_a_err_t01:06f}")
print(f"Mean Sq Error Train     : {rms_err_train_t01:06f}")
print(f"Mean Sq Error Test      : {rms_err_test_t01:06f}")
print(f"Mean Sq Error Diff      : {rms_err_diff_t01:06f}")
```
