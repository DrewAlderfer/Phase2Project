# 'id', 'price', 'sqft_living', 'sqft_lot', 'grade', 'sqft_above', 'yr_built', 'lat', 'long',      
# 'sqft_living15', 'sqft_lot15', 'POC_pct', 'median_income', 'income_3rd', 'LifeExpectancy', 
# 'TREE_PCT', 'osdist_mean', 'os_per_person_pctle', 'longitude', 'latitude', 'Shape_Area', 
# 'Single Family(Res Use/Zone)', 'Townhouse Plat', 'Vacant(Single-family)', 'Duplex', 'Apartment', 
# 'Single Family(C/I Zone)', 'Triplex', 'Condominium(Residential)', '4-Plex', 'Mobile Home', 
# 'Retail Store', 'Vacant(Multi-family)', 'Office Building', 'Apartment(Mixed Use)', 
# 'Vacant(Commercial)', 'Church/Welfare/Relig Srvc', 'Parking(Assoc)','Park, Public(Zoo/Arbor)'

target = 'LifeExpectancy'
predictors = \
['price', 'sqft_living', 'sqft_lot', 'grade', 'sqft_above',      
 'sqft_living15', 'sqft_lot15', 'POC_pct', 'median_income', 'LifeExpectancy', 
 'TREE_PCT', 'osdist_mean', 'os_per_person_pctle', 'longitude', 'latitude', 'Shape_Area', 
 'Single Family(Res Use/Zone)', 'Townhouse Plat', 'Vacant(Single-family)', 'Duplex', 'Apartment', 
 'Single Family(C/I Zone)', 'Triplex', 'Condominium(Residential)', '4-Plex', 'Mobile Home', 
 'Retail Store', 'Vacant(Multi-family)', 'Office Building', 'Apartment(Mixed Use)', 
 'Vacant(Commercial)', 'Church/Welfare/Relig Srvc', 'Parking(Assoc)','Park, Public(Zoo/Arbor)']

predictors.remove(target)

comb_parcel_house_dna = pd.read_pickle('./test_train_set03.pkl')

X_ex03 = comb_parcel_house_dna[predictors]
y_ex03 = comb_parcel_house_dna[target]

X_train_ex03, X_test_ex03, y_train_ex03, y_test_ex03 = train_test_split(X_ex03, y_ex03, test_size=.33, random_state=133)
X_num_train_ex03 = X_train_ex03.select_dtypes([np.float64, np.int64])
X_obj_train_ex03 = X_train_ex03.select_dtypes(np.object)
X_test_obj_ex03 = X_test_ex03.select_dtypes(np.object)
X_test_num_ex03 = X_test_ex03.select_dtypes([np.float64, np.int64])


scaler = StandardScaler()
scaler.fit(X_num_train_ex03)
X_train_scaled_ex03 = scaler.transform(X_num_train_ex03)
X_train_scaled_ex03 = pd.DataFrame(X_train_scaled_ex03, columns=X_num_train_ex03.columns, index=X_num_train_ex03.index)


X_test_scaled_ex03 = scaler.transform(X_test_num_ex03)
X_test_scaled_ex03 = pd.DataFrame(X_test_scaled_ex03, columns=X_test_num_ex03.columns, index=X_test_num_ex03.index)

model_sk = LinearRegression()
model_sk.fit(X_train_scaled_ex03, y_train_ex03)
score_train_ex03 = model_sk.score(X_train_scaled_ex03, y_train_ex03)
score_test_ex03 = model_sk.score(X_test_scaled_ex03, y_test_ex03)

y_train_pred_ex03 = model_sk.predict(X_train_scaled_ex03)
y_test_pred_ex03 = model_sk.predict(X_test_scaled_ex03)

coef_ex03 = dict(zip(X_train_ex03.columns, model_sk.coef_))

for item, val in coef_ex03.items():
    a = item
    a_s = " " * (24 - len(a))
    b = val
    print(f"{a}{a_s}: {b:10.6f}  |")
y_intrcpt_ex03 = model_sk.intercept_
r_score_ex03 = r2_score(y_train_ex03, y_train_pred_ex03)
m_a_err_ex03 = mean_absolute_error(y_test_pred_ex03, y_test_ex03)
rms_err_train_ex03 = np.sqrt(mean_squared_error(y_train_ex03, y_train_pred_ex03))
rms_err_test_ex03 = np.sqrt(mean_squared_error(y_test_ex03, y_test_pred_ex03))
rms_err_diff_ex03 = abs(rms_err_train_ex03 - rms_err_test_ex03)
print("-"*40)
print(f"Model trian score       : {score_train_ex03:06f}")
print(f"Model test score        : {score_test_ex03:06f}")
print(f"Model intercept         : {y_intrcpt_ex03:06f}")
print(f"Model r_sq score        : {r_score_ex03:06f}")
print(f"Mean Absolute Error     : {m_a_err_ex03:06f}")
print(f"Mean Sq Error Train     : {rms_err_train_ex03:06f}")
print(f"Mean Sq Error Test      : {rms_err_test_ex03:06f}")
print(f"Mean Sq Error Diff      : {rms_err_diff_ex03:06f}")

