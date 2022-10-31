# Starting Model Section
target = 'LifeExpectancy'
# predictors = 'price', 'sqft_living', 'sqft_lot', 'yr_built', 'sqft_living15', 'sqft_lot15', 'bins', 'POC_pct', 'median_income', 'TREE_PCT', 'osdist_mean', 'os_per_person_pctle', 'longitude', 'latitude',                     'Shape_Area', 'Single Family(Res Use/Zone)', 'Townhouse Plat',                            'Vacant(Single-family)', 'Duplex', 'Apartment',                                   'Single Family(C/I Zone)', 'Triplex', 'Condominium(Residential)',                                          '4-Plex', 'Mobile Home', 'Retail Store', 'Vacant(Multi-family)',                                                 'Office Building', 'Apartment(Mixed Use)', 'Vacant(Commercial)',                                                        'Church/Welfare/Relig Srvc', 'Parking(Assoc)',                                                               'Park, Public(Zoo/Arbor)'X_ex01 = comb_parcel_house_dna.drop(['id', target, 'PREUSE_DESC', 'grade', 'income_3rd', 'long', 'lat','sqft_above', 'view'], axis=1)
predictors = ['price', 'sqft_living', 'sqft_lot', 'POC_pct', 'median_income', 'TREE_PCT', 
               'osdist_mean', 'os_per_person_pctle', 'longitude', 'latitude',                            
               'Vacant(Single-family)','Office Building', 'Apartment(Mixed Use)']

X_ex01 = comb_parcel_house_dna[predictors]
y_ex01 = comb_parcel_house_dna[target]

X_train_ex01, X_test_ex01, y_train_ex01, y_test_ex01 = train_test_split(X_ex01, y_ex01, test_size=.33, random_state=133)
X_num_train_ex01 = X_train_ex01.select_dtypes([np.float64, np.int64])
X_obj_train_ex01 = X_train_ex01.select_dtypes(np.object)
X_test_obj_ex01 = X_test_ex01.select_dtypes(np.object)
X_test_num_ex01 = X_test_ex01.select_dtypes([np.float64, np.int64])


scaler = StandardScaler()
scaler.fit(X_num_train_ex01)
X_train_scaled_ex01 = scaler.transform(X_num_train_ex01)
X_train_scaled_ex01 = pd.DataFrame(X_train_scaled_ex01, columns=X_num_train_ex01.columns, index=X_num_train_ex01.index)


X_test_scaled_ex01 = scaler.transform(X_test_num_ex01)
X_test_scaled_ex01 = pd.DataFrame(X_test_scaled_ex01, columns=X_test_num_ex01.columns, index=X_test_num_ex01.index)

model_sk = LinearRegression()
model_sk.fit(X_train_scaled_ex01, y_train_ex01)
score_train_ex01 = model_sk.score(X_train_scaled_ex01, y_train_ex01)
score_test_ex01 = model_sk.score(X_test_scaled_ex01, y_test_ex01)

y_train_pred_ex01 = model_sk.predict(X_train_scaled_ex01)
y_test_pred_ex01 = model_sk.predict(X_test_scaled_ex01)

coef_ex01 = dict(zip(X_train_ex01.columns, model_sk.coef_))

for item, val in coef_ex01.items():
    a = item
    a_s = " " * (24 - len(a))
    b = val
    print(f"{a}{a_s}: {b:10.6f}  |")
y_intrcpt_ex01 = model_sk.intercept_
r_score_ex01 = r2_score(y_train_ex01, y_train_pred_ex01)
m_a_err_ex01 = mean_absolute_error(y_test_pred_ex01, y_test_ex01)
rms_err_train_ex01 = np.sqrt(mean_squared_error(y_train_ex01, y_train_pred_ex01))
rms_err_test_ex01 = np.sqrt(mean_squared_error(y_test_ex01, y_test_pred_ex01))
rms_err_diff_ex01 = abs(rms_err_train_ex01 - rms_err_test_ex01)
print("-"*40)
print(f"Model trian score       : {score_train_ex01:06f}")
print(f"Model test score        : {score_test_ex01:06f}")
print(f"Model intercept         : {y_intrcpt_ex01:06f}")
print(f"Model r_sq score        : {r_score_ex01:06f}")
print(f"Mean Absolute Error     : {m_a_err_ex01:06f}")
print(f"Mean Sq Error Train     : {rms_err_train_ex01:06f}")
print(f"Mean Sq Error Test      : {rms_err_test_ex01:06f}")
print(f"Mean Sq Error Diff      : {rms_err_diff_ex01:06f}")

