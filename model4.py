# Starting Model Section
X_p02 = comb_parcel_house_dna.drop(['id', 'price', 'grade', 'income_3rd', 'long', 'lat','sqft_above', 'view'], axis=1)
y_p02 = comb_parcel_house_dna['price']

X_train_p02, X_test_p02, y_train_p02, y_test_p02 = train_test_split(X_p02, y_p02, test_size=.33, random_state=133)
X_num_train_p02 = X_train_p02.select_dtypes([np.float64, np.int64])
X_obj_train_p02 = X_train_p02.select_dtypes(np.object)
X_test_obj_p02 = X_test_p02.select_dtypes(np.object)
X_test_num_p02 = X_test_p02.select_dtypes([np.float64, np.int64])


scaler = StandardScaler()
scaler.fit(X_num_train_p02)
X_train_scaled_p02 = scaler.transform(X_num_train_p02)
X_train_scaled_p02 = pd.DataFrame(X_train_scaled_p02, columns=X_num_train_p02.columns, index=X_num_train_p02.index)


X_test_scaled_p02 = scaler.transform(X_test_num_p02)
X_test_scaled_p02 = pd.DataFrame(X_test_scaled_p02, columns=X_test_num_p02.columns, index=X_test_num_p02.index)

model_sk = LinearRegression()
model_sk.fit(X_train_scaled_p02, y_train_p02)
score_train_p02 = model_sk.score(X_train_scaled_p02, y_train_p02)
score_test_p02 = model_sk.score(X_test_scaled_p02, y_test_p02)

y_train_pred_p02 = model_sk.predict(X_train_scaled_p02)
y_test_pred_p02 = model_sk.predict(X_test_scaled_p02)

coef_p02 = dict(zip(X_train_p02.columns, model_sk.coef_))

for item, val in coef_p02.items():
    a = item
    a_s = " " * (24 - len(a))
    b = val
    print(f"{a}{a_s}: {b:10.6f}  |")
y_intrcpt_p02 = model_sk.intercept_
r_score_p02 = r2_score(y_train_p02, y_train_pred_p02)
m_a_err_p02 = mean_absolute_error(y_test_pred_p02, y_test_p02)
rms_err_train_p02 = np.sqrt(mean_squared_error(y_train_p02, y_train_pred_p02))
rms_err_test_p02 = np.sqrt(mean_squared_error(y_test_p02, y_test_pred_p02))
rms_err_diff_p02 = abs(rms_err_train_p02 - rms_err_test_p02)
print("-"*40)
print(f"Model trian score       : {score_train_p02:06f}")
print(f"Model test score        : {score_test_p02:06f}")
print(f"Model intercept         : {y_intrcpt_p02:06f}")
print(f"Model r_sq score        : {r_score_p02:06f}")
print(f"Mean Absolute Error     : {m_a_err_p02:06f}")
print(f"Mean Sq Error Train     : {rms_err_train_p02:06f}")
print(f"Mean Sq Error Test      : {rms_err_test_p02:06f}")
print(f"Mean Sq Error Diff      : {rms_err_diff_p02:06f}")

