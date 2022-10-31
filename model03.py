# Starting Model Section
X_le2 = comb_parcel_house_dna.drop(['LifeExpectancy', 'income_3rd', 'long', 'latitude','sqft_above', 'view'], axis=1)
y_le2 = comb_parcel_house_dna['LifeExpectancy']

X_train_le2, X_test_le2, y_train_le2, y_test_le2 = train_test_split(X_le2, y_le2, test_size=.33, random_state=133)
X_num_train_le2 = X_train_le2.select_dtypes([np.float64, np.int64])
X_obj_train_le2 = X_train_le2.select_dtypes(np.object)
X_test_obj_le2 = X_test_le2.select_dtypes(np.object)
X_test_num_le2 = X_test_le2.select_dtypes([np.float64, np.int64])

scaler = StandardScaler()
scaler.fit(X_num_train_le2)
X_train_scaled_le2 = scaler.transform(X_num_train_le2)
X_train_scaled_le2 = pd.DataFrame(X_train_scaled_le2, columns=X_num_train_le2.columns, index=X_num_train_le2.index)

X_test_scaled_le2 = scaler.transform(X_test_num_le2)
X_test_scaled_le2 = pd.DataFrame(X_test_scaled_le2, columns=X_test_num_le2.columns, index=X_test_num_le2.index)

model_sk = LinearRegression()
model_sk.fit(X_train_scaled_le2, y_train_le2)
model_sk.score(X_train_scaled_le2, y_train_le2)

train_pred_le2 = model_sk.predict(X_train_scaled_le2)
y_pred_le2 = model_sk.predict(X_test_scaled_le2)

coef_le2 = dict(zip(X_train_le2.columns, model_sk.coef_))

for item, val in coef_le2.items():
    a = item
    a_s = " " * (24 - len(a))
    b = val
    print(f"{a}{a_s}: {b:10.6f}  |")
y_intrcpt_le2 = model_sk.intercept_
r_score_le2 = r2_score(y_train_le2, train_pred_le2)
m_a_err_le2 = mean_absolute_error(y_pred_le2, y_test_le2)
rm_sq_err_le2 = np.sqrt(mean_squared_error(y_pred_le2, y_test_le2))
print("-"*40)
print(f"Model intercept         : {y_intrcpt_le2:06f}")
print(f"Model r_sq score        : {r_score_le2:06f}")
print(f"Mean Absolute Error     : {m_a_err_le2:06f}")
print(f"Root Mean Sq Error      : {rm_sq_err_le2:06f}")
