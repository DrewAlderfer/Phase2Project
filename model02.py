# Starting Model Section
X_le = comb_parcel_house_dna.drop('LifeExpectancy', axis=1)
y_le = comb_parcel_house_dna['LifeExpectancy']

X_train_le, X_test_le, y_train_le, y_test_le = train_test_split(X_le, y_le, test_size=.33, random_state=133)
X_num_train_le = X_train_le.select_dtypes([np.float64, np.int64])
X_obj_train_le = X_train_le.select_dtypes(np.object)
X_test_obj_le = X_test_le.select_dtypes(np.object)
X_test_num_le = X_test_le.select_dtypes([np.float64, np.int64])

scaler = StandardScaler()
scaler.fit(X_num_train_le)
X_train_scaled_le = scaler.transform(X_num_train_le)
X_train_scaled_le = pd.DataFrame(X_train_scaled_le, columns=X_num_train_le.columns, index=X_num_train_le.index)

X_test_scaled_le = scaler.transform(X_test_num_le)
X_test_scaled_le = pd.DataFrame(X_test_scaled_le, columns=X_test_num_le.columns, index=X_test_num_le.index)

model_sk = LinearRegression()
model_sk.fit(X_train_scaled_le, y_train_le)
model_sk.score(X_train_scaled_le, y_train_le)

train_pred_le = model_sk.predict(X_train_scaled_le)
y_pred_le = model_sk.predict(X_test_scaled_le)

coef_le = dict(zip(X_train_le.columns, model_sk.coef_))

for item, val in coef_le.items():
    a = item
    a_s = " " * (24 - len(a))
    b = val
    print(f"{a}{a_s}: {b:10.6f}  |")
y_intrcpt_le = model_sk.intercept_
r_score_le = r2_score(y_train_le, train_pred_le)
m_a_err_le = mean_absolute_error(y_pred_le, y_test_le)
rm_sq_err_le = np.sqrt(mean_squared_error(y_pred_le, y_test_le))
print("-"*40)
print(f"Model intercept         : {y_intrcpt_le:06f}")
print(f"Model r_sq score        : {r_score_le:06f}")
print(f"Mean Absolute Error     : {m_a_err_le:06f}")
print(f"Root Mean Sq Error      : {rm_sq_err_le:06f}")
