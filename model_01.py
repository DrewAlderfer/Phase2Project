# Starting Model Section
X = comb_parcel_house_dna.drop('price', axis=1)
y = comb_parcel_house_dna['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=133)
X_num_train = X_train.select_dtypes([np.float64, np.int64])
X_obj_train = X_train.select_dtypes(np.object)
X_test_obj = X_test.select_dtypes(np.object)
X_test_num = X_test.select_dtypes([np.float64, np.int64])

scaler = StandardScaler()
scaler.fit(X_num_train)
X_train_scaled = scaler.transform(X_num_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_num_train.columns, index=X_num_train.index)

X_test_scaled = scaler.transform(X_test_num)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_num.columns, index=X_test_num.index)

model_sk = LinearRegression()
model_sk.fit(X_train_scaled, y_train)
model_sk.score(X_train_scaled, y_train)

train_preds = model_sk.predict(X_train_scaled)
y_pred = model_sk.predict(X_test_scaled)

coef_01 = dict(zip(X_train.columns, model_sk.coef_))

for item, val in coef_01.items():
    a = item
    a_s = " " * (24 - len(a))
    b = val
    print(f"{a}{a_s}: {b:10.6f}")
y_intrcpt = model_sk.intercept_
r_score = r2_score(y_train, train_preds)
m_a_err = mean_absolute_error(y_pred, y_test)
rm_sq_err = np.sqrt(mean_squared_error(y_pred, y_test))
print("-"*40)
print(f"Model intercept         : {y_intrcpt:06f}")
print(f"Model r_sq score        : {r_score:06f}")
print(f"Mean Absolute Error     : {m_a_err:06f}")
print(f"Root Mean Sq Error      : {rm_sq_err:06f}")
