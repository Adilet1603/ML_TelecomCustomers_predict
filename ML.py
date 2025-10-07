# import pandas as pd
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
#
# df = pd.read_csv("TelcoCustomers.csv")
# df = df.dropna(how='all')
# df = df.fillna('Unknown')
# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
#
# features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "OnlineSecurity", "TechSupport"]
# target = "Churn"
#
# X = df[features].copy()
# y = df[target].copy()
#
# contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
# internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
# online_map = {"Yes": 0, "No": 1, "No internet service": 2}
# tech_map = {"Yes": 0, "No": 1, "No internet service": 2}
#
# X["Contract"] = X["Contract"].map(contract_map)
# X["InternetService"] = X["InternetService"].map(internet_map)
# X["OnlineSecurity"] = X["OnlineSecurity"].map(online_map)
# X["TechSupport"] = X["TechSupport"].map(tech_map)
#
# y = LabelEncoder().fit_transform(y)
#
# num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
# scaler = StandardScaler()
# X[num_cols] = scaler.fit_transform(X[num_cols])
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# model = LogisticRegression(max_iter=1000, random_state=42)
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]
#
# print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
# print("Precision:", round(precision_score(y_test, y_pred), 3))
# print("Recall:", round(recall_score(y_test, y_pred), 3))
# print("F1-score:", round(f1_score(y_test, y_pred), 3))
# print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))
#
# joblib.dump(model, "model_telco.pkl")
# joblib.dump(scaler, "scaler_telco.pkl")
#
# print("📊 Кол-во фичей:", X.shape[1])
