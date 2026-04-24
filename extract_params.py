import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"D:\Multiscreen-Addiction\teen_multiscreen_addiction_dataset.csv.xls")
df_clean = df.drop(columns=['ID', 'Name', 'Location'])

grade_order = {'6th':6,'7th':7,'8th':8,'9th':9,'10th':10,'11th':11,'12th':12}
df_clean['School_Grade'] = df_clean['School_Grade'].map(grade_order)

le = LabelEncoder()
for col in ['Gender', 'Phone_Usage_Purpose']:
    df_clean[col] = le.fit_transform(df_clean[col])

df_clean['Phone_Active_Screen']     = df_clean['Time_on_Social_Media'] + df_clean['Time_on_Gaming'] + df_clean['Time_on_Education']
df_clean['Phone_Check_Intensity']   = df_clean['Phone_Checks_Per_Day'] / (df_clean['Daily_Usage_Hours'] + 1e-5)
df_clean['Weekend_Weekday_Ratio']   = df_clean['Weekend_Usage_Hours'] / (df_clean['Daily_Usage_Hours'] + 1e-5)
df_clean['Total_Laptop_Hours']      = df_clean['Laptop_Study_Hours'] + df_clean['Laptop_Gaming_TimePass_Hours'] + df_clean['Laptop_Usage_Before_Bed_Hours']
df_clean['Laptop_Productive_Ratio'] = df_clean['Laptop_Study_Hours'] / (df_clean['Total_Laptop_Hours'] + 1e-5)
df_clean['Gaming_Cross_Device']     = df_clean['Laptop_Gaming_TimePass_Hours'] / (df_clean['Time_on_Gaming'] + 1e-5)
df_clean['Total_All_Screen_Hours']  = df_clean['Daily_Usage_Hours'] + df_clean['Total_Laptop_Hours']
df_clean['Total_Before_Bed_Screen'] = df_clean['Screen_Time_Before_Bed'] + df_clean['Laptop_Usage_Before_Bed_Hours']
df_clean['Sleep_Deficit']           = df_clean['Sleep_Hours'] - 9
df_clean['Mental_Health_Score']     = df_clean['Anxiety_Level'] + df_clean['Depression_Level'] - df_clean['Self_Esteem']

X = df_clean.drop(columns=['Addiction_Level'])
y = df_clean['Addiction_Level']
feature_cols = list(X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# KNN
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
}
knn_search = GridSearchCV(KNeighborsRegressor(), knn_param_grid, cv=3, scoring='r2', n_jobs=1)
knn_search.fit(X_train, y_train)
print("KNN Best:", knn_search.best_params_)

# SVM
svm_param_grid = {
    'kernel': ['rbf'],
    'C': [1, 10, 50, 100],
    'epsilon': [0.01, 0.05, 0.1],
    'gamma': ['scale', 'auto'],
}
svm_search = GridSearchCV(SVR(), svm_param_grid, cv=3, scoring='r2', n_jobs=1)
svm_search.fit(X_train, y_train)
print("SVM Best:", svm_search.best_params_)

# XGBoost
xgb_param_dist = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2],
}
xgb_search = RandomizedSearchCV(XGBRegressor(verbosity=0, random_state=42), xgb_param_dist, n_iter=20, cv=3, scoring='r2', n_jobs=1, random_state=42)
xgb_search.fit(X_train, y_train)
print("XGBoost Best:", xgb_search.best_params_)

# Random Forest
rf_param_dist = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 16, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 0.5, 0.7],
}
rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=1), rf_param_dist, n_iter=20, cv=3, scoring='r2', n_jobs=1, random_state=42)
rf_search.fit(X_train, y_train)
print("Random Forest Best:", rf_search.best_params_)
