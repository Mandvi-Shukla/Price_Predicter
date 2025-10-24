# ============================================================
# PROJECT: Boston House Price Prediction
# PHASE: Data Preparation + Model Training
# AUTHOR: Mandvi
# ============================================================
import os
os.chdir(r"c:\mandvi\machine learning\final project")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib

# ============================================================
# 1. LOAD DATA
# ============================================================

housing = pd.read_csv("HousingData.csv")

# Handle NaNs in CHAS before splitting
housing["CHAS"] = housing["CHAS"].fillna(0)  # or dropna(subset=["CHAS"])

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(housing, housing["CHAS"]):
    strat_train_set = housing.iloc[train_idx].copy()
    strat_test_set = housing.iloc[test_idx].copy()

# ============================================================
# 3. FEATURE SELECTION
# ============================================================

housing_labels = housing["MEDV"].copy()
housing = housing.drop("MEDV", axis=1)

# ============================================================
# 4. PIPELINE SETUP
# ============================================================

num_features = housing.drop("CHAS", axis=1).columns
cat_features = ["CHAS"]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', OneHotEncoder(), cat_features)
])

housing_prepared = full_pipeline.fit_transform(housing)

# ============================================================
# 5. MODEL TRAINING & CROSS VALIDATION
# ============================================================

def evaluate_model(model, X, y):
    """Perform 5-fold CV and return RMSE."""
    scores = cross_val_score(model, X, y,
                             scoring="neg_mean_squared_error", cv=5)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores.mean(), rmse_scores.std()

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

print("\n================= MODEL EVALUATION =================")
results = []

for name, model in models.items():
    mean_rmse, std_rmse = evaluate_model(model, housing_prepared, housing_labels)
    results.append((name, mean_rmse, std_rmse))
    print(f"{name:<25} | RMSE: {mean_rmse:.2f} ± {std_rmse:.2f}")

# ============================================================
# 6. CHOOSE BEST MODEL
# ============================================================

best_model_name, best_rmse, _ = min(results, key=lambda x: x[1])
best_model = models[best_model_name]
best_model.fit(housing_prepared, housing_labels)

print("\n====================================================")
print(f" Best Model: {best_model_name} with RMSE = {best_rmse:.2f}")
print("====================================================\n")

# ============================================================
# 7. TEST SET EVALUATION
# ============================================================

X_test = strat_test_set.drop("MEDV", axis=1)
y_test = strat_test_set["MEDV"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = best_model.predict(X_test_prepared)

final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print(f"Final Test RMSE: {final_rmse:.2f}")

# ============================================================
# 8. SAVE MODEL + PIPELINE
# ============================================================

joblib.dump(best_model, "best_model.pkl")
joblib.dump(full_pipeline, "preprocessing_pipeline.pkl")

print("\n Model and pipeline saved successfully!")





