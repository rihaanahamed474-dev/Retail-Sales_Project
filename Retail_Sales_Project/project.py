# %% [markdown]
# # Retail Sales Analysis and Sales Prediction
# **Steps:** 1) Load data 2) Clean 3) Exploratory analysis 4) Feature engineering 5) Train models 6) Evaluate 7) Forecast next 30 days
# %%
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os, warnings; warnings.filterwarnings("ignore")
os.makedirs("images", exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")
# %% [markdown]
# ## 1. Load the dataset
# %%
df = pd.read_csv("retail_sales.csv", parse_dates=["Order_Date"])
print(df.shape); df.head()
# %%
df.info(); df.describe()
# %% [markdown]
# ## 2. Data cleaning
# %%
print("Missing values:\n", df.isnull().sum()[lambda s: s > 0])
print("Duplicate rows:", df.duplicated().sum())
df["Discount"] = df["Discount"].fillna(df["Discount"].median())   # fill missing discount with median
df = df.drop_duplicates()
# %% [markdown]
# ## 3. Exploratory Data Analysis
# %%
df["Year"] = df.Order_Date.dt.year; df["Month"] = df.Order_Date.dt.month
df["Weekday"] = df.Order_Date.dt.day_name()
print(f"Total Sales : {df.Sales.sum():,.0f}\nTotal Profit: {df.Profit.sum():,.0f}\nOrders      : {len(df):,}")
# %%
monthly = df.groupby(df.Order_Date.dt.to_period("M")).Sales.sum().to_timestamp()
plt.figure(figsize=(11, 4)); monthly.plot(marker="o")
plt.title("Monthly Sales Trend"); plt.ylabel("Sales (INR)"); plt.tight_layout()
plt.savefig("images/monthly_trend.png", dpi=150); plt.show()
# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df.groupby("Category").Sales.sum().sort_values().plot.barh(ax=ax[0], title="Sales by Category")
df.groupby("Region").Sales.sum().sort_values().plot.barh(ax=ax[1], title="Sales by Region", color="coral")
plt.tight_layout(); plt.savefig("images/category_region.png", dpi=150); plt.show()
# %%
pivot = df.pivot_table(index="Weekday", columns="Month", values="Sales", aggfunc="sum")
pivot = pivot.reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
plt.figure(figsize=(11, 4)); sns.heatmap(pivot, cmap="YlGnBu"); plt.title("Sales Heatmap: Weekday vs Month")
plt.tight_layout(); plt.savefig("images/heatmap.png", dpi=150); plt.show()
# %%
print(df.groupby("Product").Sales.sum().nlargest(5))
print(df.groupby("Discount")[["Sales", "Profit"]].mean().round(0))
sns.heatmap(df[["Unit_Price","Quantity","Discount","Sales","Profit"]].corr(), annot=True, cmap="coolwarm"); plt.show()
# %% [markdown]
# ## 4. Feature engineering (weekly sales)
# Order-level and daily sales are very noisy, so we forecast **weekly** total sales.
# %%
weekly = df.set_index("Order_Date").Sales.resample("W-SUN").sum().to_frame()
weekly = weekly.iloc[1:-1]                      # drop incomplete first/last week
def make_features(idx):
    f = pd.DataFrame(index=idx)
    f["month"] = idx.month; f["weekofyear"] = idx.isocalendar().week.astype(int).values
    f["quarter"] = idx.quarter; f["is_festive"] = idx.month.isin([11, 12]).astype(int)
    f["t"] = np.arange(len(idx)) if len(idx) == len(weekly) else ((idx - weekly.index.min()).days // 7)
    return f
X, y = make_features(weekly.index), weekly.Sales
X["t"] = (weekly.index - weekly.index.min()).days // 7   # trend index (weeks since start)
X = X.drop(columns=["month"]).join(pd.get_dummies(X["month"], prefix="m", dtype=int))   # one-hot months
# %% [markdown]
# ## 5. Train / test split (time based - last 26 weeks = test)
# %%
split = len(X) - 26
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
          "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=42)}
results, preds = [], {}
for name, m in models.items():
    m.fit(X_train, y_train); p = m.predict(X_test); preds[name] = p
    results.append([name, mean_absolute_error(y_test, p), np.sqrt(mean_squared_error(y_test, p)), r2_score(y_test, p)])
res = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"]).round(3); res
# %% [markdown]
# ## 6. Evaluation
# %%
best = res.sort_values("RMSE").iloc[0].Model; print("Best model:", best)
plt.figure(figsize=(11, 4)); plt.plot(y_test.index, y_test, marker="o", label="Actual")
plt.plot(y_test.index, preds[best], marker="s", label=f"Predicted ({best})"); plt.legend()
plt.title("Actual vs Predicted Weekly Sales (last 26 weeks)"); plt.tight_layout()
plt.savefig("images/actual_vs_pred.png", dpi=150); plt.show()
# %%
rf = models["Random Forest"]
pd.Series(rf.feature_importances_, index=X.columns).sort_values().plot.barh(title="Feature Importance (Random Forest)")
plt.tight_layout(); plt.savefig("images/importance.png", dpi=150); plt.show()
# %% [markdown]
# ## 7. Forecast the next 12 weeks
# %%
final = models[best].fit(X, y)
fidx = pd.date_range(weekly.index.max() + pd.Timedelta(weeks=1), periods=12, freq="W-SUN")
F = make_features(fidx); F["t"] = (fidx - weekly.index.min()).days // 7
F = F.drop(columns=["month"]).join(pd.get_dummies(F["month"], prefix="m", dtype=int))
F = F.reindex(columns=X.columns, fill_value=0)
forecast = pd.Series(final.predict(F[X.columns]), index=fidx)
plt.figure(figsize=(11, 4)); plt.plot(weekly.index[-52:], y[-52:], label="History")
plt.plot(forecast.index, forecast, label="Forecast (12 weeks)", color="red", marker="o"); plt.legend(); plt.title("12-Week Sales Forecast")
plt.tight_layout(); plt.savefig("images/forecast.png", dpi=150); plt.show()
print(f"Expected sales next 12 weeks: {forecast.sum():,.0f}")
# %% [markdown]
# ## Conclusion
# Sales peak in Nov-Dec; Electronics drives most revenue; deep discounts (30%) turn profit negative. Compare the model scores above to pick the best forecaster.
