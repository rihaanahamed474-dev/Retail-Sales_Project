"""Creates retail_sales.csv - a Superstore-style transactional retail dataset (synthetic).
Swap it for a real Kaggle dataset (e.g. 'Superstore Sales', 'Walmart Sales') if you prefer."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
N = 20000
dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 1461, N), unit="D")
# more orders in Nov-Dec, on weekends, and a growth trend over time
w = (1 + 0.6*dates.month.isin([11, 12]) + 0.25*dates.dayofweek.isin([5, 6])
     + 0.15*(dates - dates.min()).days/1461).values
dates = dates[rng.choice(N, N, p=w/w.sum())]
catalog = {
 "Electronics": {"Smartphone": 18000, "Headphones": 2500, "Smart Watch": 6000, "Tablet": 22000},
 "Clothing": {"T-Shirt": 600, "Jeans": 1800, "Jacket": 3500, "Sneakers": 2800},
 "Grocery": {"Rice 10kg": 750, "Cooking Oil": 420, "Snacks Pack": 250, "Coffee": 520},
 "Home & Kitchen": {"Mixer Grinder": 3200, "Cookware Set": 2400, "Bed Sheet": 900, "Lamp": 750},
 "Beauty": {"Face Cream": 450, "Perfume": 1500, "Shampoo": 350, "Makeup Kit": 1900},
}
cats = rng.choice(list(catalog), N, p=[.2, .25, .3, .15, .1])
prods = np.array([rng.choice(list(catalog[c])) for c in cats])
price = np.array([catalog[c][p] for c, p in zip(cats, prods)]) * rng.normal(1, .05, N)
qty = rng.integers(1, 6, N)
disc = rng.choice([0, .05, .10, .15, .20, .30], N, p=[.35, .2, .2, .12, .08, .05])
sales = price*qty*(1-disc)
cost = price*qty*rng.uniform(.6, .85, N)
df = pd.DataFrame({
 "Order_ID": [f"ORD-{100000+i}" for i in range(N)],
 "Order_Date": dates.date, "Store": rng.choice([f"Store_{i}" for i in range(1, 9)], N),
 "Region": rng.choice(["North", "South", "East", "West"], N, p=[.2, .35, .2, .25]),
 "Customer_Segment": rng.choice(["Consumer", "Corporate", "Home Office"], N, p=[.55, .3, .15]),
 "Category": cats, "Product": prods, "Unit_Price": price.round(2), "Quantity": qty,
 "Discount": disc, "Payment_Mode": rng.choice(["UPI", "Card", "Cash", "Net Banking"], N, p=[.4, .3, .2, .1]),
 "Sales": sales.round(2), "Profit": (sales-cost).round(2)})
df.loc[rng.choice(N, 150, replace=False), "Discount"] = np.nan   # a few missing values to clean
df.sort_values("Order_Date").to_csv("retail_sales.csv", index=False)
print(df.shape)
