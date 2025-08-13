import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

CSV_PATH = "../csv/FINALIZED_cities_data.csv"
df = pd.read_csv(CSV_PATH)

if "City" not in df.columns:
    for c in ["city","Place","place"]:
        if c in df.columns: df.rename(columns={c:"City"}, inplace=True); break
if "Country" not in df.columns:
    for c in ["country"]:
        if c in df.columns: df.rename(columns={c:"Country"}, inplace=True); break

def z(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0: return pd.Series(0.0, index=s.index)
    s = s.fillna(s.median()); std = s.std()
    return (s - s.mean()) / (std + 1e-9) if np.isfinite(std) and std != 0 else pd.Series(0.0, index=s.index)

# ---- Equal-weight static suitability score
parts = []
if "monthly_cost_usd" in df.columns: parts.append(z(-df["monthly_cost_usd"]))
temp_col = next((c for c in ["weather_avg_temp_c","climate_avg_temp_c"] if c in df.columns), None)
if temp_col:
    temp_dev = (pd.to_numeric(df[temp_col], errors="coerce") - 22.0).abs()
    parts.append(z(-temp_dev))
if "safety_score" in df.columns: parts.append(z(df["safety_score"]))
df["suitability_score"] = pd.concat(parts, axis=1).mean(axis=1).fillna(0.0) if parts else 0.0

# ---- Train model
num_feats = df.select_dtypes(include=[np.number]).columns.tolist()
num_feats = [c for c in num_feats if c != "suitability_score"]
cat_feats = [c for c in df.columns if c not in num_feats + ["suitability_score"]]

pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_feats),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
])
model = Pipeline([("pre", pre),
                  ("nn", MLPRegressor(hidden_layer_sizes=(96,48,16), activation="relu",
                                      early_stopping=True, n_iter_no_change=20,
                                      random_state=42, max_iter=800))])

X = df[num_feats + cat_feats]
y = df["suitability_score"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_tr, y_tr)

# ---- Predict & format JSON
df["predicted_score"] = model.predict(X)
# Penalize cities with unknown/missing country
unknown_mask = (
    df["Country"].isna()
    | df["Country"].astype(str).str.strip().eq("")
    | df["Country"].astype(str).str.contains(r"^unknown\b", case=False, na=True)
)

UNKNOWN_TOKENS = {"unknown country", "unknown", "n/a", "na", "none", "null", ""}
mask_unknown = (
    df["Country"].isna()
    | df["Country"].astype(str).str.strip().str.lower().isin(UNKNOWN_TOKENS)
    | df["Country"].astype(str).str.contains(r"^\s*unknown", case=False, na=True)
)

# Apply very large negative penalty so they sink
df.loc[mask_unknown, "predicted_score"] = df.loc[mask_unknown, "predicted_score"] - 1e9

top3_static_rows = df.sort_values("predicted_score", ascending=False).head(3)

# Now sort by penalized score, build JSON, and print Top 3
import uuid, json
from datetime import datetime

def city_to_json(row):
    return {
        "id": uuid.uuid4().hex,
        "slug": str(row["City"]).strip().lower().replace(" ", "-") if "City" in row else None,
        "name": row.get("City"),
        "country": row.get("Country", "Unknown Country"),
        "description": row.get("description"),
        "monthly_cost_usd": str(round(row.get("monthly_cost_usd", 0), 2)) if "monthly_cost_usd" in row else None,
        "avg_pay_rate_usd_hour": str(round(row.get("avg_pay_rate_usd_hour", 0), 2)) if "avg_pay_rate_usd_hour" in row else None,
        "weather_avg_temp_c": str(row.get("weather_avg_temp_c")) if "weather_avg_temp_c" in row else None,
        "safety_score": str(row.get("safety_score")) if "safety_score" in row else None,
        "nightlife_rating": str(row.get("nightlife_rating")) if "nightlife_rating" in row else None,
        "transport_rating": str(row.get("transport_rating")) if "transport_rating" in row else None,
        "housing_studio_usd_month": str(row.get("housing_studio_usd_month")) if "housing_studio_usd_month" in row else None,
        "housing_one_bed_usd_month": str(row.get("housing_one_bed_usd_month")) if "housing_one_bed_usd_month" in row else None,
        "housing_coliving_usd_month": str(row.get("housing_coliving_usd_month")) if "housing_coliving_usd_month" in row else None,
        "climate_avg_temp_c": str(row.get("climate_avg_temp_c")) if "climate_avg_temp_c" in row else None,
        "climate_summary": row.get("climate_summary"),
        "internet_speed": (f'{row.get("internet_speed")} Mbps' if "internet_speed" in row else None),
        "cost_pct_rent": str(row.get("cost_pct_rent")) if "cost_pct_rent" in row else None,
        "cost_pct_dining": str(row.get("cost_pct_dining")) if "cost_pct_dining" in row else None,
        "cost_pct_transport": str(row.get("cost_pct_transport")) if "cost_pct_transport" in row else None,
        "cost_pct_groceries": str(row.get("cost_pct_groceries")) if "cost_pct_groceries" in row else None,
        "cost_pct_coworking": str(row.get("cost_pct_coworking")) if "cost_pct_coworking" in row else None,
        "cost_pct_other": str(row.get("cost_pct_other")) if "cost_pct_other" in row else None,
        "travel_flight_from_usd": str(row.get("travel_flight_from_usd")) if "travel_flight_from_usd" in row else None,
        "travel_local_transport_usd_week": str(row.get("travel_local_transport_usd_week")) if "travel_local_transport_usd_week" in row else None,
        "travel_hotel_usd_week": str(row.get("travel_hotel_usd_week")) if "travel_hotel_usd_week" in row else None,
        "lifestyle_tags": row.get("lifestyle_tags", []),
        "currency": row.get("currency", "USD"),
        "last_updated": datetime.utcnow().isoformat()
    }

# Top 3 cities (static, penalized)
top3_static_rows = df.sort_values("predicted_score", ascending=False).head(3)
top3_static_json = [city_to_json(r) for _, r in top3_static_rows.iterrows()]

print(json.dumps(top3_static_json, indent=4))