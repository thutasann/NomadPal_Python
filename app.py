import pandas as pd
import numpy as np
import uuid
import json
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store the model and data
model = None
df = None
top_cities_cache = None

def load_data_and_train_model():
    """Load CSV data and train the ML model"""
    global model, df, top_cities_cache
    
    CSV_PATH = "csv/FINALIZED_cities_data.csv"
    df = pd.read_csv(CSV_PATH)
    
    # Standardize column names
    if "City" not in df.columns:
        for c in ["city", "Place", "place"]:
            if c in df.columns: 
                df.rename(columns={c: "City"}, inplace=True)
                break
    if "Country" not in df.columns:
        for c in ["country"]:
            if c in df.columns: 
                df.rename(columns={c: "Country"}, inplace=True)
                break

    def z(s):
        """Z-score normalization"""
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() == 0: 
            return pd.Series(0.0, index=s.index)
        s = s.fillna(s.median())
        std = s.std()
        return (s - s.mean()) / (std + 1e-9) if np.isfinite(std) and std != 0 else pd.Series(0.0, index=s.index)

    # Build static suitability score
    parts = []
    if "monthly_cost_usd" in df.columns: 
        parts.append(z(-df["monthly_cost_usd"]))  # cheaper = better
    
    temp_col = next((c for c in ["weather_avg_temp_c", "climate_avg_temp_c"] if c in df.columns), None)
    if temp_col:
        temp_dev = (pd.to_numeric(df[temp_col], errors="coerce") - 22.0).abs()
        parts.append(z(-temp_dev))  # closer to 22C = better
    
    if "safety_score" in df.columns: 
        parts.append(z(df["safety_score"]))  # safer = better
    
    df["suitability_score"] = pd.concat(parts, axis=1).mean(axis=1).fillna(0.0) if parts else 0.0

    # Prepare features for training
    num_feats = df.select_dtypes(include=[np.number]).columns.tolist()
    num_feats = [c for c in num_feats if c != "suitability_score"]
    cat_feats = [c for c in df.columns if c not in num_feats + ["suitability_score"]]

    # Create preprocessing pipeline
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler())
        ]), num_feats),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")), 
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_feats),
    ])
    
    # Create and train model
    model = Pipeline([
        ("pre", pre),
        ("nn", MLPRegressor(
            hidden_layer_sizes=(96, 48, 16), 
            activation="relu",
            early_stopping=True, 
            n_iter_no_change=20,
            random_state=42, 
            max_iter=800
        ))
    ])

    X = df[num_feats + cat_feats]
    y = df["suitability_score"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Suppress warnings during training
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)

    # Generate predictions and apply penalties
    df["predicted_score"] = model.predict(X)
    
    # Penalize cities with unknown/missing country
    UNKNOWN_TOKENS = {"unknown country", "unknown", "n/a", "na", "none", "null", ""}
    mask_unknown = (
        df["Country"].isna()
        | df["Country"].astype(str).str.strip().str.lower().isin(UNKNOWN_TOKENS)
        | df["Country"].astype(str).str.contains(r"^\s*unknown", case=False, na=True)
    )
    
    # Apply penalty for unknown countries
    df.loc[mask_unknown, "predicted_score"] = df.loc[mask_unknown, "predicted_score"] - 1e9
    
    # Cache top cities
    top_cities_cache = get_top_cities_data(10)
    
    print("Model loaded and trained successfully!")

def city_to_json(row):
    """Convert a city row to JSON format"""
    return {
        "id": uuid.uuid4().hex,
        "slug": str(row["City"]).strip().lower().replace(" ", "-") if "City" in row else None,
        "name": row.get("City"),
        "country": row.get("Country", "Unknown Country"),
        "description": row.get("description") if pd.notna(row.get("description")) else None,
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
        "internet_speed": f'{row.get("internet_speed")} Mbps' if "internet_speed" in row and pd.notna(row.get("internet_speed")) else None,
        "cost_pct_rent": str(row.get("cost_pct_rent")) if "cost_pct_rent" in row else None,
        "cost_pct_dining": str(row.get("cost_pct_dining")) if "cost_pct_dining" in row else None,
        "cost_pct_transport": str(row.get("cost_pct_transport")) if "cost_pct_transport" in row else None,
        "cost_pct_groceries": str(row.get("cost_pct_groceries")) if "cost_pct_groceries" in row else None,
        "cost_pct_coworking": str(row.get("cost_pct_coworking")) if "cost_pct_coworking" in row else None,
        "cost_pct_other": str(row.get("cost_pct_other")) if "cost_pct_other" in row else None,
        "travel_flight_from_usd": str(row.get("travel_flight_from_usd")) if "travel_flight_from_usd" in row else None,
        "travel_local_transport_usd_week": str(row.get("travel_local_transport_usd_week")) if "travel_local_transport_usd_week" in row else None,
        "travel_hotel_usd_week": str(row.get("travel_hotel_usd_week")) if "travel_hotel_usd_week" in row else None,
        "lifestyle_tags": eval(row.get("lifestyle_tags", "[]")) if isinstance(row.get("lifestyle_tags"), str) else row.get("lifestyle_tags", []),
        "currency": row.get("currency", "USD"),
        "predicted_score": round(row.get("predicted_score", 0), 4),
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

def get_top_cities_data(limit=10):
    """Get top N cities based on predicted scores"""
    if df is None:
        return []
    
    top_cities_rows = df.sort_values("predicted_score", ascending=False).head(limit)
    return [city_to_json(row) for _, row in top_cities_rows.iterrows()]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "NomadPal Model Service is running",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

@app.route('/cities/top', methods=['GET'])
def get_top_cities():
    """Get top cities based on ML predictions"""
    try:
        # Get limit parameter from query string (default: 10, max: 50)
        limit = min(int(request.args.get('limit', 10)), 50)
        
        if top_cities_cache is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Return cached data or generate fresh data with custom limit
        if limit <= 10:
            cities = top_cities_cache[:limit]
        else:
            cities = get_top_cities_data(limit)
        
        return jsonify({
            "status": "success",
            "data": cities,
            "total": len(cities),
            "limit": limit
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cities/search', methods=['GET'])
def search_cities():
    """Search cities by name or country"""
    try:
        query = request.args.get('q', '').strip().lower()
        limit = min(int(request.args.get('limit', 10)), 50)
        
        if not query:
            return jsonify({"error": "Query parameter 'q' is required"}), 400
        
        if df is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Filter cities by name or country
        filtered_df = df[
            df["City"].str.lower().str.contains(query, na=False) |
            df["Country"].str.lower().str.contains(query, na=False)
        ].sort_values("predicted_score", ascending=False).head(limit)
        
        cities = [city_to_json(row) for _, row in filtered_df.iterrows()]
        
        return jsonify({
            "status": "success",
            "data": cities,
            "total": len(cities),
            "query": query
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cities/by-country/<country>', methods=['GET'])
def get_cities_by_country(country):
    """Get cities filtered by country"""
    try:
        limit = min(int(request.args.get('limit', 10)), 50)
        
        if df is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Filter cities by country (case insensitive)
        filtered_df = df[
            df["Country"].str.lower().str.contains(country.lower(), na=False)
        ].sort_values("predicted_score", ascending=False).head(limit)
        
        cities = [city_to_json(row) for _, row in filtered_df.iterrows()]
        
        return jsonify({
            "status": "success",
            "data": cities,
            "total": len(cities),
            "country": country
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/countries', methods=['GET'])
def get_countries():
    """Get list of all available countries"""
    try:
        if df is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        countries = df["Country"].dropna().unique().tolist()
        countries = [c for c in countries if c.lower() not in ["unknown", "unknown country", "n/a", "na"]]
        countries.sort()
        
        return jsonify({
            "status": "success",
            "data": countries,
            "total": len(countries)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get basic statistics about the dataset"""
    try:
        if df is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        stats = {
            "total_cities": len(df),
            "total_countries": len(df["Country"].dropna().unique()),
            "avg_monthly_cost": round(df["monthly_cost_usd"].mean(), 2) if "monthly_cost_usd" in df else None,
            "avg_safety_score": round(df["safety_score"].mean(), 2) if "safety_score" in df else None,
            "avg_temperature": round(df["weather_avg_temp_c"].mean(), 2) if "weather_avg_temp_c" in df else None,
        }
        
        return jsonify({
            "status": "success",
            "data": stats
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load data and train model on startup
    load_data_and_train_model()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)