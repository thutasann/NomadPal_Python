import json
import uuid
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import pandas as pd

# Import the trained model and data from without_login
import without_login

PREFERRED_TEMP_C_DEFAULT = 22.0
UNKNOWN_PENALTY = 1e9

# -----------------------------
# Helpers
# -----------------------------
def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "City" not in df.columns:
        for c in ["city", "Place", "place"]:
            if c in df.columns:
                df = df.rename(columns={c: "City"})
                break
    if "Country" not in df.columns:
        for c in ["country"]:
            if c in df.columns:
                df = df.rename(columns={c: "Country"})
                break
    return df

def z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    s = s.fillna(s.median())
    std = s.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (s - s.mean()) / (std + 1e-9)

def build_factors(frame: pd.DataFrame, preferred_temp_c: float) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    if "monthly_cost_usd" in frame.columns:
        out["budget"] = z(-frame["monthly_cost_usd"])
    else:
        out["budget"] = pd.Series(0.0, index=frame.index)
    tcol = next((c for c in ["weather_avg_temp_c", "climate_avg_temp_c"] if c in frame.columns), None)
    if tcol:
        dev = (pd.to_numeric(frame[tcol], errors="coerce") - float(preferred_temp_c)).abs()
        out["climate"] = z(-dev)
    else:
        out["climate"] = pd.Series(0.0, index=frame.index)
    return out

def minmax_0_100(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx > mn:
        return ((series - mn) / (mx - mn) * 100).round(2)
    return pd.Series(100.0, index=series.index)

def penalize_unknown_countries(work: pd.DataFrame, score_col: str, penalty: float = UNKNOWN_PENALTY) -> None:
    UNKNOWN_TOKENS = {"unknown country", "unknown", "n/a", "na", "none", "null", ""}
    mask_unknown = (
        work["Country"].isna()
        | work["Country"].astype(str).str.strip().str.lower().isin(UNKNOWN_TOKENS)
        | work["Country"].astype(str).str.contains(r"^\s*unknown", case=False, na=True)
    )
    work.loc[mask_unknown, score_col] = work.loc[mask_unknown, score_col] - penalty

def row_to_city_json(row: pd.Series) -> dict:
    """Convert a DataFrame row to the full structured city JSON - same format as without_login.py"""
    return {
        "id": uuid.uuid4().hex,
        "slug": str(row.get("City", "")).strip().lower().replace(" ", "-") if "City" in row else None,
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
        "internet_speed": str(row.get("internet_speed")) if "internet_speed" in row else None,
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
        "predicted_score": row.get("_user_score", 0),  # Use personalized score
        "last_updated": datetime.utcnow().isoformat()
    }

def rank_cities_personalized(
    weights: Dict[str, float],
    preferred_temp_c: float = PREFERRED_TEMP_C_DEFAULT,
    constraints: Optional[Dict[str, float]] = None,
    top_k: Optional[int] = None,
    unknown_penalty: float = UNKNOWN_PENALTY,
) -> pd.DataFrame:
    """
    Rank cities with personalized preferences using data from without_login.py
    """
    # Use the DataFrame from without_login.py (already loaded with ML predictions)
    df = without_login.df.copy()
    
    F = build_factors(df, preferred_temp_c)
    w = np.array([max(0.0, float(weights.get(k, 0.0))) for k in ["budget", "climate"]], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones(2, dtype=float) / 2.0
    M = pd.concat([F["budget"], F["climate"]], axis=1).values
    work = df.copy()
    work["_user_score"] = (M @ w).astype(float)
    
    # Apply constraints
    constraints = constraints or {}
    sel = pd.Series(True, index=work.index)
    if "max_monthly_cost_usd" in constraints and "monthly_cost_usd" in work.columns:
        sel &= pd.to_numeric(work["monthly_cost_usd"], errors="coerce") <= float(constraints["max_monthly_cost_usd"])
    tcol = next((c for c in ["weather_avg_temp_c", "climate_avg_temp_c"] if c in work.columns), None)
    if tcol and "temp_band" in constraints:
        band = float(constraints["temp_band"])
        temps = pd.to_numeric(work[tcol], errors="coerce")
        sel &= (np.abs(temps - float(preferred_temp_c)) <= band)
    work = work[sel].copy()
    
    # Penalize unknown countries
    if "Country" in work.columns:
        penalize_unknown_countries(work, "_user_score", unknown_penalty)
    
    work = work.sort_values("_user_score", ascending=False).reset_index(drop=True)
    work["user_score_0_100"] = minmax_0_100(work["_user_score"])
    return work if top_k is None else work.head(top_k).copy()

# Alias for backward compatibility
rank_cities_personalized_without_safety = rank_cities_personalized

# ============================
# Personalized Functions (using database data)
# ============================

def get_personalized_cities(
    weights: Dict[str, float] = None,
    preferred_temp_c: float = PREFERRED_TEMP_C_DEFAULT,
    constraints: Optional[Dict[str, float]] = None,
    limit: int = 10
):
    """Get personalized city recommendations using database data"""
    if weights is None:
        weights = {"budget": 0.6, "climate": 0.4}
    
    ranked = rank_cities_personalized(
        weights=weights,
        preferred_temp_c=preferred_temp_c,
        constraints=constraints,
        top_k=limit
    )
    
    return [row_to_city_json(row) for _, row in ranked.iterrows()]

def search_personalized_cities(
    query: str,
    weights: Dict[str, float] = None,
    preferred_temp_c: float = PREFERRED_TEMP_C_DEFAULT,
    constraints: Optional[Dict[str, float]] = None,
    limit: int = 10
):
    """Search cities with personalized ranking"""
    if weights is None:
        weights = {"budget": 0.6, "climate": 0.4}
    
    # Get all personalized rankings first
    ranked = rank_cities_personalized(
        weights=weights,
        preferred_temp_c=preferred_temp_c,
        constraints=constraints
    )
    
    # Filter by search query
    filtered = ranked[
        ranked["City"].str.lower().str.contains(query.lower(), na=False) |
        ranked["Country"].str.lower().str.contains(query.lower(), na=False)
    ].head(limit)
    
    return [row_to_city_json(row) for _, row in filtered.iterrows()]

def get_personalized_cities_by_country(
    country: str,
    weights: Dict[str, float] = None,
    preferred_temp_c: float = PREFERRED_TEMP_C_DEFAULT,
    constraints: Optional[Dict[str, float]] = None,
    limit: int = 10
):
    """Get personalized cities filtered by country"""
    if weights is None:
        weights = {"budget": 0.6, "climate": 0.4}
    
    # Get all personalized rankings first
    ranked = rank_cities_personalized(
        weights=weights,
        preferred_temp_c=preferred_temp_c,
        constraints=constraints
    )
    
    # Filter by country
    filtered = ranked[
        ranked["Country"].str.lower().str.contains(country.lower(), na=False)
    ].head(limit)
    
    return [row_to_city_json(row) for _, row in filtered.iterrows()]

# -----------------------------
# Main
# -----------------------------
def main():
    """Example usage of personalized city ranking with database data"""
    print("🤖 Personalized City Ranking using Database")
    print("=" * 50)
    
    # Example user preferences
    weights = {"budget": 0.6, "climate": 0.4}
    constraints = {"max_monthly_cost_usd": 2000, "temp_band": 6}
    preferred_temp = 22.0
    
    print(f"User Preferences:")
    print(f"  - Budget weight: {weights['budget']}")
    print(f"  - Climate weight: {weights['climate']}")
    print(f"  - Max cost: ${constraints['max_monthly_cost_usd']}")
    print(f"  - Preferred temp: {preferred_temp}°C ±{constraints['temp_band']}°C")
    print()
    
    # Get personalized recommendations
    top_cities = get_personalized_cities(
        weights=weights,
        preferred_temp_c=preferred_temp,
        constraints=constraints,
        limit=5
    )
    
    print("🏆 Top 5 Personalized Recommendations:")
    for i, city in enumerate(top_cities, 1):
        print(f"{i}. {city['name']}, {city['country']} (Score: {city['predicted_score']})")
    
    print("\n📄 Full JSON Output:")
    print(json.dumps(top_cities[:3], indent=2))

if __name__ == "__main__":
    main()