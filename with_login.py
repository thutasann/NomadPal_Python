import json
import uuid
import math
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
        "id": row.get("id"),  # Use the actual database ID
        "slug": row.get("slug"),  # Use the actual database slug
        "name": row.get("name") or row.get("City"),  # Prefer 'name' column, fallback to 'City'
        "country": row.get("country") or row.get("Country", "Unknown Country"),
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
        "predicted_score": float(row.get("_user_score", 0)),  # Use personalized score
        "ml_enhanced": True,
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

def get_personalized_cities_with_user_preferences(
    user_preferences: Dict,
    limit: int = 20,
    offset: int = 0
):
    """
    Get personalized city recommendations based on user preferences from database
    
    Args:
        user_preferences: Dict containing user preference data:
            - monthly_budget_min_usd: float
            - monthly_budget_max_usd: float
            - preferred_climate: str
            - timezone: str
            - lifestyle_priorities: list
            - monthly_budget_min_usd: float
            - monthly_budget_max_usd: float
        limit: Maximum number of cities to return
        offset: Number of cities to skip (for pagination)
    
    Returns:
        List of ranked cities with personalized scores
    """
    try:
        # Extract user preferences
        budget_min = float(user_preferences.get('monthly_budget_min_usd', 0))
        budget_max = float(user_preferences.get('monthly_budget_max_usd', 10000))
        preferred_climate = user_preferences.get('preferred_climate', 'moderate')
        timezone = user_preferences.get('timezone', 'UTC')
        lifestyle_priorities = user_preferences.get('lifestyle_priorities', [])
        
        # Calculate budget weight based on user's budget range
        if budget_max > 0:
            budget_weight = 0.7  # Higher weight for budget-conscious users
            climate_weight = 0.3
        else:
            budget_weight = 0.5
            climate_weight = 0.5
        
        # Map climate preferences to temperature ranges
        climate_temp_map = {
            'tropical': 28.0,
            'subtropical': 25.0,
            'temperate': 20.0,
            'moderate': 22.0,
            'continental': 18.0,
            'polar': 5.0,
            'arid': 30.0,
            'mediterranean': 24.0
        }
        
        preferred_temp = climate_temp_map.get(preferred_climate.lower(), 22.0)
        
        # Build constraints based on user preferences (make them less restrictive)
        constraints = {}
        if budget_max > 0 and budget_max < 5000:  # Only apply budget constraint if it's reasonable
            constraints['max_monthly_cost_usd'] = budget_max * 1.5  # Allow 50% buffer
        if budget_min > 0:
            constraints['min_monthly_cost_usd'] = budget_min * 0.8  # Allow 20% buffer
        
        # Make climate constraints less restrictive
        constraints['temp_band'] = 15.0  # ±15°C tolerance (much more permissive)
        
        # Get personalized rankings (get all cities first for proper pagination)
        # Use a much larger limit to ensure we get all cities
        ranked = rank_cities_personalized(
            weights={"budget": budget_weight, "climate": climate_weight},
            preferred_temp_c=preferred_temp,
            constraints=constraints,
            top_k=10000  # Get all cities from database
        )
        
        # If we got too few cities due to strict constraints, try without constraints
        if len(ranked) < 100:  # If we have less than 100 cities, constraints are too strict
            print(f"Warning: Only got {len(ranked)} cities with constraints, trying without constraints")
            ranked = rank_cities_personalized(
                weights={"budget": budget_weight, "climate": climate_weight},
                preferred_temp_c=preferred_temp,
                constraints={},  # No constraints
                top_k=10000
            )
            print(f"Now got {len(ranked)} cities without constraints")
        
        # Apply additional lifestyle-based filtering if priorities exist
        if lifestyle_priorities and len(lifestyle_priorities) > 0:
            # Filter cities based on lifestyle priorities
            lifestyle_priority_map = {
                'nightlife': 'nightlife_rating',
                'safety': 'safety_score',
                'transport': 'transport_rating',
                'internet': 'internet_speed',
                'cost': 'monthly_cost_usd'
            }
            
            # Apply lifestyle scoring
            for _, row in ranked.iterrows():
                lifestyle_score = 0
                for priority in lifestyle_priorities:
                    if priority.lower() in lifestyle_priority_map:
                        col_name = lifestyle_priority_map[priority.lower()]
                        if col_name in row and pd.notna(row[col_name]):
                            try:
                                value = float(row[col_name])
                                if col_name == 'monthly_cost_usd':
                                    # Lower cost = higher score
                                    lifestyle_score += (10000 - value) / 100
                                else:
                                    # Higher rating = higher score
                                    lifestyle_score += value
                            except (ValueError, TypeError):
                                pass
                
                # Add lifestyle score to user score
                row['_user_score'] += lifestyle_score * 0.1
            
            # Re-sort by updated score
            ranked = ranked.sort_values('_user_score', ascending=False).reset_index(drop=True)
        
        # Get total count before pagination
        total_count = len(ranked)
        
        # Apply pagination
        start_idx = offset
        end_idx = start_idx + limit
        paginated_ranked = ranked.iloc[start_idx:end_idx]
        
        # Convert to JSON
        cities = [row_to_city_json(row) for _, row in paginated_ranked.iterrows()]
        
        # Calculate pagination metadata correctly
        total_pages = math.ceil(total_count / limit)
        current_page = (offset // limit) + 1
        has_next_page = end_idx < total_count
        has_prev_page = offset > 0
        
        print(f"🔢 Pagination calculation: total={total_count}, limit={limit}, offset={offset}")
        print(f"🔢 Calculated: current_page={current_page}, total_pages={total_pages}, has_next_page={has_next_page}")
        
        result = {
            "success": True,
            "cities": cities,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_next_page": has_next_page,
            "has_prev_page": has_prev_page,
            "current_page": current_page,
            "total_pages": total_pages,
            "user_preferences": {
                "budget_range": f"${budget_min}-${budget_max}",
                "preferred_climate": preferred_climate,
                "preferred_temperature": f"{preferred_temp}°C",
                "lifestyle_priorities": lifestyle_priorities
            }
        }
        
        print(f"✅ Python ML returning: {len(cities)} cities, page {current_page}/{total_pages}")
        return result
        
    except Exception as e:
        print(f"Error in get_personalized_cities_with_user_preferences: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "cities": []
        }

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