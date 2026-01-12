import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import datetime
import warnings
import joblib
from math import radians, sin, cos, sqrt, atan2

# --- FIX 3: SILENCE WARNINGS ---
# This stops the "X does not have valid feature names" clutter
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. DATA LOADING & PREPARATION
# ==========================================

df_violations = pd.read_csv('Violations.csv.csv')
df_hazards = pd.read_csv('Hazardous.csv')

# --- CLEANING ---
df_violations = df_violations[df_violations['CLUSTER_ID'] != -1].copy()
df_hazards = df_hazards[df_hazards['CLUSTER_ID'] != -1].copy()

# ==========================================
# 2. MODEL TRAINING
# ==========================================

# --- Handle Violation Outliers ---
vio_cap = df_violations['Risk_Index'].quantile(0.90)
print(f"Clipping Violation Risk at: {vio_cap}")

df_violations['Risk_Index_Clipped'] = df_violations['Risk_Index'].clip(upper=vio_cap)
df_violations['Log_Risk'] = np.log1p(df_violations['Risk_Index_Clipped'])

# --- Handle Hazard Outliers ---
haz_cap = df_hazards['Risk_Index'].quantile(0.90)
print(f"Clipping Hazard Risk at: {haz_cap}")

df_hazards['Risk_Index_Clipped'] = df_hazards['Risk_Index'].clip(upper=haz_cap)
df_hazards['Log_Risk'] = np.log1p(df_hazards['Risk_Index_Clipped'])

# --- FIT SCALERS ---
scaler_vio = MinMaxScaler().fit(df_violations[['Log_Risk']])
scaler_haz = MinMaxScaler().fit(df_hazards[['Log_Risk']])

# --- MODEL A: VIOLATIONS (KNN on XCoord/YCoord = Long/Lat, same as hazards) ---
X_vio = df_violations[['XCoord', 'YCoord']]
y_vio = df_violations['Log_Risk'] 

print("Training Violation Model (KNN)...")
model_vio = KNeighborsRegressor(n_neighbors=3)
model_vio.fit(X_vio, y_vio)

# --- MODEL B: HAZARDS (KNN on XCoord/YCoord = Long/Lat) ---
X_haz = df_hazards[['XCoord', 'YCoord']]
y_haz = df_hazards['Log_Risk']

print("Training Hazard Model (KNN)...")
model_haz = KNeighborsRegressor(n_neighbors=1)
model_haz.fit(X_haz, y_haz)

# ==========================================
# 2.5 SAVE MODELS FOR ARCGIS PRO
# ==========================================
print("\nSaving models for ArcGIS Pro...")

# Save the trained models
joblib.dump(model_vio, 'model_violation_knn.joblib')
joblib.dump(model_haz, 'model_hazard_knn.joblib')

# Save the scalers (needed for inference)
joblib.dump(scaler_vio, 'scaler_violation.joblib')
joblib.dump(scaler_haz, 'scaler_hazard.joblib')

print("Models saved:")
print("  - model_violation_knn.joblib (KNN for violations)")
print("  - model_hazard_knn.joblib (KNN for hazards)")
print("  - scaler_violation.joblib")
print("  - scaler_hazard.joblib")

# ==========================================
# 3. THE LOGIC ENGINE
# ==========================================

def haversine_meters(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/long points in meters."""
    R = 6371000  # Earth's radius in meters
    
    lat1_r, lat2_r = radians(lat1), radians(lat2)
    lon1_r, lon2_r = radians(lon1), radians(lon2)
    
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    
    a = sin(dlat/2)**2 + cos(lat1_r) * cos(lat2_r) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def get_safe_location(current_lat, current_long, danger_lat, danger_long):
    """Calculate safe point and distance to reach safety (2km from danger)."""
    SAFE_RADIUS_M = 1500  # 2km safety radius
    
    # Current distance from danger
    current_dist_from_danger = haversine_meters(current_lat, current_long, danger_lat, danger_long)
    
    # How far to travel to reach safety?
    dist_to_safety = max(0, SAFE_RADIUS_M - current_dist_from_danger)
    
    # Vector from danger to current position
    delta_lat = current_lat - danger_lat
    delta_long = current_long - danger_long
    
    # Normalize and scale (approx: 1 deg lat ≈ 111km, 1 deg long ≈ 85km)
    lat_km = 111.0
    long_km = 85.0
    
    dist_km = sqrt((delta_lat * lat_km)**2 + (delta_long * long_km)**2)
    
    if dist_km < 0.001:  # On top of danger - move north
        safe_lat = danger_lat + (SAFE_RADIUS_M / 1000) / lat_km
        safe_long = danger_long
    else:
        # Safe point is 1km from danger, in direction of current position
        scale = (SAFE_RADIUS_M / 1000) / dist_km
        safe_lat = danger_lat + delta_lat * scale
        safe_long = danger_long + delta_long * scale
    
    return safe_lat, safe_long, dist_to_safety

def analyze_new_instance(lat, long, time_obj):
    # --- PREDICTION 1: VIOLATIONS (KNN on Long/Lat - same as hazards) ---
    vio_dist, vio_indices = model_vio.kneighbors([[long, lat]])
    
    # Get nearest violation coordinates for safe location calculation
    nearest_vio_idx = vio_indices[0][0]
    danger_long = df_violations.iloc[nearest_vio_idx]['XCoord']
    danger_lat = df_violations.iloc[nearest_vio_idx]['YCoord']
    
    # Distance Check (0.01 deg ≈ 1km)
    if vio_dist[0][0] > 0.01:
        prob_vio = 0.0
    else:
        raw_vio_risk = model_vio.predict([[long, lat]])
        prob_vio = scaler_vio.transform(raw_vio_risk.reshape(-1,1))[0][0]
    
    # --- PREDICTION 2: HAZARDS (KNN on Long/Lat) ---
    haz_dist, _ = model_haz.kneighbors([[long, lat]])
    
    # Distance Check (0.01 deg ≈ 1km)
    if haz_dist[0][0] > 0.01:
        prob_haz = 0.0
    else:
        raw_haz_risk = model_haz.predict([[long, lat]])
        prob_haz = scaler_haz.transform(raw_haz_risk.reshape(-1,1))[0][0]
    
    # --- FIND SAFE SPOT (1km away from danger) ---
    safe_lat, safe_long, safe_dist = get_safe_location(lat, long, danger_lat, danger_long)
    
    # --- GENERATE MESSAGE ---
    message = []
    status_label = "SAFE"
    
    # 1. Violations
    if prob_vio > 0.6:
        status_label = "WARNING"
        message.append(f"High violation risk ({prob_vio:.2f}).")
    elif prob_vio > 0.3:
        status_label = "CAUTION"
        message.append("Moderate violation risk.")
        
    # 2. Hazards (OVERWRITE status because physical danger > ticket)
    if prob_haz > 0.3:
        status_label = "DANGER"  # Always overwrite to DANGER
        message.append("CRITICAL: Physical road hazard nearby!")
    
    if not message:
        message.append("Route appears clear. No significant hazards.")
        
    final_msg = " ".join(message)
    
    # --- BUILD REPORT ---
    report = {
        "Status": status_label,
        "Message": final_msg,
        "Violation_Probability": f"{prob_vio*100:.1f}%",
        "Hazard_Probability": f"{prob_haz*100:.1f}%",
        "Nearest_Safe_Location": {
            "Lat": safe_lat,
            "Long": safe_long,
            "Distance_m": safe_dist,
            "Distance": f"{safe_dist:.0f} m" if safe_dist < 1000 else f"{safe_dist/1000:.2f} km"
        }
    }
    return report

# ==========================================
# 4. TEST SUITE
# ==========================================
print("\n" + "="*40)
print("       RUNNING FINAL TEST SUITE")
print("="*40)

test_cases = [
    {
        "name": "SCENARIO 1: The 'Red Zone' (High Violation Risk)",
        "lat": 39.1155, "long": -77.1658,
        "time": datetime.datetime(2025, 12, 12, 18, 30, 0)
    },
    {
        "name": "SCENARIO 2: The 'Physical Hazard' (NJ Hotspot)",
        "lat": 40.464, "long": -74.403, 
        "time": datetime.datetime(2025, 12, 12, 10, 0, 0)
    },
    {
        "name": "SCENARIO 3: The 'Quiet Night' (Low Risk)",
        "lat": 39.1618, "long": -77.2536,
        "time": datetime.datetime(2025, 12, 12, 23, 15, 0)
    }
]

for test in test_cases:
    print(f"\n--- {test['name']} ---")
    result = analyze_new_instance(test['lat'], test['long'], test['time'])
    print(f"STATUS: {result['Status']}")
    print(f"MSG:    {result['Message']}")
    print(f"SCORES: Vio: {result['Violation_Probability']} | Haz: {result['Hazard_Probability']}")
    loc = result['Nearest_Safe_Location']
    print(f"SAFE SPOT: Lat {loc['Lat']:.4f}, Long {loc['Long']:.4f} ({loc['Distance']} away)")
    print("-" * 20)