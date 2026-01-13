import pandas as pd
import numpy as np
import joblib
import datetime
from math import radians, sin, cos, sqrt, atan2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. LOAD YOUR TRAINED MODELS
# ==========================================
print("Loading models and data...")
model_vio = joblib.load('model_violation_knn.joblib')
model_haz = joblib.load('model_hazard_knn.joblib')
scaler_vio = joblib.load('scaler_violation.joblib')
scaler_haz = joblib.load('scaler_hazard.joblib')

# ==========================================
# 2. CONSTANTS (all distances in METERS for consistency)
# ==========================================
DIST_THRESHOLD_DANGER_M = 1000   # 1km - inside this = DANGER/WARNING
DIST_THRESHOLD_CAUTION_M = 5500  # 5.5km - inside this = CAUTION
SAFE_RADIUS_M = 5500             # 5.5km - matches CAUTION threshold exactly
MAX_VIO_SEVERITY = 1500.0        # 90th percentile cap for violations
MAX_HAZ_SEVERITY = 20.0          # 90th percentile cap for hazards

# Vehicle parameters (hardcoded for now)
SPEED_KMH = 60.0                 # Vehicle speed in km/h
DIRECTION_DEG = 0.0              # Vehicle heading in degrees (0=North, 90=East)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def haversine_meters(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two lat/long points."""
    R = 6371000 
    lat1_r, lat2_r = radians(lat1), radians(lat2)
    lon1_r, lon2_r = radians(lon1), radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = sin(dlat/2)**2 + cos(lat1_r) * cos(lat2_r) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def get_safe_location(current_lat, current_long, danger_lat, danger_long):
    """Calculate safe point and distance to reach safety (1.5km from danger)."""
    current_dist = haversine_meters(current_lat, current_long, danger_lat, danger_long)
    dist_to_safety = max(0, SAFE_RADIUS_M - current_dist)
    
    delta_lat = current_lat - danger_lat
    delta_long = current_long - danger_long
    
    # Approx: 1 deg lat ≈ 111km, 1 deg long ≈ 85km
    lat_km, long_km = 111.0, 85.0
    dist_km = sqrt((delta_lat * lat_km)**2 + (delta_long * long_km)**2)
    
    if dist_km < 0.001:  # On top of danger - move north
        safe_lat = danger_lat + (SAFE_RADIUS_M / 1000) / lat_km
        safe_long = danger_long
    else:
        scale = (SAFE_RADIUS_M / 1000) / dist_km
        safe_lat = danger_lat + delta_lat * scale
        safe_long = danger_long + delta_long * scale
    
    return safe_lat, safe_long, dist_to_safety

def get_nearest_from_knn(model, lat, long):
    """
    Get nearest point info from any KNN model.
    Returns: (dist_deg, dist_m, severity, danger_lat, danger_long)
    """
    dist_deg, indices = model.kneighbors([[long, lat]])
    dist_deg = dist_deg[0][0]
    nearest_idx = indices[0][0]
    
    # Get coordinates (XCoord=Long, YCoord=Lat)
    danger_long = model._fit_X[nearest_idx][0]
    danger_lat = model._fit_X[nearest_idx][1]
    
    # Distance in meters
    dist_m = haversine_meters(lat, long, danger_lat, danger_long)
    
    # Severity from prediction (log scale)
    predicted_log_risk = model.predict([[long, lat]])[0]
    severity = np.expm1(predicted_log_risk)
    
    return dist_deg, dist_m, severity, danger_lat, danger_long

def calculate_risk_status(dist_m, severity, max_severity, labels):
    """
    Calculate status, score, and message based on distance (meters) and severity.
    labels: dict with keys 'danger', 'caution', 'far', 'danger_msg', 'caution_msg'
    Returns: (status, score, message)
    """
    severity_pct = min(100.0, (severity / max_severity) * 100.0)
    
    if dist_m < DIST_THRESHOLD_DANGER_M:
        status = labels['danger']
        score = severity_pct
        msg = labels['danger_msg'].format(score=score)
    elif dist_m < DIST_THRESHOLD_CAUTION_M:
        status = labels['caution']
        distance_factor = 1 - ((dist_m - DIST_THRESHOLD_DANGER_M) / (DIST_THRESHOLD_CAUTION_M - DIST_THRESHOLD_DANGER_M))
        score = severity_pct * distance_factor
        msg = labels['caution_msg'].format(score=score)
    else:
        status = labels['far']
        score = 0.0
        msg = ""
    
    return status, score, msg

# ==========================================
# 4. MAIN ANALYSIS FUNCTION
# ==========================================
def analyze_new_instance(lat, long, time_obj, speed_kmh=SPEED_KMH, direction_deg=DIRECTION_DEG):
    # --- VIOLATIONS ---
    vio_dist_deg, vio_dist_m, vio_severity, vio_danger_lat, vio_danger_long = get_nearest_from_knn(model_vio, lat, long)
    vio_safe_lat, vio_safe_long, vio_safe_dist = get_safe_location(lat, long, vio_danger_lat, vio_danger_long)
    
    vio_labels = {
        'danger': 'WARNING',
        'caution': 'CAUTION', 
        'far': 'FAR',
        'danger_msg': 'High violation risk ({score:.0f}/100)!',
        'caution_msg': 'Approaching violation zone ({score:.0f}/100).'
    }
    vio_status, vio_score, vio_msg = calculate_risk_status(vio_dist_m, vio_severity, MAX_VIO_SEVERITY, vio_labels)
    
    # --- HAZARDS ---
    haz_dist_deg, haz_dist_m, haz_severity, haz_danger_lat, haz_danger_long = get_nearest_from_knn(model_haz, lat, long)
    haz_safe_lat, haz_safe_long, haz_safe_dist = get_safe_location(lat, long, haz_danger_lat, haz_danger_long)
    
    haz_labels = {
        'danger': 'DANGER',
        'caution': 'CAUTION',
        'far': 'FAR',
        'danger_msg': 'CRITICAL: Hazard nearby!',
        'caution_msg': 'Approaching hazard zone.'
    }
    haz_status, haz_score, haz_msg = calculate_risk_status(haz_dist_m, haz_severity, MAX_HAZ_SEVERITY, haz_labels)
    
    # --- SAFE LOCATION (use worst case, or current location if already safe) ---
    if haz_safe_dist > vio_safe_dist:
        safe_lat, safe_long, safe_dist = haz_safe_lat, haz_safe_long, haz_safe_dist
    else:
        safe_lat, safe_long, safe_dist = vio_safe_lat, vio_safe_long, vio_safe_dist
    
    # If already safe (dist=0), use current location instead of pointing to faraway danger zone
    if safe_dist == 0:
        safe_lat, safe_long = lat, long
    
    # --- CALCULATE ETA TO SAFETY ---
    if safe_dist > 0 and speed_kmh > 0:
        eta_minutes = (safe_dist / 1000) / speed_kmh * 60  # Convert to minutes
    else:
        eta_minutes = 0.0
    
    # --- OVERALL STATUS (physical danger > tickets) ---
    status_label = "SAFE"
    message = []
    
    # Violations
    if vio_status == "WARNING":
        status_label = "WARNING"
        message.append(vio_msg)
    elif vio_status == "CAUTION":
        status_label = "CAUTION"
        message.append(vio_msg)
    
    # Hazards (DANGER overrides WARNING)
    if haz_status == "DANGER":
        status_label = "DANGER"
        message.append(haz_msg)
    elif haz_status == "CAUTION":
        if status_label == "SAFE":
            status_label = "CAUTION"
        message.append(haz_msg)
    
    # --- SPEED-AWARE WARNINGS ---
    if haz_status == "DANGER" and speed_kmh > 30:
        message.append(f"SLOW DOWN! Reduce speed to 30 km/h.")
    elif haz_status == "CAUTION" and speed_kmh > 50:
        message.append(f"Reduce speed to 50 km/h.")
    elif vio_status == "WARNING" and speed_kmh > 40:
        message.append(f"High enforcement area - reduce to 40 km/h.")
    
    # Add speed-aware ETA info to message if not safe
    if safe_dist > 0:
        message.append(f"At {speed_kmh:.0f} km/h, safety in {eta_minutes:.1f} min.")
    
    if not message:
        message.append("Route appears clear.")

    return {
        "Status": status_label,
        "Message": " ".join(message),
        "Vio_Status": vio_status,
        "Vio_Score": vio_score,
        "Vio_Severity": vio_severity,
        "Haz_Status": haz_status,
        "Haz_Score": haz_score,
        "Haz_Severity": haz_severity,
        "Safe_Lat": safe_lat,
        "Safe_Long": safe_long,
        "Safe_Dist_m": safe_dist,
        "Speed_kmh": speed_kmh,
        "Direction_deg": direction_deg,
        "ETA_minutes": eta_minutes
    }

# ==========================================
# 3. CHOOSE YOUR PATH
# ==========================================
def create_path(start_lat, start_long, end_lat, end_long, steps=50):
    lats = np.linspace(start_lat, end_lat, steps)
    longs = np.linspace(start_long, end_long, steps)
    return lats, longs

# ========================================
# OPTION 1: MARYLAND PATH (Violation Detection)
# Goes through high-risk violation areas
# ========================================
violation_waypoints = [
    (38.9195596, -77.0252176),
    (38.9328876, -77.0267795),
    (38.9734972, -76.9964544),
    (38.9816924, -76.9889000),  
    (38.9917788, -76.9873021), 
    (39.0027640, -76.9814767), 
    (39.0191925, -76.9767990), 
    (39.0202050, -76.9526073), 
    
]

# ========================================
# OPTION 2: NEW JERSEY PATH (Hazard Detection)  
# Goes through physical hazard zone
# ========================================
hazard_waypoints = [
    (32.014978, -104.491776),
    (32.3903545, -104.2197176),      # Start: Before hazard
    (32.5585287, -104.0786752)     # HAZARD ZONE (Risk: 24.19  # End: After hazard
]

# ========================================
# RUN BOTH PATHS AND EXPORT BOTH CSVs
# ========================================
paths_config = {
    "violation": {
        "waypoints": violation_waypoints,
        "steps_per_segment": 20,
        "description": "violation detection demo"
    },
    "hazard": {
        "waypoints": hazard_waypoints,
        "steps_per_segment": 30,
        "description": "hazard detection demo"
    }
}

for path_name, config in paths_config.items():
    print(f"\n{'='*50}")
    print(f"Running {path_name} path ({config['description']})")
    print('='*50)
    
    waypoints = config['waypoints']
    steps_per_segment = config['steps_per_segment']
    
    # Build full path from waypoints
    full_lats = []
    full_longs = []
    for i in range(len(waypoints) - 1):
        lats, longs = create_path(
            waypoints[i][0], waypoints[i][1],
            waypoints[i+1][0], waypoints[i+1][1],
            steps_per_segment
        )
        if i > 0:  # Skip first point to avoid duplicates
            lats = lats[1:]
            longs = longs[1:]
        full_lats.extend(lats)
        full_longs.extend(longs)
    
    full_lats = np.array(full_lats)
    full_longs = np.array(full_longs)
    
    # Run simulation
    simulated_data = []
    current_time = datetime.datetime(2025, 12, 12, 10, 0, 0)
    
    print(f"Simulating {len(full_lats)} points...")
    
    for i in range(len(full_lats)):
        lat = full_lats[i]
        long = full_longs[i]
        
        report = analyze_new_instance(lat, long, current_time, SPEED_KMH, DIRECTION_DEG)
        
        # Format distance string
        safe_dist = report['Safe_Dist_m']
        if safe_dist < 1000:
            dist_str = f"{safe_dist:.0f} m"
        else:
            dist_str = f"{safe_dist/1000:.2f} km"
        
        # Extract scores
        vio_status = report['Vio_Status']
        vio_score = report['Vio_Score']
        haz_status = report['Haz_Status']
        haz_score = report['Haz_Score']
        eta_min = report['ETA_minutes']
        
        # Build display text with speed and ETA
        display_text = (
            f"STATUS: {report['Status']}\n"
            f"MSG: {report['Message']}\n"
            f"SPEED: {report['Speed_kmh']:.0f} km/h | DIR: {report['Direction_deg']:.0f}°\n"
            f"SCORES: Vio: {vio_status} (Score: {vio_score:.1f}) | Haz: {haz_status} (Score: {haz_score:.1f})\n"
            f"SAFE SPOT: Lat {report['Safe_Lat']:.4f}, Long {report['Safe_Long']:.4f} ({dist_str} away, ETA: {eta_min:.1f} min)"
        )
        
        simulated_data.append({
            'Sequence': i,
            'Timestamp': current_time.isoformat(),
            'Latitude': lat,
            'Longitude': long,
            'Speed_kmh': report['Speed_kmh'],
            'Direction_deg': report['Direction_deg'],
            'Status': report['Status'],
            'Message': report['Message'],
            'Vio_Status': vio_status,
            'Vio_Score': round(vio_score, 1),
            'Vio_Severity': round(report['Vio_Severity'], 1),
            'Haz_Status': haz_status,
            'Haz_Score': round(haz_score, 1),
            'Haz_Severity': round(report['Haz_Severity'], 2),
            'Scores': f"Vio: {vio_status} ({vio_score:.1f}) | Haz: {haz_status} ({haz_score:.1f})",
            'Safe_Lat': report['Safe_Lat'],
            'Safe_Long': report['Safe_Long'],
            'Safe_Dist': dist_str,
            'ETA_minutes': round(eta_min, 1),
            'Display_Text': display_text
        })
        
        # Print progress for key status changes
        if i == 0 or simulated_data[-1]['Status'] != simulated_data[-2]['Status']:
            print(f"  Step {i}: {report['Status']} - Vio:{vio_status} ({vio_score:.0f}) | Haz:{haz_status} ({haz_score:.0f})")
        
        current_time += datetime.timedelta(minutes=1)
    
    # Save to CSV
    df_sim = pd.DataFrame(simulated_data)
    output_file = f'Car_Simulation_{path_name}.csv'
    df_sim.to_csv(output_file, index=False)
    
    print(f"\nSaved: {output_file}")
    print(f"Total points: {len(df_sim)}")
    print(f"Status distribution:")
    print(df_sim['Status'].value_counts().to_string())

print(f"\n{'='*50}")
print("Done! Exported both Car_Simulation_violation.csv and Car_Simulation_hazard.csv")
print('='*50)
