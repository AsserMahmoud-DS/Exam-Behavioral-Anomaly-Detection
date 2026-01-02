from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
import warnings
from fastapi.middleware.cors import CORSMiddleware
import os
warnings.filterwarnings('ignore')

app = FastAPI(title="Cheat Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
contamination = 0.3  # Match your training contamination parameter

def load_model():
    global model
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'demo_gmm_model.pkl')
        
        with open(model_path, 'rb') as f:
            loaded_data = joblib.load(f)
        
        # Check if the loaded data is a dictionary (common when models are saved with metadata)
        if isinstance(loaded_data, dict):
            print("Loaded data is a dictionary. Looking for model...")
            # Try to find the actual model in common key names
            if 'model' in loaded_data:
                model = loaded_data['model']
                print("Found model in 'model' key")
            elif 'classifier' in loaded_data:
                model = loaded_data['classifier']
                print("Found model in 'classifier' key")
            elif 'estimator' in loaded_data:
                model = loaded_data['estimator']
                print("Found model in 'estimator' key")
            else:
                print("Dictionary keys found:", list(loaded_data.keys()))
                # If it's a dictionary but no obvious model key, try the first value that has score_samples method
                for key, value in loaded_data.items():
                    if hasattr(value, 'score_samples'):
                        model = value
                        print(f"Found model in '{key}' key")
                        break
                else:
                    print("No model found in dictionary")
                    model = None
        else:
            # Direct model object
            model = loaded_data
            print("Model loaded directly")
        
        # Verify the model has the required methods for GMM/anomaly detection
        if model is not None and hasattr(model, 'score_samples'):
            print("Model loaded successfully and has score_samples method")
        else:
            print("ERROR: Model does not have score_samples method")
            model = None
            
    except FileNotFoundError:
        print("Model file not found. Please ensure 'cheat_detection_model.pkl' exists.")
        model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None


def safe_agg(series, func):
    """Safely apply aggregation function, handling NaN/empty data"""
    clean_series = series.dropna()
    return func(clean_series) if len(clean_series) > 0 else 0

def preprocess_session_data(df):
    """Preprocess session data with mouse/keyboard event separation"""
    
    # Select available columns
    useful_columns = ['Time (seconds)', 'Event Type', 'X Coordinate', 'Y Coordinate', 'Action', 'Is Cheating']
    available_columns = [col for col in useful_columns if col in df.columns]
    df_clean = df[available_columns].copy()
    
    # Remove session summary overhead
    summary_idx = df_clean[df_clean.iloc[:, 0] == "SESSION SUMMARY"].index
    if len(summary_idx) > 0:
        df_clean = df_clean.iloc[:summary_idx[0]]
    
    # Convert time to numeric and drop invalid rows
    df_clean["Time (seconds)"] = pd.to_numeric(df_clean["Time (seconds)"], errors='coerce')
    df_clean = df_clean.dropna(subset=["Time (seconds)"])
    
    # Categorize events
    mouse_events = ['mousemove', 'click']
    keyboard_events = ['keydown', 'copy', 'paste']
    action_events = ['tab-switch']
    
    df_clean['is_mouse_event'] = df_clean['Event Type'].isin(mouse_events)
    df_clean['is_keyboard_event'] = df_clean['Event Type'].isin(keyboard_events)
    df_clean['is_action_event'] = df_clean['Event Type'].isin(action_events)
    
    # Handle coordinates (forward fill for mouse events)
    if 'X Coordinate' in df_clean.columns and 'Y Coordinate' in df_clean.columns:
        df_clean['X Coordinate'] = pd.to_numeric(df_clean['X Coordinate'], errors='coerce')
        df_clean['Y Coordinate'] = pd.to_numeric(df_clean['Y Coordinate'], errors='coerce')
        df_clean['X Coordinate'] = df_clean['X Coordinate'].ffill()
        df_clean['Y Coordinate'] = df_clean['Y Coordinate'].ffill()
        df_clean = df_clean.dropna(subset=['X Coordinate', 'Y Coordinate'])
    
    # Create action features
    if 'Action' in df_clean.columns:
        df_clean["is_copy"] = ((df_clean["Action"] == "copy") | (df_clean["Event Type"] == "copy")).astype(int)
        df_clean["is_paste"] = ((df_clean["Action"] == "paste") | (df_clean["Event Type"] == "paste")).astype(int)
        df_clean["is_blur"] = (df_clean["Action"] == "blur").astype(int)
        df_clean["is_focus"] = (df_clean["Action"] == "focus").astype(int)
    else:
        for action in ["is_copy", "is_paste", "is_blur", "is_focus"]:
            df_clean[action] = 0
    
    df_clean["is_tab_switch"] = (df_clean["Event Type"] == "tab-switch").astype(int)
    
    # Handle cheating column
    if 'Is Cheating' in df_clean.columns:
        df_clean["Is Cheating"] = df_clean["Is Cheating"].astype(bool)
    else:
        df_clean["Is Cheating"] = False
    
    print(f"Final shape: {df_clean.shape}")
    return df_clean

def extract_mouse_features(mouse_chunk):
    """Extract comprehensive mouse movement features"""
    if len(mouse_chunk) < 2:
        return {f'mouse_{feat}': 0 for feat in [
            'path_length', 'straightness', 'velocity_mean', 'velocity_std', 'velocity_max',
            'acceleration_mean', 'acceleration_std', 'acceleration_max', 'jerk_mean', 'jerk_std',
            'direction_changes', 'click_count', 'idle_time_ratio', 'angular_velocity_mean',
            'angular_velocity_std', 'angular_velocity_min', 'angular_velocity_max',
            'curvature_mean', 'curvature_std', 'curvature_min', 'curvature_max',
            'direction_class', 'sum_of_angles', 'largest_deviation', 'sharp_angles'
        ]}
    
    mouse_chunk = mouse_chunk.sort_values("Time (seconds)").copy()
    
    # Calculate motion derivatives
    mouse_chunk["dx"] = mouse_chunk["X Coordinate"].diff()
    mouse_chunk["dy"] = mouse_chunk["Y Coordinate"].diff()
    mouse_chunk["dt"] = mouse_chunk["Time (seconds)"].diff().replace(0, 1e-6)
    
    # Velocities and accelerations
    mouse_chunk["vx"] = mouse_chunk["dx"] / mouse_chunk["dt"]
    mouse_chunk["vy"] = mouse_chunk["dy"] / mouse_chunk["dt"]
    mouse_chunk["velocity"] = np.sqrt(mouse_chunk["vx"]**2 + mouse_chunk["vy"]**2)
    
    mouse_chunk["ax"] = mouse_chunk["vx"].diff() / mouse_chunk["dt"]
    mouse_chunk["ay"] = mouse_chunk["vy"].diff() / mouse_chunk["dt"]
    mouse_chunk["acceleration"] = np.sqrt(mouse_chunk["ax"]**2 + mouse_chunk["ay"]**2)
    
    mouse_chunk["jerk"] = mouse_chunk["acceleration"].diff() / mouse_chunk["dt"]
    
    # Angular calculations
    mouse_chunk["angle"] = np.arctan2(mouse_chunk["dy"], mouse_chunk["dx"])
    mouse_chunk["angular_velocity"] = mouse_chunk["angle"].diff() / mouse_chunk["dt"]
    
    # Curvature
    velocity_mag_cubed = mouse_chunk["velocity"]**3
    cross_product = mouse_chunk["vx"] * mouse_chunk["ay"] - mouse_chunk["vy"] * mouse_chunk["ax"]
    mouse_chunk["curvature"] = np.abs(cross_product) / velocity_mag_cubed.replace(0, 1e-6)
    
    # Path metrics
    distances = np.sqrt(mouse_chunk["dx"]**2 + mouse_chunk["dy"]**2)
    path_length = distances.sum()
    
    start_x, start_y = mouse_chunk.iloc[0][['X Coordinate', 'Y Coordinate']]
    end_x, end_y = mouse_chunk.iloc[-1][['X Coordinate', 'Y Coordinate']]
    end_to_end_dist = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    
    straightness = end_to_end_dist / path_length if path_length > 0 else 0
    direction_changes = (np.abs(mouse_chunk["angle"].diff()) > np.pi/4).sum()
    click_count = (mouse_chunk["Event Type"] == "click").sum()
    
    # Idle time analysis
    velocity_clean = mouse_chunk["velocity"].dropna()
    if len(velocity_clean) > 0:
        low_velocity_threshold = np.percentile(velocity_clean, 10)
        idle_time = mouse_chunk[mouse_chunk["velocity"] <= low_velocity_threshold]["dt"].sum()
        total_time = mouse_chunk["dt"].sum()
        idle_ratio = idle_time / total_time if total_time > 0 else 0
    else:
        idle_ratio = 0
    
    # Direction classification
    if end_to_end_dist > 0:
        direction_angle = np.arctan2(end_y - start_y, end_x - start_x)
        direction_degrees = (np.degrees(direction_angle) + 360) % 360
        direction_class = min(int(direction_degrees // 45) + 1, 8)
    else:
        direction_class = 0
    
    # Largest deviation from straight line
    if len(mouse_chunk) >= 3 and end_to_end_dist > 0:
        x1, y1, x2, y2 = start_x, start_y, end_x, end_y
        line_length = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        
        deviations = []
        for _, point in mouse_chunk.iterrows():
            x0, y0 = point["X Coordinate"], point["Y Coordinate"]
            deviation = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / line_length
            deviations.append(deviation)
        largest_deviation = max(deviations)
    else:
        largest_deviation = 0
    
    # Sharp angles
    angle_changes = np.abs(mouse_chunk["angle"].diff())
    sharp_angles = (angle_changes > 0.0005).sum()
    sum_of_angles = safe_agg(np.abs(mouse_chunk["angle"].diff()), np.sum)
    
    return {
        'mouse_path_length': path_length,
        'mouse_straightness': straightness,
        'mouse_velocity_mean': safe_agg(mouse_chunk["velocity"], np.mean),
        'mouse_velocity_std': safe_agg(mouse_chunk["velocity"], np.std),
        'mouse_velocity_max': safe_agg(mouse_chunk["velocity"], np.max),
        'mouse_acceleration_mean': safe_agg(mouse_chunk["acceleration"], np.mean),
        'mouse_acceleration_std': safe_agg(mouse_chunk["acceleration"], np.std),
        'mouse_acceleration_max': safe_agg(mouse_chunk["acceleration"], np.max),
        'mouse_jerk_mean': safe_agg(mouse_chunk["jerk"], np.mean),
        'mouse_jerk_std': safe_agg(mouse_chunk["jerk"], np.std),
        'mouse_direction_changes': direction_changes,
        'mouse_click_count': click_count,
        'mouse_idle_time_ratio': idle_ratio,
        'mouse_angular_velocity_mean': safe_agg(mouse_chunk["angular_velocity"], np.mean),
        'mouse_angular_velocity_std': safe_agg(mouse_chunk["angular_velocity"], np.std),
        'mouse_angular_velocity_min': safe_agg(mouse_chunk["angular_velocity"], np.min),
        'mouse_angular_velocity_max': safe_agg(mouse_chunk["angular_velocity"], np.max),
        'mouse_curvature_mean': safe_agg(mouse_chunk["curvature"], np.mean),
        'mouse_curvature_std': safe_agg(mouse_chunk["curvature"], np.std),
        'mouse_curvature_min': safe_agg(mouse_chunk["curvature"], np.min),
        'mouse_curvature_max': safe_agg(mouse_chunk["curvature"], np.max),
        'mouse_direction_class': direction_class,
        'mouse_sum_of_angles': sum_of_angles,
        'mouse_largest_deviation': largest_deviation,
        'mouse_sharp_angles': sharp_angles,
    }

def extract_keyboard_features(keyboard_chunk):
    """Extract keyboard typing behavior features"""
    if len(keyboard_chunk) == 0:
        return {
            'keyboard_typing_rate': 0,
            'keyboard_burst_count': 0,
            'keyboard_pause_count': 0,
        }
    
    keyboard_chunk = keyboard_chunk.sort_values("Time (seconds)")
    
    # Typing rate
    if len(keyboard_chunk) >= 2:
        time_span = keyboard_chunk.iloc[-1]["Time (seconds)"] - keyboard_chunk.iloc[0]["Time (seconds)"]
        typing_rate = len(keyboard_chunk) / time_span if time_span > 0 else 0
        
        # Burst and pause analysis
        time_diffs = keyboard_chunk["Time (seconds)"].diff().dropna()
        burst_events = (time_diffs < 0.2).sum()  # <0.2s = burst
        pause_events = (time_diffs > 1.0).sum()  # >1s = pause
    else:
        typing_rate = burst_events = pause_events = 0
    
    return {
        'keyboard_typing_rate': typing_rate,
        'keyboard_burst_count': burst_events,
        'keyboard_pause_count': pause_events,
    }

def extract_features_from_chunk(chunk, cheating_threshold=0.5):
    """Extract all features from a data chunk"""
    # Determine if chunk is cheating
    if 'Is Cheating' in chunk.columns:
        cheating_ratio = chunk["Is Cheating"].mean()
        chunk_is_cheating = cheating_ratio >= cheating_threshold
    else:
        chunk_is_cheating = False
    
    # Calculate elapsed time
    elapsed_time = max(chunk.iloc[-1]["Time (seconds)"] - chunk.iloc[0]["Time (seconds)"], 1e-6)
    
    # Separate event types
    mouse_chunk = chunk[chunk["is_mouse_event"]].copy()
    keyboard_chunk = chunk[chunk["is_keyboard_event"]].copy()
    
    # Extract features
    mouse_features = extract_mouse_features(mouse_chunk)
    keyboard_features = extract_keyboard_features(keyboard_chunk)
    
    # Basic features
    basic_features = {
        "elapsed_time": elapsed_time,
        "copy_events": chunk["is_copy"].sum(),
        "paste_events": chunk["is_paste"].sum(),
        "blur_events": chunk["is_blur"].sum(),
        "focus_events": chunk["is_focus"].sum(),
        "tab_switch_events": chunk["is_tab_switch"].sum(),
        "is_cheating": chunk_is_cheating
    }
    
    return {**basic_features, **mouse_features, **keyboard_features}

def extract_features_from_session(df_session, chunk_size=10, step_size=5, cheating_threshold=0.7, session_name=""):
    """Extract features from a complete session using sliding windows"""
    print(f"\n=== Processing session: {session_name} ===")
    
    if len(df_session) < chunk_size:
        print(f"Warning: Insufficient data ({len(df_session)} rows, need {chunk_size})")
        return pd.DataFrame()
    
    chunks = []
    for start in range(0, len(df_session) - chunk_size + 1, step_size):
        chunk = df_session.iloc[start:start+chunk_size]
        features = extract_features_from_chunk(chunk, cheating_threshold)
        chunks.append(features)
    
    features_df = pd.DataFrame(chunks)
    if len(features_df) > 0:
        cheating_count = features_df['is_cheating'].sum()
        print(f"Extracted {len(chunks)} chunks, {cheating_count} cheating ({cheating_count/len(chunks)*100:.1f}%)")
    
    return features_df

def clean_features(df, drop_cheating=False):
    """Clean and prepare features for modeling - matches training pipeline"""
    exclude_cols = ['is_cheating', 'session_name', 'chunk_start_time']
    if drop_cheating:
        exclude_cols.remove('is_cheating')
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    
    # Handle infinite values and NaNs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    # Cap extreme outliers
    for col in X.select_dtypes(include=[np.number]).columns:
        q99, q01 = X[col].quantile([0.995, 0.005])
        X[col] = X[col].clip(lower=q01, upper=q99)
    
    return X, feature_cols

# Pydantic models for API
class Event(BaseModel):
    timestamp: float
    event_type: str
    x_coordinate: float = 0
    y_coordinate: float = 0
    action: str = ""

class EventBatch(BaseModel):
    events: List[Event]

class PredictionResponse(BaseModel):
    is_cheating: bool
    chunks_processed: int
    cheating_chunks: int
    confidence_score: float
    anomaly_scores: List[float]

# API endpoints
@app.on_event("startup")
async def startup_event():
    load_model()
    if model is not None:
        print(f"Model type: {type(model)}")

@app.get("/")
async def root():
    return {"message": "Cheat Detection API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": str(type(model)) if model is not None else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_cheating(event_batch: EventBatch):
    """
    Process events and predict cheating using GMM anomaly detection
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Minimum events needed for reliable prediction
    min_events = 30  
    if len(event_batch.events) < min_events:
        raise HTTPException(
            status_code=400, 
            detail=f"Need at least {min_events} events for reliable prediction. Received {len(event_batch.events)}"
        )
    
    try:
        # Convert events to DataFrame
        events_data = []
        for event in event_batch.events:
            events_data.append({
                "Time (seconds)": event.timestamp,
                "Event Type": event.event_type,
                "X Coordinate": event.x_coordinate,
                "Y Coordinate": event.y_coordinate,
                "Action": event.action
            })
        
        df = pd.DataFrame(events_data)
        
        # Preprocess data exactly like training
        df_processed = preprocess_session_data(df)
        
        if len(df_processed) < 10:  # Minimum for meaningful chunks
            raise HTTPException(status_code=400, detail="Insufficient valid events after preprocessing")
        
        # Extract features from chunks (chunk_size=10, stride=5)
        chunks_features = extract_features_from_session(df_processed, chunk_size=10, step_size=5, cheating_threshold=0.7)
        
        if len(chunks_features) == 0:
            raise HTTPException(status_code=400, detail="Could not generate any valid chunks")
        
        # Clean features exactly like training pipeline
        X_features, feature_columns = clean_features(chunks_features, drop_cheating=True)
        
        # Make predictions for each chunk using GMM score_samples
        chunk_predictions = model.score_samples(X_features)
        
        # Convert scores to anomaly detection
        # Lower scores indicate more anomalous (cheating) behavior

        threshold = -230

        is_anomaly = chunk_predictions < threshold
        
        cheating_chunks = int(is_anomaly.sum())
        confidence_score = cheating_chunks / len(chunks_features)
        

        # Overall prediction: if >20% (1 chunk) of chunks are anomalous, classify as cheating
        is_cheating = confidence_score >= 0.2

        
        return PredictionResponse(
            is_cheating=is_cheating,
            chunks_processed=len(chunks_features),
            cheating_chunks=cheating_chunks,
            confidence_score=float(confidence_score),
            anomaly_scores=chunk_predictions.tolist()
        )
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(event_batches: List[EventBatch]):
    """Predict cheating for multiple sessions"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    for i, batch in enumerate(event_batches):
        try:
            prediction = await predict_cheating(batch)
            results.append({
                "session_id": i,
                "prediction": prediction
            })
        except HTTPException as e:
            results.append({
                "session_id": i,
                "error": str(e.detail)
            })
    
    return {"batch_results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
