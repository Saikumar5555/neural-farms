import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
# from sklearn.model_selection import StratifiedKFold  # Custom CV strategy
from sklearn.model_selection import LeaveOneOut  # Alternative cross-validation strategy
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Data Models
class SoilData(BaseModel):
    ph: float
    nitrogen: float
    phosphorus: float
    potassium: float
    organic_matter: Optional[float] = None
    temperature: float
    humidity: float
    rainfall: float

class CropRecommendationRequest(BaseModel):
    soil_data: SoilData
    region: str
    previous_crops: List[str] = []
    
class CropRecommendation(BaseModel):
    recommended_crop: str
    confidence: float
    alternatives: List[str]
    uniqueness_score: float

# Initialize FastAPI
app = FastAPI(title="ML-Powered Unique Crop Recommendation API")

# Load sample data and train model 
# In production, you would load a pre-trained model
def create_and_train_model():
    # Sample data - in production this would be real training data
    # Format: N, P, K, temperature, humidity, ph, rainfall, crop_label
    data = {
        'N': [90, 85, 60, 74, 78, 20, 45, 30, 85, 60, 40],
        'P': [42, 58, 55, 35, 42, 82, 32, 40, 40, 18, 36],
        'K': [43, 41, 38, 45, 40, 44, 26, 35, 42, 38, 45],
        'temperature': [20.8, 21.8, 23.1, 26.5, 28.0, 25.5, 24.5, 26.2, 28.5, 24.5, 27.2],
        'humidity': [82, 80, 77, 80, 70, 73, 65, 75, 83, 70, 60],
        'ph': [6.5, 7.0, 6.8, 6.4, 7.1, 7.5, 6.3, 6.8, 6.5, 7.0, 7.2],
        'rainfall': [202, 226, 177, 160, 140, 185, 120, 250, 185, 200, 180],
        'crop': ['rice', 'wheat', 'mungbean', 'maize', 'cotton', 'coffee', 'chickpea', 'lentil', 'tomato', 'mustard', 'sunflower']
    }
    
    df = pd.DataFrame(data)
    
    # Features and target
    X = df.drop('crop', axis=1)
    y = df['crop']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define hyperparameter grid for tuning
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    
    # Use LeaveOneOut cross-validation
    cv = LeaveOneOut()  # No arguments needed
    
    # Grid search for hyperparameter optimization
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    return best_model, df

# Create/train model and store crop uniqueness information
model, crop_data = create_and_train_model()

# Calculate crop uniqueness scores based on frequency in dataset
def calculate_uniqueness_scores(df):
    crop_counts = df['crop'].value_counts()
    total_crops = len(df)
    
    # Invert frequency to get uniqueness (rarer crops = higher uniqueness)
    uniqueness_scores = {crop: 1 - (count / total_crops) for crop, count in crop_counts.items()}
    
    return uniqueness_scores

uniqueness_scores = calculate_uniqueness_scores(crop_data)

@app.post("/recommend", response_model=CropRecommendation)
def recommend_crop(request: CropRecommendationRequest):
    try:
        # Extract features from request
        features = [
            request.soil_data.nitrogen,
            request.soil_data.phosphorus,
            request.soil_data.potassium,
            request.soil_data.temperature,
            request.soil_data.humidity,
            request.soil_data.ph,
            request.soil_data.rainfall
        ]
        
        # Reshape for single prediction
        input_features = np.array(features).reshape(1, -1)
        
        # Get probability predictions
        probabilities = model.predict_proba(input_features)[0]
        
        # Get all classes/crops from the model
        classes = model.classes_
        
        # Sort by probability and get top 3
        sorted_indices = np.argsort(probabilities)[::-1]
        top_crop = classes[sorted_indices[0]]
        alternative_crops = [classes[i] for i in sorted_indices[1:4]]  # Next 3 best options
        
        # Check uniqueness score
        uniqueness = uniqueness_scores.get(top_crop, 0.5)  # Default if not in our dataset
        
        # Filter out crops that the user already grew (avoid repetition)
        if top_crop in request.previous_crops and len(alternative_crops) > 0:
            # Recommend the next best crop that hasn't been grown before
            for alt_crop in alternative_crops:
                if alt_crop not in request.previous_crops:
                    # Swap the top crop with this alternative
                    alternative_crops.remove(alt_crop)
                    alternative_crops.insert(0, top_crop)
                    top_crop = alt_crop
                    uniqueness = uniqueness_scores.get(top_crop, 0.5)
                    break
        
        return CropRecommendation(
            recommended_crop=top_crop,
            confidence=float(probabilities[sorted_indices[0]]),
            alternatives=alternative_crops,
            uniqueness_score=uniqueness
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/crops")
def get_available_crops():
    return {"available_crops": list(model.classes_)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
