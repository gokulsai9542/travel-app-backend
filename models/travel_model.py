import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class TravelTimePredictionModel:
    """
    ML Model to predict visit duration and personalized ratings based on:
    - Place type (attraction, food, shopping, hotel)
    - Place rating
    - Travel preference (Solo, Family, Couple, Friends)
    - Place category characteristics
    """
    
    def __init__(self):
        self.time_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rating_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        
    def prepare_training_data(self, json_files):
        """
        Prepare training data from JSON files
        Creates synthetic training data based on typical visit patterns
        """
        all_places = []
        
        for file_data in json_files:
            places = file_data.get('places', [])
            for place in places:
                all_places.append(place)
        
        # Create training dataset with synthetic visit duration patterns
        training_data = []
        preferences = ['Solo', 'Family', 'Couple', 'Friends']
        
        for place in all_places:
            category = place.get('category', place.get('type', 'attraction'))
            rating = place.get('rating', 4.0)
            
            for preference in preferences:
                # Generate visit duration based on rules
                base_time = self._calculate_base_time(category)
                preference_modifier = self._get_preference_modifier(preference, category)
                visit_duration = base_time * preference_modifier
                
                # Generate personalized rating
                personalized_rating = self._calculate_personalized_rating(
                    rating, category, preference
                )
                
                training_data.append({
                    'category': category,
                    'base_rating': rating,
                    'preference': preference,
                    'visit_duration_minutes': visit_duration,
                    'personalized_rating': personalized_rating
                })
        
        return pd.DataFrame(training_data)
    
    def _calculate_base_time(self, category):
        """Base visit duration in minutes by category"""
        time_mapping = {
            'attraction': 90,
            'food': 60,
            'shopping': 75,
            'hotel': 30
        }
        return time_mapping.get(category, 60)
    
    def _get_preference_modifier(self, preference, category):
        """Modify duration based on travel preference"""
        modifiers = {
            'Solo': {
                'attraction': 0.8,
                'food': 0.7,
                'shopping': 0.9,
                'hotel': 1.0
            },
            'Family': {
                'attraction': 1.3,
                'food': 1.2,
                'shopping': 1.4,
                'hotel': 1.0
            },
            'Couple': {
                'attraction': 1.1,
                'food': 1.3,
                'shopping': 1.2,
                'hotel': 1.0
            },
            'Friends': {
                'attraction': 1.2,
                'food': 1.4,
                'shopping': 1.3,
                'hotel': 1.0
            }
        }
        return modifiers.get(preference, {}).get(category, 1.0)
    
    def _calculate_personalized_rating(self, base_rating, category, preference):
        """Calculate personalized rating based on preference"""
        preference_boost = {
            'Solo': {'attraction': 0.2, 'food': 0.1, 'shopping': 0.15, 'hotel': 0},
            'Family': {'attraction': 0.3, 'food': 0.2, 'shopping': 0.1, 'hotel': 0},
            'Couple': {'attraction': 0.15, 'food': 0.3, 'shopping': 0.2, 'hotel': 0},
            'Friends': {'attraction': 0.2, 'food': 0.25, 'shopping': 0.25, 'hotel': 0}
        }
        
        boost = preference_boost.get(preference, {}).get(category, 0)
        personalized = base_rating + boost + np.random.uniform(-0.1, 0.1)
        return min(5.0, max(1.0, personalized))
    
    def train(self, training_df):
        """Train both time and rating prediction models"""
        self.label_encoders['category'] = LabelEncoder()
        self.label_encoders['preference'] = LabelEncoder()
        
        training_df['category_encoded'] = self.label_encoders['category'].fit_transform(
            training_df['category']
        )
        training_df['preference_encoded'] = self.label_encoders['preference'].fit_transform(
            training_df['preference']
        )
        
        features = ['category_encoded', 'preference_encoded', 'base_rating']
        X = training_df[features]
        
        y_time = training_df['visit_duration_minutes']
        self.time_model.fit(X, y_time)
        
        y_rating = training_df['personalized_rating']
        self.rating_model.fit(X, y_rating)
        
        print("✅ Models trained successfully!")
        print(f"Training samples: {len(training_df)}")
    
    def predict(self, category, preference, base_rating):
        """Predict visit duration and personalized rating"""
        category_encoded = self.label_encoders['category'].transform([category])[0]
        preference_encoded = self.label_encoders['preference'].transform([preference])[0]
        
        features = np.array([[category_encoded, preference_encoded, base_rating]])
        
        duration = self.time_model.predict(features)[0]
        rating = self.rating_model.predict(features)[0]
        
        return {
            'visit_duration_minutes': round(duration),
            'visit_duration_hours': round(duration / 60, 1),
            'personalized_rating': round(rating, 1)
        }
    
    def save_model(self, filepath='travel_model.pkl'):
        """Save trained model to file"""
        model_data = {
            'time_model': self.time_model,
            'rating_model': self.rating_model,
            'label_encoders': self.label_encoders
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='travel_model.pkl'):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        self.time_model = model_data['time_model']
        self.rating_model = model_data['rating_model']
        self.label_encoders = model_data['label_encoders']
        print(f"✅ Model loaded from {filepath}")