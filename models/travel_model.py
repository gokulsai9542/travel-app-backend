import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class TravelTimePredictionModel:
    """
    ML Model to predict visit duration and personalized ratings based on:
    - Place category (attraction, food, shopping, hotel)
    - Place rating
    - Travel preference (Solo, Family, Couple, Friends)
    - Place tags and characteristics
    """
    
    def __init__(self):
        self.time_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rating_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        
    def load_json_data(self, json_directory):
        """
        Load all JSON files from directory
        """
        all_places = []
        
        for filename in os.listdir(json_directory):
            if filename.endswith('.json'):
                filepath = os.path.join(json_directory, filename)
                city_name = filename.replace('_tourism.json', '').replace('.json', '')
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    places = json.load(f)
                    
                    for place in places:
                        place['city'] = city_name
                        all_places.append(place)
        
        print(f"✅ Loaded {len(all_places)} places from {len(os.listdir(json_directory))} cities")
        return all_places
        
    def prepare_training_data(self, places_data):
        """
        Prepare training data from JSON files
        Creates synthetic training data based on typical visit patterns
        """
        training_data = []
        preferences = ['Solo', 'Family', 'Couple', 'Friends']
        
        for place in places_data:
            category = place.get('category', 'attraction')
            rating = place.get('rating', 4.0)
            base_time = place.get('average_time', 90)  # Use existing average_time from JSON
            tags = place.get('tags', [])
            
            for preference in preferences:
                # Generate visit duration based on rules
                preference_modifier = self._get_preference_modifier(preference, category, tags)
                visit_duration = base_time * preference_modifier
                
                # Add some randomness to make training more realistic
                visit_duration = visit_duration * np.random.uniform(0.9, 1.1)
                
                # Generate personalized rating
                personalized_rating = self._calculate_personalized_rating(
                    rating, category, preference, tags
                )
                
                training_data.append({
                    'category': category,
                    'base_rating': rating,
                    'base_time': base_time,
                    'preference': preference,
                    'has_heritage_tag': int('Heritage' in tags or 'History' in tags),
                    'has_family_tag': int('Family' in tags),
                    'has_nature_tag': int('Nature' in tags or 'Park' in tags),
                    'has_adventure_tag': int('Adventure' in tags or 'Trekking' in tags),
                    'has_religious_tag': int('Religious' in tags or 'Spiritual' in tags),
                    'visit_duration_minutes': visit_duration,
                    'personalized_rating': personalized_rating
                })
        
        return pd.DataFrame(training_data)
    
    def _get_preference_modifier(self, preference, category, tags):
        """Modify duration based on travel preference and tags"""
        base_modifiers = {
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
        
        modifier = base_modifiers.get(preference, {}).get(category, 1.0)
        
        # Adjust based on tags
        if preference == 'Family':
            if 'Family' in tags or 'Park' in tags:
                modifier *= 1.2
            if 'Museum' in tags or 'Educational' in tags:
                modifier *= 1.15
        
        if preference == 'Solo':
            if 'Heritage' in tags or 'History' in tags:
                modifier *= 1.1
            if 'Museum' in tags:
                modifier *= 1.15
        
        if preference == 'Friends':
            if 'Entertainment' in tags or 'Nightlife' in tags:
                modifier *= 1.3
            if 'Adventure' in tags:
                modifier *= 1.25
        
        if preference == 'Couple':
            if 'Romantic' in tags or 'Scenic' in tags:
                modifier *= 1.2
            if 'Fine Dining' in tags:
                modifier *= 1.15
        
        return modifier
    
    def _calculate_personalized_rating(self, base_rating, category, preference, tags):
        """Calculate personalized rating based on preference and tags"""
        preference_boost = {
            'Solo': {
                'attraction': 0.2,
                'food': 0.1,
                'shopping': 0.15,
                'hotel': 0
            },
            'Family': {
                'attraction': 0.3,
                'food': 0.2,
                'shopping': 0.1,
                'hotel': 0
            },
            'Couple': {
                'attraction': 0.15,
                'food': 0.3,
                'shopping': 0.2,
                'hotel': 0
            },
            'Friends': {
                'attraction': 0.2,
                'food': 0.25,
                'shopping': 0.25,
                'hotel': 0
            }
        }
        
        boost = preference_boost.get(preference, {}).get(category, 0)
        
        # Additional boosts based on tags
        if preference == 'Family' and 'Family' in tags:
            boost += 0.3
        if preference == 'Couple' and ('Romantic' in tags or 'Scenic' in tags):
            boost += 0.25
        if preference == 'Friends' and ('Entertainment' in tags or 'Nightlife' in tags):
            boost += 0.3
        if preference == 'Solo' and ('Heritage' in tags or 'Museum' in tags):
            boost += 0.25
        
        # Add small random variation
        boost += np.random.uniform(-0.05, 0.05)
        
        personalized = base_rating + boost
        return min(5.0, max(1.0, personalized))
    
    def train(self, training_df):
        """Train both time and rating prediction models"""
        # Encode categorical features
        self.label_encoders['category'] = LabelEncoder()
        self.label_encoders['preference'] = LabelEncoder()
        
        training_df['category_encoded'] = self.label_encoders['category'].fit_transform(
            training_df['category']
        )
        training_df['preference_encoded'] = self.label_encoders['preference'].fit_transform(
            training_df['preference']
        )
        
        # Features for prediction
        features = [
            'category_encoded', 
            'preference_encoded', 
            'base_rating',
            'base_time',
            'has_heritage_tag',
            'has_family_tag',
            'has_nature_tag',
            'has_adventure_tag',
            'has_religious_tag'
        ]
        
        X = training_df[features]
        
        # Train time prediction model
        y_time = training_df['visit_duration_minutes']
        X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(
            X, y_time, test_size=0.2, random_state=42
        )
        self.time_model.fit(X_train_time, y_train_time)
        time_score = self.time_model.score(X_test_time, y_test_time)
        
        # Train rating prediction model
        y_rating = training_df['personalized_rating']
        X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(
            X, y_rating, test_size=0.2, random_state=42
        )
        self.rating_model.fit(X_train_rating, y_train_rating)
        rating_score = self.rating_model.score(X_test_rating, y_test_rating)
        
        print("✅ Models trained successfully!")
        print(f"📊 Training samples: {len(training_df)}")
        print(f"🎯 Time model R² score: {time_score:.3f}")
        print(f"⭐ Rating model R² score: {rating_score:.3f}")
        
        return {
            'time_score': time_score,
            'rating_score': rating_score,
            'training_samples': len(training_df)
        }
    
    def predict(self, category, preference, base_rating, base_time=90, tags=None):
        """Predict visit duration and personalized rating"""
        if tags is None:
            tags = []
        
        try:
            category_encoded = self.label_encoders['category'].transform([category])[0]
            preference_encoded = self.label_encoders['preference'].transform([preference])[0]
            
            features = np.array([[
                category_encoded,
                preference_encoded,
                base_rating,
                base_time,
                int('Heritage' in tags or 'History' in tags),
                int('Family' in tags),
                int('Nature' in tags or 'Park' in tags),
                int('Adventure' in tags or 'Trekking' in tags),
                int('Religious' in tags or 'Spiritual' in tags)
            ]])
            
            duration = self.time_model.predict(features)[0]
            rating = self.rating_model.predict(features)[0]
            
            return {
                'visit_duration_minutes': int(round(duration)),
                'visit_duration_hours': round(duration / 60, 1),
                'personalized_rating': round(rating, 1)
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback to simple calculation
            modifier = self._get_preference_modifier(preference, category, tags)
            duration = base_time * modifier
            rating = self._calculate_personalized_rating(base_rating, category, preference, tags)
            
            return {
                'visit_duration_minutes': int(round(duration)),
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
        try:
            model_data = joblib.load(filepath)
            self.time_model = model_data['time_model']
            self.rating_model = model_data['rating_model']
            self.label_encoders = model_data['label_encoders']
            print(f"✅ Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


# Training script
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Training Travel Time Prediction Model")
    print("=" * 60)
    
    # Initialize model
    model = TravelTimePredictionModel()
    
    # Load data from JSON files
    json_directory = '/mnt/user-data/outputs'  # Directory with tourism JSON files
    
    if not os.path.exists(json_directory):
        print(f"❌ Directory not found: {json_directory}")
        print("Please ensure tourism JSON files are in the correct directory")
        exit(1)
    
    # Load all places
    places_data = model.load_json_data(json_directory)
    
    # Prepare training data
    print("\n📊 Preparing training data...")
    training_df = model.prepare_training_data(places_data)
    
    print(f"Generated {len(training_df)} training samples")
    print(f"Categories: {training_df['category'].unique()}")
    print(f"Preferences: {training_df['preference'].unique()}")
    
    # Train models
    print("\n🎓 Training models...")
    scores = model.train(training_df)
    
    # Save model
    print("\n💾 Saving model...")
    model.save_model('travel_model.pkl')
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    print("\nExample predictions:")
    
    test_cases = [
        ('attraction', 'Family', 4.5, 120, ['Heritage', 'Family']),
        ('food', 'Couple', 4.3, 90, ['Fine Dining', 'Romantic']),
        ('shopping', 'Friends', 4.0, 180, ['Mall', 'Entertainment']),
        ('attraction', 'Solo', 4.7, 60, ['Museum', 'History'])
    ]
    
    for category, preference, rating, time, tags in test_cases:
        prediction = model.predict(category, preference, rating, time, tags)
        print(f"\n{preference} at {category} (rating: {rating}, tags: {tags})")
        print(f"  → Duration: {prediction['visit_duration_minutes']} min ({prediction['visit_duration_hours']} hrs)")
        print(f"  → Personalized Rating: {prediction['personalized_rating']}/5.0")
    
    print("\n" + "=" * 60)
    print("✅ Model training complete!")
    print("=" * 60)