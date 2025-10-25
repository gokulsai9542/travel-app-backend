import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class UnifiedTravelRecommendationModel:
    """
    Comprehensive ML model for travel recommendations combining:
    - Visit probability prediction
    - Duration estimation
    - Personalized ratings based on user preferences
    - Contextual recommendations (time, season, user type)
    """
    
    def __init__(self):
        self.category_encoder = LabelEncoder()
        self.preference_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Three models for different predictions
        self.preference_model = None      # Will user visit? (Classification)
        self.duration_model = None        # How long will they stay? (Regression)
        self.rating_model = None          # Personalized rating (Regression)
        
        self.features = []
        
    def load_all_city_data(self, data_dir='data'):
        """Load data from all city JSON files"""
        all_places = []
        
        if not os.path.exists(data_dir):
            print(f"⚠️  Warning: Directory '{data_dir}' not found")
            return all_places
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        city = data.get('city', filename.replace('.json', ''))
                        
                        for place in data.get('places', []):
                            # Support both lat/lng and latitude/longitude
                            lat = place.get('lat') or place.get('latitude')
                            lng = place.get('lng') or place.get('longitude')
                            
                            if lat and lng:
                                place['city'] = city
                                place['lat'] = lat
                                place['lng'] = lng
                                all_places.append(place)
                                
                    print(f"✓ Loaded {len(data.get('places', []))} places from {filename}")
                except Exception as e:
                    print(f"✗ Error loading {filename}: {e}")
        
        print(f"\nTotal places loaded: {len(all_places)}")
        return all_places
    
    def extract_features(self, place, user_context=None):
        """Extract comprehensive features from a place"""
        features = {}
        
        # Basic features
        features['rating'] = place.get('rating', 4.0)
        features['category'] = place.get('category') or place.get('type', 'attraction')
        
        # Time-based features
        opening = place.get('opening_hours', '09:00')
        closing = place.get('closing_hours', '18:00')
        
        try:
            open_hour = int(opening.split(':')[0])
            close_hour = int(closing.split(':')[0])
            features['opening_hour'] = open_hour
            features['closing_hour'] = close_hour
            features['available_hours'] = close_hour - open_hour
        except:
            features['opening_hour'] = 9
            features['closing_hour'] = 18
            features['available_hours'] = 9
        
        # Popularity indicator
        features['is_popular'] = 1 if features['rating'] >= 4.3 else 0
        
        # Category-based scoring
        category_scores = {
            'attraction': 5,
            'temple': 4,
            'museum': 4,
            'park': 4,
            'beach': 5,
            'food': 3,
            'shopping': 3,
            'hotel': 2
        }
        features['category_score'] = category_scores.get(
            features['category'].lower(), 3
        )
        
        # User context features
        if user_context:
            features['user_available_time'] = user_context.get('available_hours', 4)
            features['time_of_day'] = user_context.get('time_of_day', 12)
            features['user_preference'] = user_context.get('preference', 'Solo')
        else:
            features['user_available_time'] = 4
            features['time_of_day'] = 12
            features['user_preference'] = 'Solo'
        
        return features
    
    def create_training_data(self, places):
        """Create comprehensive training data with multiple scenarios"""
        training_data = []
        
        # User preference types
        preferences = ['Solo', 'Family', 'Couple', 'Friends']
        
        # Time scenarios
        time_scenarios = [
            {'available_hours': 2, 'time_of_day': 10},   # Morning short
            {'available_hours': 4, 'time_of_day': 14},   # Afternoon medium
            {'available_hours': 6, 'time_of_day': 9},    # Full day
            {'available_hours': 3, 'time_of_day': 17},   # Evening
        ]
        
        for place in places:
            for preference in preferences:
                for scenario in time_scenarios:
                    context = {**scenario, 'preference': preference}
                    features = self.extract_features(place, context)
                    
                    # Generate labels
                    will_visit = self._calculate_visit_probability(features)
                    visit_duration = self._estimate_visit_duration(features)
                    personalized_rating = self._calculate_personalized_rating(features)
                    
                    training_data.append({
                        **features,
                        'will_visit': will_visit,
                        'visit_duration': visit_duration,
                        'personalized_rating': personalized_rating
                    })
        
        return pd.DataFrame(training_data)
    
    def _calculate_visit_probability(self, features):
        """Determine if user will visit based on multiple factors"""
        score = 0
        
        # Rating influence
        score += (features['rating'] - 3.0) * 2
        
        # Popularity bonus
        score += features['is_popular'] * 1.5
        
        # Category attractiveness
        score += features['category_score'] * 0.3
        
        # Time compatibility
        available = features.get('user_available_time', 4)
        required = features['available_hours']
        if available >= required * 0.5:  # At least half the time needed
            score += 2
        else:
            score -= 1
        
        # Preference-category match
        preference = features.get('user_preference', 'Solo')
        category = features['category'].lower()
        
        preference_boost = {
            'Solo': ['museum', 'temple', 'park'],
            'Family': ['park', 'beach', 'attraction'],
            'Couple': ['food', 'beach', 'attraction'],
            'Friends': ['food', 'shopping', 'attraction']
        }
        
        if category in preference_boost.get(preference, []):
            score += 1.5
        
        # Convert to binary
        probability = 1 / (1 + np.exp(-score))
        return 1 if probability > 0.6 else 0
    
    def _estimate_visit_duration(self, features):
        """Estimate visit duration based on place type and user preference"""
        # Base durations by category
        base_durations = {
            'attraction': 90,
            'museum': 120,
            'temple': 45,
            'park': 90,
            'beach': 120,
            'food': 60,
            'shopping': 75,
            'hotel': 30
        }
        
        category = features['category'].lower()
        base = base_durations.get(category, 60)
        
        # Rating multiplier (better places = longer visits)
        rating_multiplier = 0.8 + (features['rating'] - 3.0) * 0.15
        
        # Preference multiplier
        preference = features.get('user_preference', 'Solo')
        preference_modifiers = {
            'Solo': 0.85,
            'Family': 1.25,
            'Couple': 1.1,
            'Friends': 1.2
        }
        preference_multiplier = preference_modifiers.get(preference, 1.0)
        
        # Special adjustments by preference-category combo
        if preference == 'Family' and category in ['park', 'beach']:
            preference_multiplier *= 1.2
        elif preference == 'Friends' and category == 'food':
            preference_multiplier *= 1.3
        
        duration = base * rating_multiplier * preference_multiplier
        return int(duration)
    
    def _calculate_personalized_rating(self, features):
        """Calculate personalized rating based on user preferences"""
        base_rating = features['rating']
        category = features['category'].lower()
        preference = features.get('user_preference', 'Solo')
        
        # Preference-category boost
        preference_boost = {
            'Solo': {'attraction': 0.2, 'museum': 0.3, 'temple': 0.25, 'park': 0.15},
            'Family': {'park': 0.3, 'beach': 0.3, 'attraction': 0.25},
            'Couple': {'food': 0.3, 'beach': 0.25, 'attraction': 0.2},
            'Friends': {'food': 0.25, 'shopping': 0.25, 'attraction': 0.2}
        }
        
        boost = preference_boost.get(preference, {}).get(category, 0.1)
        personalized = base_rating + boost + np.random.uniform(-0.05, 0.05)
        
        return min(5.0, max(1.0, personalized))
    
    def prepare_features_for_ml(self, df):
        """Prepare features for ML models"""
        # Encode categorical variables
        df['category_encoded'] = self.category_encoder.fit_transform(df['category'])
        df['preference_encoded'] = self.preference_encoder.fit_transform(df['user_preference'])
        
        # Select features
        feature_cols = [
            'rating', 'opening_hour', 'closing_hour', 'available_hours',
            'is_popular', 'category_score', 'user_available_time',
            'time_of_day', 'category_encoded', 'preference_encoded'
        ]
        
        X = df[feature_cols].copy()
        X_scaled = self.scaler.fit_transform(X)
        
        self.features = feature_cols
        return X_scaled
    
    def train(self, data_dir='data'):
        """Train all models"""
        print("=" * 70)
        print("Training Unified Travel Recommendation Model")
        print("=" * 70)
        
        # Load data
        print("\n1. Loading city data...")
        places = self.load_all_city_data(data_dir)
        
        if len(places) < 5:
            print("⚠️  Warning: Need at least 5 places for training.")
            return False
        
        # Create training data
        print("\n2. Creating comprehensive training dataset...")
        df = self.create_training_data(places)
        print(f"   Generated {len(df)} training samples")
        print(f"   Preferences: {df['user_preference'].unique()}")
        print(f"   Categories: {df['category'].unique()}")
        
        # Prepare features
        print("\n3. Preparing features...")
        X = self.prepare_features_for_ml(df)
        y_visit = df['will_visit'].values
        y_duration = df['visit_duration'].values
        y_rating = df['personalized_rating'].values
        
        # Split data
        X_train, X_test, y_visit_train, y_visit_test = train_test_split(
            X, y_visit, test_size=0.2, random_state=42
        )
        
        # Train Preference Model (Classification)
        print("\n4. Training preference model (visit probability)...")
        self.preference_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.preference_model.fit(X_train, y_visit_train)
        
        y_pred = self.preference_model.predict(X_test)
        accuracy = accuracy_score(y_visit_test, y_pred)
        print(f"   ✓ Accuracy: {accuracy:.2%}")
        
        # Train Duration Model (Regression)
        print("\n5. Training duration model...")
        X_train_dur, X_test_dur, y_dur_train, y_dur_test = train_test_split(
            X, y_duration, test_size=0.2, random_state=42
        )
        
        self.duration_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.duration_model.fit(X_train_dur, y_dur_train)
        
        y_pred_dur = self.duration_model.predict(X_test_dur)
        rmse = np.sqrt(mean_squared_error(y_dur_test, y_pred_dur))
        print(f"   ✓ RMSE: {rmse:.2f} minutes")
        
        # Train Rating Model (Regression)
        print("\n6. Training personalized rating model...")
        X_train_rat, X_test_rat, y_rat_train, y_rat_test = train_test_split(
            X, y_rating, test_size=0.2, random_state=42
        )
        
        self.rating_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.rating_model.fit(X_train_rat, y_rat_train)
        
        y_pred_rat = self.rating_model.predict(X_test_rat)
        rmse_rat = np.sqrt(mean_squared_error(y_rat_test, y_pred_rat))
        print(f"   ✓ RMSE: {rmse_rat:.3f} stars")
        
        # Feature importance
        print("\n7. Top 5 Feature Importances:")
        importances = self.preference_model.feature_importances_
        for feat, imp in sorted(zip(self.features, importances), 
                               key=lambda x: x[1], reverse=True)[:5]:
            print(f"   • {feat}: {imp:.3f}")
        
        print("\n" + "=" * 70)
        print("✓ Training Complete!")
        print("=" * 70)
        return True
    
    def predict(self, place, user_context):
        """Make comprehensive predictions for a place"""
        features = self.extract_features(place, user_context)
        
        # Prepare feature vector
        feature_dict = {k: [v] for k, v in features.items() if k in self.features}
        df_temp = pd.DataFrame(feature_dict)
        
        # Encode categoricals
        if 'category' in features:
            df_temp['category_encoded'] = self.category_encoder.transform([features['category']])
        if 'user_preference' in features:
            df_temp['preference_encoded'] = self.preference_encoder.transform([features['user_preference']])
        
        # Scale features
        X = df_temp[self.features].values
        X_scaled = self.scaler.transform(X)
        
        # Predictions
        visit_prob = self.preference_model.predict_proba(X_scaled)[0][1]
        duration = self.duration_model.predict(X_scaled)[0]
        pers_rating = self.rating_model.predict(X_scaled)[0]
        
        return {
            'visit_probability': float(visit_prob),
            'will_recommend': bool(visit_prob > 0.6),
            'predicted_duration_minutes': int(duration),
            'predicted_duration_hours': round(duration / 60, 1),
            'personalized_rating': round(float(pers_rating), 1),
            'base_rating': features['rating'],
            'rating_boost': round(float(pers_rating) - features['rating'], 2)
        }
    
    def save_model(self, filepath='unified_travel_model.pkl'):
        """Save trained model to disk"""
        model_data = {
            'preference_model': self.preference_model,
            'duration_model': self.duration_model,
            'rating_model': self.rating_model,
            'category_encoder': self.category_encoder,
            'preference_encoder': self.preference_encoder,
            'scaler': self.scaler,
            'features': self.features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        file_size = os.path.getsize(filepath) / 1024
        print(f"\n✓ Model saved to {filepath}")
        print(f"  File size: {file_size:.2f} KB")
    
    def load_model(self, filepath='unified_travel_model.pkl'):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.preference_model = model_data['preference_model']
        self.duration_model = model_data['duration_model']
        self.rating_model = model_data['rating_model']
        self.category_encoder = model_data['category_encoder']
        self.preference_encoder = model_data['preference_encoder']
        self.scaler = model_data['scaler']
        self.features = model_data['features']
        
        print(f"✓ Model loaded from {filepath}")


def main():
    """Main training and testing function"""
    print("\n" + "🚀 " * 25)
    print("   Unified AI Travel Planner - Model Training")
    print("🚀 " * 25 + "\n")
    
    # Initialize model
    model = UnifiedTravelRecommendationModel()
    
    # Train model
    success = model.train(data_dir='data')
    
    if not success:
        print("\n❌ Training failed. Please check your data directory.")
        return
    
    # Save model
    model.save_model('unified_travel_model.pkl')
    
    # Test predictions
    print("\n" + "=" * 70)
    print("Testing Predictions")
    print("=" * 70)
    
    test_place = {
        'name': 'Marina Beach',
        'category': 'beach',
        'rating': 4.5,
        'opening_hours': '06:00',
        'closing_hours': '20:00',
        'lat': 13.05,
        'lng': 80.28
    }
    
    test_contexts = [
        {'available_hours': 3, 'time_of_day': 10, 'preference': 'Solo'},
        {'available_hours': 5, 'time_of_day': 14, 'preference': 'Family'},
        {'available_hours': 2, 'time_of_day': 17, 'preference': 'Couple'},
    ]
    
    for i, context in enumerate(test_contexts, 1):
        print(f"\nTest Case {i}: {context['preference']} traveler")
        prediction = model.predict(test_place, context)
        
        print(f"  Place: {test_place['name']}")
        print(f"  Visit Probability: {prediction['visit_probability']:.1%}")
        print(f"  Will Recommend: {'✓ Yes' if prediction['will_recommend'] else '✗ No'}")
        print(f"  Duration: {prediction['predicted_duration_minutes']} min ({prediction['predicted_duration_hours']} hrs)")
        print(f"  Personalized Rating: {prediction['personalized_rating']}/5.0 "
              f"(base: {prediction['base_rating']}, boost: {prediction['rating_boost']:+.1f})")
    
    print("\n" + "✅ " * 25)
    print("   Training Complete - Ready to Use!")
    print("✅ " * 25 + "\n")


if __name__ == '__main__':
    main()