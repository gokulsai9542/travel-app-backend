import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import os
from pathlib import Path

class TravelTimePredictionModel:
    def __init__(self):
        self.duration_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rating_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_columns = None  # Store feature column names
        
    def load_destinations_data(self, data_dir='data'):
        """Load all destination JSON files"""
        destinations = []
        data_path = Path(data_dir)
        
        print("\n📂 Loading travel destinations data...")
        
        for json_file in data_path.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        destinations.extend(data)
                    else:
                        destinations.append(data)
                print(f"✅ Loaded {json_file.name}")
            except Exception as e:
                print(f"❌ Error loading {json_file.name}: {e}")
        
        print(f"\n✅ Loaded {len(destinations)} travel destinations\n")
        return destinations
    
    def prepare_training_data(self, destinations):
        """Prepare training data from destinations with multiple user preferences"""
        training_data = []
        preferences = ['Solo', 'Couple', 'Family', 'Friends']
        
        for dest in destinations:
            # Base features
            base_rating = dest.get('rating', 4.0)
            entry_fee = dest.get('entry_fee', 0)
            tags = dest.get('tags', [])
            num_tags = len(tags)
            category = dest.get('category', 'attraction').lower()
            
            # Visit duration (use provided or estimate)
            base_duration = dest.get('visit_duration', None)
            if base_duration is None:
                # Estimate based on category
                duration_estimates = {
                    'attraction': 120,
                    'food': 60,
                    'shopping': 180,
                    'hotel': 30
                }
                base_duration = duration_estimates.get(category, 90)
            
            # Create training samples for each preference
            for preference in preferences:
                # Adjust duration based on preference
                duration_multipliers = {
                    'Solo': 0.7,      # Solo travelers spend less time
                    'Couple': 1.0,    # Base duration
                    'Family': 1.5,    # Families need more time
                    'Friends': 1.2    # Friends groups take a bit more time
                }
                
                adjusted_duration = base_duration * duration_multipliers[preference]
                
                # Adjust rating based on preference and category
                rating_adjustments = {
                    ('attraction', 'Family'): 0.2,
                    ('attraction', 'Friends'): 0.1,
                    ('food', 'Couple'): 0.15,
                    ('food', 'Friends'): 0.1,
                    ('shopping', 'Family'): 0.1,
                    ('shopping', 'Friends'): 0.05,
                    ('hotel', 'Couple'): 0.2,
                    ('hotel', 'Family'): 0.15,
                }
                
                rating_adj = rating_adjustments.get((category, preference), 0)
                personalized_rating = min(5.0, base_rating + rating_adj)
                
                training_data.append({
                    'category': category,
                    'preference': preference,
                    'base_rating': base_rating,
                    'entry_fee': entry_fee,
                    'num_tags': num_tags,
                    'visit_duration_minutes': int(adjusted_duration),
                    'personalized_rating': round(personalized_rating, 2)
                })
        
        return pd.DataFrame(training_data)
    
    def create_features(self, df):
        """Create feature matrix with proper column names"""
        # Encode categorical variables
        category_dummies = pd.get_dummies(df['category'], prefix='category')
        preference_dummies = pd.get_dummies(df['preference'], prefix='preference')
        
        # Combine numerical and categorical features
        features = pd.concat([
            df[['base_rating', 'entry_fee', 'num_tags']],
            category_dummies,
            preference_dummies
        ], axis=1)
        
        # Store feature column names for later use
        if self.feature_columns is None:
            self.feature_columns = features.columns.tolist()
        
        return features
    
    def train(self, destinations):
        """Train the prediction models"""
        print("📈 Preparing training data...")
        
        # Prepare training data
        df = self.prepare_training_data(destinations)
        
        # Save training data
        df.to_csv('training_data.csv', index=False)
        print(f"💾 Training data saved to training_data.csv ({len(df)} records)")
        
        # Print data summary
        print(f"\n📊 Data Summary:")
        print(f"  • Total training samples: {len(df)}")
        print(f"  • Categories: {df['category'].unique().tolist()}")
        print(f"  • Preferences: {df['preference'].unique().tolist()}")
        print(f"  • Avg visit duration: {df['visit_duration_minutes'].mean():.1f} minutes")
        print(f"  • Avg personalized rating: {df['personalized_rating'].mean():.2f}")
        print(f"  • Duration range: {df['visit_duration_minutes'].min()} - {df['visit_duration_minutes'].max()} minutes")
        
        # Create features
        X = self.create_features(df)
        y_duration = df['visit_duration_minutes']
        y_rating = df['personalized_rating']
        
        print(f"\n🎓 Training Random Forest models...")
        
        # Train models
        self.duration_model.fit(X, y_duration)
        self.rating_model.fit(X, y_rating)
        
        print("✅ Models trained successfully!")
        
        return df
    
    def predict(self, category, preference, base_rating, entry_fee=0, num_tags=0):
        """
        Predict visit duration and personalized rating using DataFrame with feature names
        """
        # Create feature dictionary with all possible features
        features_dict = {
            'base_rating': base_rating,
            'entry_fee': entry_fee,
            'num_tags': num_tags
        }
        
        # Add category encoding (one-hot)
        for cat in ['attraction', 'food', 'hotel', 'shopping']:
            features_dict[f'category_{cat}'] = 1 if category == cat else 0
        
        # Add preference encoding (one-hot)
        for pref in ['Couple', 'Family', 'Friends', 'Solo']:
            features_dict[f'preference_{pref}'] = 1 if preference == pref else 0
        
        # Create DataFrame with a single row
        features_df = pd.DataFrame([features_dict])
        
        # Ensure columns are in the same order as training
        if self.feature_columns:
            features_df = features_df[self.feature_columns]
        
        # Predict
        duration = self.duration_model.predict(features_df)[0]
        rating = self.rating_model.predict(features_df)[0]
        
        return {
            'visit_duration_minutes': int(duration),
            'visit_duration_hours': round(duration / 60, 1),
            'personalized_rating': round(float(rating), 2)
        }
    
    def save_model(self, filename='travel_model.pkl'):
        """Save trained model to file"""
        model_data = {
            'duration_model': self.duration_model,
            'rating_model': self.rating_model,
            'feature_columns': self.feature_columns
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {filename}")
    
    def load_model(self, filename='travel_model.pkl'):
        """Load trained model from file"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.duration_model = model_data['duration_model']
        self.rating_model = model_data['rating_model']
        self.feature_columns = model_data.get('feature_columns')
        print(f"✅ Model loaded from {filename}")


def main():
    """Main training script"""
    print("=" * 70)
    print("🚀 TRAVEL TIME PREDICTION MODEL TRAINING")
    print("=" * 70)
    
    # Initialize model
    print("\n🤖 Initializing TravelTimePredictionModel...")
    model = TravelTimePredictionModel()
    
    # Load destinations data
    destinations = model.load_destinations_data('data')
    
    if not destinations:
        print("❌ No destination data found! Please check the 'data' directory.")
        return
    
    # Train model
    training_df = model.train(destinations)
    
    # Test predictions
    print("\n" + "=" * 70)
    print("🧪 TESTING PREDICTIONS")
    print("=" * 70)
    
    test_cases = [
        {
            'name': 'TEMPLE VISIT',
            'category': 'attraction',
            'preference': 'Solo',
            'base_rating': 4.5,
            'entry_fee': 0,
            'num_tags': 5
        },
        {
            'name': 'ROMANTIC DINNER',
            'category': 'food',
            'preference': 'Couple',
            'base_rating': 4.2,
            'entry_fee': 50,
            'num_tags': 3
        },
        {
            'name': 'MALL SHOPPING',
            'category': 'shopping',
            'preference': 'Family',
            'base_rating': 3.8,
            'entry_fee': 0,
            'num_tags': 4
        },
        {
            'name': 'LUXURY STAY',
            'category': 'hotel',
            'preference': 'Friends',
            'base_rating': 4.7,
            'entry_fee': 200,
            'num_tags': 6
        },
        {
            'name': 'BEACH VISIT',
            'category': 'attraction',
            'preference': 'Family',
            'base_rating': 4.4,
            'entry_fee': 10,
            'num_tags': 7
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        prediction = model.predict(
            category=test['category'],
            preference=test['preference'],
            base_rating=test['base_rating'],
            entry_fee=test['entry_fee'],
            num_tags=test['num_tags']
        )
        
        print(f"\n{i}. {test['name']}")
        print(f"   Category: {test['category'].title()} | Preference: {test['preference']} | Rating: {test['base_rating']}")
        print(f"   ⏱️  Predicted Duration: {prediction['visit_duration_minutes']} min ({prediction['visit_duration_hours']}h)")
        print(f"   ⭐ Personalized Rating: {prediction['personalized_rating']}/5.0")
    
    # Save model
    print("\n" + "=" * 70)
    print("💾 SAVING MODEL")
    print("=" * 70)
    model.save_model('travel_model.pkl')
    
    # Success message
    print("\n✅ TRAINING COMPLETE!")
    print("=" * 70)
    print("\n📦 Output Files:")
    print("  • travel_model.pkl ........... Trained model (Random Forest)")
    print("  • training_data.csv ......... Training dataset")
    
    print("\n🎯 Model Capabilities:")
    print("  • Predicts optimal visit duration based on user preferences")
    print("  • Calculates personalized ratings for destinations")
    print("  • Handles multiple user segments (Solo, Couple, Family, Friends)")
    print("  • Considers 4 destination categories (attraction, food, shopping, hotel)")
    
    print("\n💡 Usage Example:")
    print("""
    from train_model import TravelTimePredictionModel
    
    model = TravelTimePredictionModel()
    model.load_model('travel_model.pkl')
    
    prediction = model.predict(
        category='attraction',
        preference='Family',
        base_rating=4.5,
        entry_fee=15,
        num_tags=6
    )
    print(f"Duration: {prediction['visit_duration_minutes']} minutes")
    print(f"Rating: {prediction['personalized_rating']}/5.0")
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()