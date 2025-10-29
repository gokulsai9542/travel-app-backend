import json
import os
import sys
import numpy as np
import pandas as pd
from models.travel_model import TravelTimePredictionModel

def load_json_files(data_dir='data'):
    """Load all JSON files from data directory"""
    json_files = []
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    json_files.append(data)
                print(f"✅ Loaded {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
    
    return json_files

def main():
    print("="*60)
    print("🚀 TRAVEL TIME PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load JSON files
    print("\n📂 Loading JSON files...")
    json_files = load_json_files('data')
    
    if not json_files:
        print("❌ No JSON files loaded. Exiting.")
        sys.exit(1)
    
    print(f"\n📊 Loaded {len(json_files)} destination files")
    
    # Initialize model
    print("\n🤖 Initializing TravelTimePredictionModel...")
    model = TravelTimePredictionModel()
    
    # Prepare training data
    print("\n📈 Preparing training data...")
    training_df = model.prepare_training_data(json_files)
    
    # Save training data for inspection
    training_df.to_csv('training_data.csv', index=False)
    print(f"💾 Training data saved to training_data.csv ({len(training_df)} records)")
    
    # Display data summary
    print(f"\n📊 Data Summary:")
    print(f"  - Total training samples: {len(training_df)}")
    print(f"  - Categories: {training_df['category'].unique().tolist()}")
    print(f"  - Preferences: {training_df['preference'].unique().tolist()}")
    print(f"  - Avg visit duration: {training_df['visit_duration_minutes'].mean():.1f} minutes")
    print(f"  - Avg personalized rating: {training_df['personalized_rating'].mean():.2f}")
    
    # Train model
    print("\n🎓 Training Random Forest models...")
    model.train(training_df)
    
    # Test predictions
    print("\n" + "="*60)
    print("🧪 TESTING PREDICTIONS")
    print("="*60)
    
    test_cases = [
        {'category': 'attraction', 'preference': 'Solo', 'rating': 4.5},
        {'category': 'food', 'preference': 'Couple', 'rating': 4.2},
        {'category': 'shopping', 'preference': 'Family', 'rating': 3.8},
        {'category': 'attraction', 'preference': 'Friends', 'rating': 4.7}
    ]
    
    for test in test_cases:
        prediction = model.predict(
            test['category'],
            test['preference'],
            test['rating']
        )
        print(f"\n{test['category'].title()} | {test['preference']} | Base Rating: {test['rating']}")
        print(f"  ⏱️  Visit Duration: {prediction['visit_duration_minutes']} minutes ({prediction['visit_duration_hours']} hours)")
        print(f"  ⭐ Personalized Rating: {prediction['personalized_rating']}")
    
    # Save model
    print("\n" + "="*60)
    print("💾 Saving model...")
    model.save_model('travel_model.pkl')
    
    print("\n✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n📝 Files created:")
    print("  - travel_model.pkl (trained model)")
    print("  - training_data.csv (training data)")
    print("\n🎯 Model Features:")
    print("  - Random Forest Regressor (100 estimators)")
    print("  - Visit duration prediction")
    print("  - Personalized rating prediction")
    print("\n💡 Next steps:")
    print("  1. Use model.load_model('travel_model.pkl') to load")
    print("  2. Use model.predict(category, preference, base_rating) for predictions")

if __name__ == "__main__":
    main()