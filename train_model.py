import json
import os
import sys
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
    print("\n🤖 Initializing model...")
    model = TravelTimePredictionModel()
    
    # Prepare training data
    print("\n📈 Preparing training data...")
    training_df = model.prepare_training_data(json_files)
    
    # Save training data for inspection
    training_df.to_csv('training_data.csv', index=False)
    print(f"💾 Training data saved to training_data.csv")
    
    # Train model
    print("\n🎓 Training models...")
    model.train(training_df)
    
    # Test predictions
    print("\n" + "="*60)
    print("🧪 TESTING PREDICTIONS")
    print("="*60)
    
    test_cases = [
        ("attraction", "Solo", 4.6),
        ("attraction", "Family", 4.5),
        ("food", "Couple", 4.4),
        ("shopping", "Friends", 4.2)
    ]
    
    for category, preference, rating in test_cases:
        prediction = model.predict(category, preference, rating)
        print(f"\n{category.upper()} | {preference} | Rating: {rating}")
        print(f"  ⏱️  Duration: {prediction['visit_duration_hours']} hours "
              f"({prediction['visit_duration_minutes']} min)")
        print(f"  ⭐ Personalized: {prediction['personalized_rating']}/5.0")
    
    # Save model
    print("\n" + "="*60)
    print("💾 Saving model...")
    model.save_model('travel_model.pkl')
    
    print("\n✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n📝 Files created:")
    print("  - travel_model.pkl (trained model)")
    print("  - training_data.csv (training data)")
    print("\n🎯 Next steps:")
    print("  1. Start Flask API: python flask_api.py")
    print("  2. Test API: curl http://localhost:5000/api/health")

if __name__ == "__main__":
    main()