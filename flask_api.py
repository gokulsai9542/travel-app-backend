from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import sys

app = Flask(__name__)
CORS(app)

class TravelMLAPI:
    def __init__(self, model_path='travel_model.pkl'):
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.time_model = model_data['time_model']
                self.rating_model = model_data['rating_model']
                self.label_encoders = model_data['label_encoders']
                print(f"✅ Model loaded successfully from {model_path}")
            else:
                print(f"⚠️  Model file not found. Using fallback predictions.")
                self.time_model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.time_model = None
    
    def predict(self, category, preference, base_rating):
        """Make predictions"""
        if self.time_model is not None:
            try:
                category_encoded = self.label_encoders['category'].transform([category])[0]
                preference_encoded = self.label_encoders['preference'].transform([preference])[0]
                
                features = np.array([[category_encoded, preference_encoded, base_rating]])
                
                duration = self.time_model.predict(features)[0]
                rating = self.rating_model.predict(features)[0]
                
                return {
                    'visit_duration_minutes': int(round(duration)),
                    'visit_duration_hours': round(duration / 60, 1),
                    'personalized_rating': round(rating, 1)
                }
            except Exception as e:
                print(f"Model prediction error: {e}, using fallback")
        
        return self._fallback_prediction(category, preference, base_rating)
    
    def _fallback_prediction(self, category, preference, base_rating):
        """Fallback prediction logic"""
        base_time = self._get_base_time(category)
        modifier = self._get_preference_modifier(preference, category)
        duration = base_time * modifier
        
        personalized_rating = self._calculate_personalized_rating(
            base_rating, category, preference
        )
        
        return {
            'visit_duration_minutes': int(round(duration)),
            'visit_duration_hours': round(duration / 60, 1),
            'personalized_rating': round(personalized_rating, 1)
        }
    
    def _get_base_time(self, category):
        time_mapping = {
            'attraction': 90,
            'food': 60,
            'shopping': 75,
            'hotel': 30
        }
        return time_mapping.get(category, 60)
    
    def _get_preference_modifier(self, preference, category):
        modifiers = {
            'Solo': {'attraction': 0.8, 'food': 0.7, 'shopping': 0.9, 'hotel': 1.0},
            'Family': {'attraction': 1.3, 'food': 1.2, 'shopping': 1.4, 'hotel': 1.0},
            'Couple': {'attraction': 1.1, 'food': 1.3, 'shopping': 1.2, 'hotel': 1.0},
            'Friends': {'attraction': 1.2, 'food': 1.4, 'shopping': 1.3, 'hotel': 1.0}
        }
        return modifiers.get(preference, {}).get(category, 1.0)
    
    def _calculate_personalized_rating(self, base_rating, category, preference):
        preference_boost = {
            'Solo': {'attraction': 0.2, 'food': 0.1, 'shopping': 0.15, 'hotel': 0},
            'Family': {'attraction': 0.3, 'food': 0.2, 'shopping': 0.1, 'hotel': 0},
            'Couple': {'attraction': 0.15, 'food': 0.3, 'shopping': 0.2, 'hotel': 0},
            'Friends': {'attraction': 0.2, 'food': 0.25, 'shopping': 0.25, 'hotel': 0}
        }
        
        boost = preference_boost.get(preference, {}).get(category, 0)
        rating = base_rating + boost
        return min(5.0, max(1.0, rating))

# Initialize API
ml_api = TravelMLAPI()

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'Travel ML Prediction API',
        'version': '1.0',
        'endpoints': {
            'predict': '/api/predict',
            'batch_predict': '/api/batch_predict',
            'generate_itinerary': '/api/generate_itinerary',
            'health': '/api/health'
        }
    })

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': ml_api.time_model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        required_fields = ['category', 'preference', 'base_rating']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        category = data['category']
        preference = data['preference']
        base_rating = float(data['base_rating'])
        
        prediction = ml_api.predict(category, preference, base_rating)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'input': {
                'category': category,
                'preference': preference,
                'base_rating': base_rating
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        
        if 'preference' not in data or 'places' not in data:
            return jsonify({'error': 'Missing preference or places field'}), 400
        
        preference = data['preference']
        places = data['places']
        
        results = []
        for place in places:
            prediction = ml_api.predict(
                place['category'],
                preference,
                float(place['base_rating'])
            )
            
            results.append({
                'name': place.get('name', 'Unknown'),
                'category': place['category'],
                'base_rating': place['base_rating'],
                'prediction': prediction
            })
        
        return jsonify({
            'success': True,
            'preference': preference,
            'results': results,
            'total_places': len(results)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate_itinerary', methods=['POST'])
def generate_itinerary():
    try:
        data = request.get_json()
        
        preference = data.get('preference', 'Solo')
        duration_hours = int(data.get('duration_hours', 8))
        places = data.get('places', [])
        travel_type = data.get('travel_type', 'relaxed')
        
        # Add predictions
        places_with_predictions = []
        for place in places:
            prediction = ml_api.predict(
                place['category'],
                preference,
                float(place['base_rating'])
            )
            
            place['predicted_duration_minutes'] = prediction['visit_duration_minutes']
            place['predicted_duration_hours'] = prediction['visit_duration_hours']
            place['personalized_rating'] = prediction['personalized_rating']
            places_with_predictions.append(place)
        
        # Filter by travel type
        if travel_type == 'adventurous':
            places_with_predictions = [
                p for p in places_with_predictions 
                if p['category'] == 'attraction' and p['base_rating'] >= 4.3
            ]
        elif travel_type == 'foodie':
            places_with_predictions = [
                p for p in places_with_predictions 
                if p['category'] == 'food'
            ]
        elif travel_type == 'shopping':
            places_with_predictions = [
                p for p in places_with_predictions 
                if p['category'] == 'shopping'
            ]
        elif travel_type == 'cultural':
            places_with_predictions = [
                p for p in places_with_predictions 
                if p['category'] == 'attraction'
            ]
        
        # Sort by rating
        places_with_predictions.sort(
            key=lambda x: x['personalized_rating'],
            reverse=True
        )
        
        # Select places
        selected_places = []
        total_minutes = 0
        available_minutes = duration_hours * 60
        
        for place in places_with_predictions:
            duration = place['predicted_duration_minutes']
            if total_minutes + duration <= available_minutes:
                selected_places.append(place)
                total_minutes += duration
        
        return jsonify({
            'success': True,
            'itinerary': {
                'preference': preference,
                'duration_hours': duration_hours,
                'travel_type': travel_type,
                'places': selected_places,
                'total_places': len(selected_places),
                'total_duration_minutes': total_minutes,
                'total_duration_hours': round(total_minutes / 60, 1)
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Travel ML API Server...")
    print("📍 API available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  - POST /api/predict")
    print("  - POST /api/batch_predict")
    print("  - POST /api/generate_itinerary")
    print("  - GET  /api/health")
    
    app.run(host='0.0.0.0', port=5000, debug=True)