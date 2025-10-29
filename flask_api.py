from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import requests
import joblib
import numpy as np
from datetime import datetime, timedelta
import math

app = Flask(__name__)
CORS(app)

# ==================== API Keys Configuration ====================
# WeatherAPI.com - Free tier: 1M calls/month
WEATHERAPI_KEY = os.environ.get('WEATHERAPI_KEY', 'glIsJVxQEiiCItEgu144dgLrKKWQXIkG')

# TomTom API - For routing and traffic
TOMTOM_API_KEY = os.environ.get('TOMTOM_API_KEY', 'd6114ba0264f8ce0bef0a49497064a36')

# Weather Provider Selection
# Options: 'open-meteo' (no key), 'weatherapi', 'openweathermap'
WEATHER_PROVIDER = 'weatherapi'  # Using WeatherAPI since key is provided

# ==================== ML Model Class ====================
class TravelMLAPI:
    def __init__(self, model_path='travel_model.pkl'):
        self.time_model = None
        self.rating_model = None
        self.label_encoders = None
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
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
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
            'museum': 120,
            'park': 75,
            'temple': 60,
            'food': 60,
            'restaurant': 75,
            'shopping': 75,
            'hotel': 30,
            'historical': 100,
            'nature': 90,
            'adventure': 120,
            'religious': 60,
            'entertainment': 90
        }
        return time_mapping.get(category.lower(), 60)
    
    def _get_preference_modifier(self, preference, category):
        modifiers = {
            'Solo': {
                'attraction': 0.8, 'food': 0.7, 'shopping': 0.9, 
                'historical': 0.9, 'nature': 0.8, 'adventure': 1.0,
                'religious': 0.8, 'entertainment': 0.7
            },
            'Family': {
                'attraction': 1.3, 'food': 1.2, 'shopping': 1.4, 
                'historical': 1.2, 'nature': 1.4, 'adventure': 1.3,
                'religious': 1.2, 'entertainment': 1.5
            },
            'Couple': {
                'attraction': 1.1, 'food': 1.3, 'shopping': 1.2, 
                'historical': 1.0, 'nature': 1.2, 'adventure': 1.1,
                'religious': 0.9, 'entertainment': 1.3
            },
            'Friends': {
                'attraction': 1.2, 'food': 1.4, 'shopping': 1.3, 
                'historical': 1.0, 'nature': 1.1, 'adventure': 1.5,
                'religious': 0.9, 'entertainment': 1.4
            }
        }
        return modifiers.get(preference, {}).get(category.lower(), 1.0)
    
    def _calculate_personalized_rating(self, base_rating, category, preference):
        preference_boost = {
            'Solo': {
                'attraction': 0.2, 'food': 0.1, 'shopping': 0.15,
                'historical': 0.3, 'nature': 0.25, 'adventure': 0.2
            },
            'Family': {
                'attraction': 0.3, 'food': 0.2, 'shopping': 0.1,
                'historical': 0.2, 'nature': 0.35, 'entertainment': 0.3
            },
            'Couple': {
                'attraction': 0.15, 'food': 0.3, 'shopping': 0.2,
                'historical': 0.15, 'nature': 0.25, 'entertainment': 0.25
            },
            'Friends': {
                'attraction': 0.2, 'food': 0.25, 'shopping': 0.25,
                'adventure': 0.35, 'entertainment': 0.3, 'nature': 0.2
            }
        }
        
        boost = preference_boost.get(preference, {}).get(category.lower(), 0)
        rating = base_rating + boost
        return min(5.0, max(1.0, rating))
    
    def rank_places(self, places, preferences):
        """Rank places based on preferences"""
        preference = preferences.get('travel_type', 'Solo')
        preferred_categories = preferences.get('preferred_categories', [])
        budget_per_place = preferences.get('budget_per_place', 1000)
        
        ranked = []
        
        for place in places:
            # Skip if over budget
            if place.get('entry_fee', 0) > budget_per_place:
                continue
            
            prediction = self.predict(
                place.get('category', 'attraction'),
                preference,
                place.get('rating', 4.0)
            )
            
            score = prediction['personalized_rating']
            
            # Boost score for preferred categories
            if place.get('category', '').lower() in [c.lower() for c in preferred_categories]:
                score += 0.5
            
            ranked.append({
                'place': place,
                'score': score,
                'prediction': prediction
            })
        
        ranked.sort(key=lambda x: x['score'], reverse=True)
        return ranked

# Initialize ML API
ml_api = TravelMLAPI()

# ==================== Load City Data ====================
cities_data = {}
data_dir = 'data'

if os.path.exists(data_dir):
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            city_name = filename.replace('.json', '')
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                cities_data[city_name] = json.load(f)
    print(f"📂 Loaded {len(cities_data)} cities")

# ==================== Weather Functions ====================
def get_weather_weatherapi(lat, lon):
    """Get weather from WeatherAPI.com"""
    try:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={lat},{lon}&days=2&aqi=no"
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            print(f"WeatherAPI error: {response.status_code}")
            return get_weather_fallback()
        
        data = response.json()
        
        return {
            'current': {
                'temperature': data['current']['temp_c'],
                'feels_like': data['current']['feelslike_c'],
                'condition': data['current']['condition']['text'],
                'description': data['current']['condition']['text'].lower(),
                'humidity': data['current']['humidity'],
                'wind_speed': data['current']['wind_kph'] / 3.6,
                'visibility': data['current']['vis_km'],
                'pressure': data['current']['pressure_mb']
            },
            'forecast': [
                {
                    'date': hour['time'],
                    'temp': hour['temp_c'],
                    'condition': hour['condition']['text'],
                    'description': hour['condition']['text'].lower()
                }
                for hour in data['forecast']['forecastday'][0]['hour'][:8]
            ]
        }
    except Exception as e:
        print(f"WeatherAPI error: {e}")
        return get_weather_fallback()

def get_weather_open_meteo(lat, lon):
    """Get weather from Open-Meteo (NO API KEY REQUIRED)"""
    try:
        current_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,pressure_msl&timezone=auto"
        forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,weather_code&forecast_days=2&timezone=auto"
        
        current_response = requests.get(current_url, timeout=5)
        forecast_response = requests.get(forecast_url, timeout=5)
        
        current_data = current_response.json()
        forecast_data = forecast_response.json()
        
        weather_codes = {
            0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
            45: "Foggy", 48: "Foggy", 51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
            61: "Rain", 63: "Rain", 65: "Rain", 71: "Snow", 73: "Snow", 75: "Snow",
            80: "Rain Showers", 81: "Rain Showers", 82: "Rain Showers",
            95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm"
        }
        
        current = current_data['current']
        weather_code = current['weather_code']
        
        forecast_list = []
        for i in range(min(8, len(forecast_data['hourly']['time']))):
            forecast_list.append({
                'date': forecast_data['hourly']['time'][i],
                'temp': forecast_data['hourly']['temperature_2m'][i],
                'condition': weather_codes.get(forecast_data['hourly']['weather_code'][i], 'Unknown'),
                'description': weather_codes.get(forecast_data['hourly']['weather_code'][i], 'Unknown').lower()
            })
        
        return {
            'current': {
                'temperature': current['temperature_2m'],
                'feels_like': current['apparent_temperature'],
                'condition': weather_codes.get(weather_code, 'Unknown'),
                'description': weather_codes.get(weather_code, 'Unknown').lower(),
                'humidity': current['relative_humidity_2m'],
                'wind_speed': current['wind_speed_10m'],
                'visibility': 10,
                'pressure': current['pressure_msl']
            },
            'forecast': forecast_list
        }
    except Exception as e:
        print(f"Open-Meteo API error: {e}")
        return get_weather_fallback()

def get_weather_fallback():
    """Fallback weather data"""
    return {
        'current': {
            'temperature': 25,
            'feels_like': 27,
            'condition': 'Clear',
            'description': 'clear sky',
            'humidity': 60,
            'wind_speed': 5,
            'visibility': 10,
            'pressure': 1013
        },
        'forecast': []
    }

# ==================== Route Calculation ====================
def calculate_tomtom_route(origin, destination):
    """Calculate route using TomTom API"""
    try:
        locations = f"{origin['lat']},{origin['lng']}:{destination['lat']},{destination['lng']}"
        route_url = f"https://api.tomtom.com/routing/1/calculateRoute/{locations}/json"
        params = {
            'key': TOMTOM_API_KEY,
            'traffic': 'true',
            'travelMode': 'car'
        }
        
        response = requests.get(route_url, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"TomTom API error: {response.status_code}")
            return calculate_route_fallback(origin, destination)
        
        route_data = response.json()
        
        if 'routes' in route_data and len(route_data['routes']) > 0:
            route_info = route_data['routes'][0]['summary']
            
            return {
                'distance_km': route_info['lengthInMeters'] / 1000,
                'duration_minutes': route_info['travelTimeInSeconds'] / 60,
                'traffic_delay_minutes': route_info.get('trafficDelayInSeconds', 0) / 60,
                'route_geometry': route_data['routes'][0].get('legs', [])[0].get('points', [])
            }
        else:
            return calculate_route_fallback(origin, destination)
    except Exception as e:
        print(f"TomTom API error: {e}")
        return calculate_route_fallback(origin, destination)

def calculate_route_fallback(origin, destination):
    """Fallback route calculation using Haversine formula"""
    lat1, lon1 = origin['lat'], origin['lng']
    lat2, lon2 = destination['lat'], destination['lng']
    
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c
    
    return {
        'distance_km': distance,
        'duration_minutes': distance * 2.5,  # Assume 24 km/h average in city
        'traffic_delay_minutes': distance * 0.3,  # Assume 30% traffic delay
        'route_geometry': []
    }

# ==================== API Endpoints ====================

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'Travel Planner API with ML',
        'version': '2.1',
        'backend_url': 'https://travel-app-backend-xhd2.onrender.com',
        'endpoints': {
            'cities': '/api/cities',
            'places': '/api/places/<city>',
            'weather': '/api/weather/<city>',
            'predict': '/api/predict',
            'batch_predict': '/api/batch_predict',
            'generate_itinerary': '/api/itinerary/generate',
            'budget_estimate': '/api/budget/estimate',
            'search': '/api/search',
            'route': '/api/route/tomtom',
            'health': '/api/health'
        }
    })

@app.route('/health')
@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cities_loaded': len(cities_data),
        'model_loaded': ml_api.time_model is not None,
        'weather_provider': WEATHER_PROVIDER,
        'api_keys': {
            'weatherapi': WEATHERAPI_KEY != 'YOUR_WEATHERAPI_KEY',
            'tomtom': TOMTOM_API_KEY != 'YOUR_TOMTOM_API_KEY'
        }
    })

@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get list of available cities"""
    cities_list = []
    for city_name, city_data in cities_data.items():
        cities_list.append({
            'name': city_name,
            'display_name': city_data.get('name', city_name.title()),
            'latitude': city_data.get('latitude', 0),
            'longitude': city_data.get('longitude', 0),
            'places_count': len(city_data.get('places', []))
        })
    return jsonify({'cities': cities_list})

@app.route('/api/places/<city>', methods=['GET'])
def get_places(city):
    """Get all places for a city"""
    if city not in cities_data:
        return jsonify({'error': 'City not found'}), 404
    
    category = request.args.get('category')
    max_budget = request.args.get('max_budget', type=int)
    indoor_only = request.args.get('indoor', type=bool)
    
    places = cities_data[city].get('places', [])
    
    if category:
        places = [p for p in places if p.get('category') == category]
    if max_budget is not None:
        places = [p for p in places if p.get('entry_fee', 0) <= max_budget]
    if indoor_only:
        places = [p for p in places if p.get('indoor', False)]
    
    return jsonify({
        **cities_data[city],
        'places': places,
        'filtered_count': len(places)
    })

@app.route('/api/weather/<city>', methods=['GET'])
def get_weather(city):
    """Get weather for a city"""
    if city not in cities_data:
        return jsonify({'error': 'City not found'}), 404
    
    city_info = cities_data[city]
    lat = city_info.get('latitude', 0)
    lon = city_info.get('longitude', 0)
    
    if WEATHER_PROVIDER == 'weatherapi':
        return jsonify(get_weather_weatherapi(lat, lon))
    elif WEATHER_PROVIDER == 'open-meteo':
        return jsonify(get_weather_open_meteo(lat, lon))
    else:
        return jsonify(get_weather_fallback())

@app.route('/api/predict', methods=['POST'])
def predict():
    """Single place prediction"""
    try:
        data = request.get_json()
        
        required_fields = ['category', 'preference', 'base_rating']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        prediction = ml_api.predict(
            data['category'],
            data['preference'],
            float(data['base_rating'])
        )
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'input': data
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/itinerary/generate', methods=['POST'])
def generate_itinerary():
    """Generate optimized itinerary"""
    try:
        data = request.json
        city = data.get('city')
        preferences = data.get('preferences', {})
        available_time = data.get('available_time', 480)
        current_location = data.get('current_location')
        
        if city not in cities_data:
            return jsonify({'error': 'City not found'}), 404
        
        city_data = cities_data[city]
        places = city_data.get('places', [])
        
        # Rank places using ML
        ranked_places = ml_api.rank_places(places, preferences)
        
        # Generate itinerary
        itinerary = []
        total_time = 0
        total_cost = 0
        total_distance = 0
        current_pos = current_location
        
        for item in ranked_places:
            place = item['place']
            prediction = item['prediction']
            
            if current_pos:
                route_data = calculate_tomtom_route(current_pos, {
                    'lat': place.get('latitude'),
                    'lng': place.get('longitude')
                })
                travel_time = route_data['duration_minutes']
                distance = route_data['distance_km']
            else:
                travel_time = 0
                distance = 0
            
            visit_duration = prediction['visit_duration_minutes']
            
            if total_time + travel_time + visit_duration <= available_time:
                itinerary.append({
                    'place': place,
                    'travel_time': travel_time,
                    'visit_duration': visit_duration,
                    'arrival_time': total_time + travel_time,
                    'distance_from_previous': distance,
                    'score': item['score'],
                    'personalized_rating': prediction['personalized_rating']
                })
                
                total_time += travel_time + visit_duration
                total_cost += place.get('entry_fee', 0)
                total_distance += distance
                current_pos = {
                    'lat': place.get('latitude'),
                    'lng': place.get('longitude')
                }
            else:
                break
        
        return jsonify({
            'success': True,
            'itinerary': itinerary,
            'summary': {
                'total_places': len(itinerary),
                'total_time_minutes': total_time,
                'total_cost': total_cost,
                'total_distance_km': round(total_distance, 2),
                'time_remaining': available_time - total_time
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/budget/estimate', methods=['POST'])
def estimate_budget():
    """Estimate budget"""
    data = request.json
    city = data.get('city')
    days = data.get('days', 1)
    preferences = data.get('preferences', {})
    people = data.get('people', 1)
    
    if city not in cities_data:
        return jsonify({'error': 'City not found'}), 404
    
    budget_levels = {
        'low': {'accommodation': 1000, 'food': 500, 'transport': 300},
        'mid': {'accommodation': 2500, 'food': 1000, 'transport': 600},
        'high': {'accommodation': 5000, 'food': 2000, 'transport': 1000}
    }
    
    budget_level = preferences.get('budget_level', 'mid')
    rates = budget_levels.get(budget_level, budget_levels['mid'])
    
    city_data = cities_data[city]
    places = city_data.get('places', [])
    max_places_per_day = 5
    attraction_cost = sum(
        place.get('entry_fee', 0) 
        for place in sorted(places, key=lambda x: x.get('popularity', 0), reverse=True)[:max_places_per_day * days]
    )
    
    accommodation_total = rates['accommodation'] * days * people
    food_total = rates['food'] * days * people
    transport_total = rates['transport'] * days * people
    subtotal = accommodation_total + food_total + transport_total + attraction_cost
    miscellaneous = subtotal * 0.1
    total_budget = subtotal + miscellaneous
    
    return jsonify({
        'total_budget': round(total_budget),
        'per_person': round(total_budget / people),
        'per_day': round(total_budget / days),
        'breakdown': {
            'accommodation': accommodation_total,
            'food': food_total,
            'transport': transport_total,
            'attractions': attraction_cost,
            'miscellaneous': round(miscellaneous)
        },
        'budget_level': budget_level,
        'currency': 'INR'
    })

@app.route('/api/search', methods=['GET'])
def search_places():
    """Search places"""
    query = request.args.get('q', '').lower()
    category = request.args.get('category')
    max_budget = request.args.get('max_budget', type=int)
    
    results = []
    
    for city_name, city_data in cities_data.items():
        for place in city_data.get('places', []):
            if query and query not in place.get('name', '').lower() and query not in place.get('description', '').lower():
                continue
            
            if category and place.get('category') != category:
                continue
            if max_budget is not None and place.get('entry_fee', 0) > max_budget:
                continue
            
            results.append({
                **place,
                'city': city_name
            })
    
    return jsonify({
        'results': results[:20],
        'count': len(results)
    })

@app.route('/api/route/tomtom', methods=['POST'])
def calculate_route():
    """Calculate route"""
    data = request.json
    origin = data.get('origin')
    destination = data.get('destination')
    
    if not origin or not destination:
        return jsonify({'error': 'Origin and destination required'}), 400
    
    result = calculate_tomtom_route(origin, destination)
    return jsonify(result)

# ==================== Main ====================
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Travel Planner API - Render Deployment")
    print("=" * 60)
    print(f"📂 Loaded {len(cities_data)} cities")
    print(f"🤖 ML Model: {'✅ Loaded' if ml_api.time_model else '⚠️  Using fallback'}")
    print(f"🌤️  Weather: WeatherAPI.com ✅")
    print(f"🗺️  Routing: TomTom API ✅")
    print("=" * 60)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)