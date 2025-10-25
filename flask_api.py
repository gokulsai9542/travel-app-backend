from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import os
import requests
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# TomTom API Configuration
TOMTOM_API_KEY = 'glIsJVxQEiiCItEgu144dgLrKKWQXIkG'
TOMTOM_ROUTING_URL = 'https://api.tomtom.com/routing/1/calculateRoute/{locations}/json'
TOMTOM_SEARCH_URL = 'https://api.tomtom.com/search/2/geocode/{query}.json'

# Load ML Model
try:
    with open('travel_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data  # Dictionary with all model components
    print("✓ ML Model loaded successfully")
except FileNotFoundError:
    model = None
    print("⚠️  ML Model not found. Using rule-based recommendations.")
    print("   Run 'python train_model.py' to train the model.")
except Exception as e:
    model = None
    print(f"⚠️  Error loading model: {e}")

# Load city data
def load_city_data(city):
    """Load city data from JSON file"""
    file_path = f'data/{city.lower()}.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def get_coordinates_from_address(address):
    """Get lat/lon from address using TomTom Geocoding"""
    url = TOMTOM_SEARCH_URL.format(query=address)
    params = {'key': TOMTOM_API_KEY}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['results']:
            position = data['results'][0]['position']
            return position['lat'], position['lon']
    except Exception as e:
        print(f"Geocoding error: {e}")
    
    return None, None

def calculate_route(origin_lat, origin_lon, dest_lat, dest_lon, departure_time=None):
    """Calculate route using TomTom Routing API"""
    locations = f"{origin_lat},{origin_lon}:{dest_lat},{dest_lon}"
    url = TOMTOM_ROUTING_URL.format(locations=locations)
    
    params = {
        'key': TOMTOM_API_KEY,
        'traffic': 'true',
        'travelMode': 'car',
        'routeType': 'fastest'
    }
    
    if departure_time:
        params['departAt'] = departure_time
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'routes' in data and len(data['routes']) > 0:
            route = data['routes'][0]
            summary = route['summary']
            
            return {
                'distance_meters': summary['lengthInMeters'],
                'distance_km': round(summary['lengthInMeters'] / 1000, 2),
                'travel_time_seconds': summary['travelTimeInSeconds'],
                'travel_time_minutes': round(summary['travelTimeInSeconds'] / 60),
                'traffic_delay_seconds': summary.get('trafficDelayInSeconds', 0),
                'departure_time': summary.get('departureTime'),
                'arrival_time': summary.get('arrivalTime'),
                'route_points': route['legs'][0]['points'] if 'legs' in route else []
            }
    except Exception as e:
        print(f"Routing error: {e}")
    
    return None

def get_visit_duration(place_type):
    """Estimate visit duration based on place type"""
    duration_map = {
        # Attractions
        'attraction': 60,
        'museum': 120,
        'temple': 45,
        'park': 90,
        'beach': 120,
        'monument': 60,
        'fort': 90,
        'market': 90,
        'viewpoint': 30,
        # Food
        'food': 60,
        'restaurant': 60,
        'cafe': 30,
        # Shopping
        'shopping': 90,
        'mall': 120,
        # Hotels (just viewing/visiting)
        'hotel': 30,
    }
    return duration_map.get(place_type.lower(), 60)

def enhance_with_ml_predictions(places, user_context, model_data):
    """Enhance recommendations with ML predictions"""
    if not model_data:
        return places
    
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        preference_model = model_data.get('preference_model')
        duration_model = model_data.get('duration_model')
        scaler = model_data.get('scaler')
        
        if not all([preference_model, duration_model, scaler]):
            return places
        
        enhanced_places = []
        for place in places:
            # Extract features
            rating = place.get('rating', 4.0)
            opening = place.get('opening_hours', '09:00')
            closing = place.get('closing_hours', '18:00')
            
            try:
                open_hour = int(opening.split(':')[0])
                close_hour = int(closing.split(':')[0])
            except:
                open_hour, close_hour = 9, 18
            
            category_scores = {
                'attraction': 5, 'temple': 4, 'museum': 4,
                'park': 4, 'beach': 5, 'food': 3,
                'shopping': 3, 'hotel': 2
            }
            category = place.get('category') or place.get('type', 'attraction')
            category_score = category_scores.get(category.lower(), 3)
            
            # Create feature vector
            features = np.array([[
                rating,
                open_hour,
                close_hour,
                close_hour - open_hour,
                1 if rating >= 4.3 else 0,
                category_score,
                user_context.get('available_hours', 4),
                user_context.get('time_of_day', 12)
            ]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Predict
            visit_prob = preference_model.predict_proba(features_scaled)[0][1]
            duration = int(duration_model.predict(features_scaled)[0])
            
            # Add ML predictions to place
            place['ml_visit_probability'] = float(visit_prob)
            place['ml_recommended_duration'] = duration
            place['ml_confidence'] = 'high' if visit_prob > 0.7 else 'medium' if visit_prob > 0.5 else 'low'
            
            enhanced_places.append(place)
        
        # Sort by ML probability
        enhanced_places.sort(key=lambda x: x.get('ml_visit_probability', 0), reverse=True)
        return enhanced_places
        
    except Exception as e:
        print(f"ML enhancement error: {e}")
        return places

def filter_places_by_time(places, available_hours, origin_lat, origin_lon, user_context=None):
    """Filter and rank places based on available time"""
    available_minutes = available_hours * 60
    recommended_places = []
    
    for place in places:
        # Support both 'latitude'/'longitude' and 'lat'/'lng' formats
        place_lat = place.get('latitude') or place.get('lat')
        place_lng = place.get('longitude') or place.get('lng')
        
        if not place_lat or not place_lng:
            continue
        
        # Calculate route
        route_info = calculate_route(
            origin_lat, origin_lon,
            place_lat, place_lng
        )
        
        if not route_info:
            continue
        
        # Average visit duration based on place type
        visit_duration = get_visit_duration(place.get('category') or place.get('type', 'attraction'))
        
        # Total time needed (travel to + visit + travel back)
        total_time_needed = (route_info['travel_time_minutes'] * 2) + visit_duration
        
        if total_time_needed <= available_minutes:
            # Normalize place data to include both formats
            place_with_route = {
                **place,
                'latitude': place_lat,
                'longitude': place_lng,
                'lat': place_lat,
                'lng': place_lng,
                'route_info': route_info,
                'visit_duration_minutes': visit_duration,
                'total_time_minutes': total_time_needed,
                'time_breakdown': {
                    'travel_to': route_info['travel_time_minutes'],
                    'visit': visit_duration,
                    'travel_back': route_info['travel_time_minutes']
                }
            }
            recommended_places.append(place_with_route)
    
    # Apply ML enhancement if model is available and user_context provided
    if model and user_context:
        recommended_places = enhance_with_ml_predictions(
            recommended_places, 
            user_context, 
            model
        )
    else:
        # Sort by total time (closest first) if no ML
        recommended_places.sort(key=lambda x: x['total_time_minutes'])
    
    return recommended_places

def create_optimized_itinerary(places, available_hours, origin_lat, origin_lon):
    """Create an optimized multi-stop itinerary"""
    available_minutes = available_hours * 60
    itinerary = []
    current_lat, current_lon = origin_lat, origin_lon
    remaining_time = available_minutes
    visited_indices = set()
    
    while remaining_time > 30:  # At least 30 mins needed
        best_place = None
        best_route = None
        best_index = None
        
        for idx, place in enumerate(places):
            if idx in visited_indices:
                continue
            
            # Support both formats
            place_lat = place.get('latitude') or place.get('lat')
            place_lng = place.get('longitude') or place.get('lng')
            
            if not place_lat or not place_lng:
                continue
            
            route_info = calculate_route(
                current_lat, current_lon,
                place_lat, place_lng
            )
            
            if not route_info:
                continue
            
            visit_duration = get_visit_duration(place.get('category') or place.get('type', 'attraction'))
            time_needed = route_info['travel_time_minutes'] + visit_duration
            
            if time_needed <= remaining_time:
                # Prioritize ML probability if available, otherwise use time
                if model and 'ml_visit_probability' in place:
                    if best_place is None or place['ml_visit_probability'] > best_place.get('ml_visit_probability', 0):
                        best_place = place
                        best_route = route_info
                        best_index = idx
                else:
                    if best_place is None or time_needed < (best_route['travel_time_minutes'] + get_visit_duration(best_place.get('category') or best_place.get('type', 'attraction'))):
                        best_place = place
                        best_route = route_info
                        best_index = idx
        
        if best_place is None:
            break
        
        visit_duration = get_visit_duration(best_place.get('category') or best_place.get('type', 'attraction'))
        
        # Get coordinates
        place_lat = best_place.get('latitude') or best_place.get('lat')
        place_lng = best_place.get('longitude') or best_place.get('lng')
        
        itinerary.append({
            **best_place,
            'latitude': place_lat,
            'longitude': place_lng,
            'lat': place_lat,
            'lng': place_lng,
            'route_from_previous': best_route,
            'visit_duration_minutes': visit_duration,
            'order': len(itinerary) + 1
        })
        
        visited_indices.add(best_index)
        current_lat = place_lat
        current_lon = place_lng
        remaining_time -= (best_route['travel_time_minutes'] + visit_duration)
    
    # Add return route to origin
    if itinerary:
        last_place = itinerary[-1]
        return_route = calculate_route(
            last_place['latitude'], last_place['longitude'],
            origin_lat, origin_lon
        )
        
        return {
            'places': itinerary,
            'return_route': return_route,
            'total_places': len(itinerary),
            'total_time_used': available_minutes - remaining_time,
            'time_remaining': remaining_time
        }
    
    return None

# API ENDPOINTS

@app.route('/api/geocode', methods=['POST'])
def geocode():
    """Convert address to coordinates"""
    data = request.json
    address = data.get('address')
    
    if not address:
        return jsonify({'error': 'Address required'}), 400
    
    lat, lon = get_coordinates_from_address(address)
    
    if lat and lon:
        return jsonify({
            'latitude': lat,
            'longitude': lon,
            'address': address
        })
    
    return jsonify({'error': 'Location not found'}), 404

@app.route('/api/route', methods=['POST'])
def get_route():
    """Calculate route between two points"""
    data = request.json
    
    origin_lat = data.get('origin_lat')
    origin_lon = data.get('origin_lon')
    dest_lat = data.get('dest_lat')
    dest_lon = data.get('dest_lon')
    
    if not all([origin_lat, origin_lon, dest_lat, dest_lon]):
        return jsonify({'error': 'All coordinates required'}), 400
    
    route_info = calculate_route(origin_lat, origin_lon, dest_lat, dest_lon)
    
    if route_info:
        return jsonify(route_info)
    
    return jsonify({'error': 'Route calculation failed'}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend_places():
    """Recommend places based on time and location with optional ML enhancement"""
    data = request.json
    
    city = data.get('city')
    available_hours = data.get('available_hours', 2)
    origin_lat = data.get('origin_lat')
    origin_lon = data.get('origin_lon')
    user_address = data.get('user_address')
    
    # Optional: user context for ML enhancement
    user_context = {
        'available_hours': available_hours,
        'time_of_day': data.get('time_of_day', 12)
    }
    
    # Get origin coordinates from address if not provided
    if not origin_lat or not origin_lon:
        if user_address:
            origin_lat, origin_lon = get_coordinates_from_address(user_address)
            if not origin_lat:
                return jsonify({'error': 'Could not geocode address'}), 400
        else:
            return jsonify({'error': 'Location required (coordinates or address)'}), 400
    
    # Load city data
    city_data = load_city_data(city)
    if not city_data:
        return jsonify({'error': 'City data not found'}), 404
    
    places = city_data.get('places', [])
    
    # Filter places by available time with ML enhancement
    recommended = filter_places_by_time(
        places, 
        available_hours, 
        origin_lat, 
        origin_lon,
        user_context
    )
    
    return jsonify({
        'city': city,
        'origin': {'latitude': origin_lat, 'longitude': origin_lon},
        'available_hours': available_hours,
        'recommended_places': recommended[:10],  # Top 10
        'total_found': len(recommended),
        'ml_enhanced': model is not None
    })

@app.route('/api/itinerary', methods=['POST'])
def generate_itinerary():
    """Generate optimized multi-stop itinerary with optional ML enhancement"""
    data = request.json
    
    city = data.get('city')
    available_hours = data.get('available_hours', 4)
    origin_lat = data.get('origin_lat')
    origin_lon = data.get('origin_lon')
    user_address = data.get('user_address')
    preferences = data.get('preferences', {})
    
    # User context for ML enhancement
    user_context = {
        'available_hours': available_hours,
        'time_of_day': data.get('time_of_day', 12)
    }
    
    # Get origin coordinates
    if not origin_lat or not origin_lon:
        if user_address:
            origin_lat, origin_lon = get_coordinates_from_address(user_address)
            if not origin_lat:
                return jsonify({'error': 'Could not geocode address'}), 400
        else:
            return jsonify({'error': 'Location required'}), 400
    
    # Load city data
    city_data = load_city_data(city)
    if not city_data:
        return jsonify({'error': 'City data not found'}), 404
    
    places = city_data.get('places', [])
    
    # Filter by preferences if provided
    if preferences:
        place_category = preferences.get('category') or preferences.get('type')
        if place_category:
            places = [p for p in places if 
                     (p.get('category', '').lower() == place_category.lower() or 
                      p.get('type', '').lower() == place_category.lower())]
    
    # Apply ML enhancement to places before creating itinerary
    if model:
        places = enhance_with_ml_predictions(places, user_context, model)
    
    # Generate optimized itinerary
    itinerary = create_optimized_itinerary(places, available_hours, origin_lat, origin_lon)
    
    if itinerary:
        return jsonify({
            'city': city,
            'origin': {'latitude': origin_lat, 'longitude': origin_lon},
            'available_hours': available_hours,
            'itinerary': itinerary,
            'ml_enhanced': model is not None
        })
    
    return jsonify({'error': 'Could not generate itinerary'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok', 
        'model_loaded': model is not None,
        'model_type': 'ML-enhanced' if model else 'rule-based'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)