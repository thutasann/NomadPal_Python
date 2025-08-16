from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timezone
import without_login  # Import our ML module directly!

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# All ML logic is now handled by without_login.py!
# No need to duplicate the code here

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "NomadPal Model Service is running",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

@app.route('/cities/top', methods=['GET'])
def get_top_cities():
    """Get top cities using without_login module"""
    try:
        limit = min(int(request.args.get('limit', 10)), 50)
        cities = without_login.get_top_cities(limit)
        
        # Debug: Print first city to verify ID format
        if cities:
            print(f"🔍 Sample city ID: {cities[0].get('id')} (type: {type(cities[0].get('id'))})")
            print(f"🔍 Sample city name: {cities[0].get('name')}")
            print(f"🔍 Sample city slug: {cities[0].get('slug')}")
        
        return jsonify({
            "status": "success",
            "data": cities,
            "total": len(cities),
            "limit": limit
        })
    except Exception as e:
        print(f"❌ Error in get_top_cities: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/cities/search', methods=['GET'])
def search_cities():
    """Search cities using without_login module"""
    try:
        query = request.args.get('q', '').strip()
        limit = min(int(request.args.get('limit', 10)), 50)
        
        if not query:
            return jsonify({"error": "Query parameter 'q' is required"}), 400
        
        cities = without_login.search_cities(query, limit)
        
        return jsonify({
            "status": "success",
            "data": cities,
            "total": len(cities),
            "query": query
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cities/by-country/<country>', methods=['GET'])
def get_cities_by_country_route(country):
    """Get cities by country using without_login module"""
    try:
        limit = min(int(request.args.get('limit', 10)), 50)
        cities = without_login.get_cities_by_country(country, limit)
        
        return jsonify({
            "status": "success",
            "data": cities,
            "total": len(cities),
            "country": country
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/countries', methods=['GET'])
def get_countries_route():
    """Get countries using without_login module"""
    try:
        countries = without_login.get_countries()
        
        return jsonify({
            "status": "success",
            "data": countries,
            "total": len(countries)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats_route():
    """Get statistics using without_login module"""
    try:
        stats = without_login.get_stats()
        
        return jsonify({
            "status": "success",
            "data": stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Model is automatically loaded when we import without_login!
    print("🚀 Starting NomadPal API using without_login.py")
    print("📊 Model training happens automatically on import")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)