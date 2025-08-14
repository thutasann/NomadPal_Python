# NomadPal Model Service

A Flask REST API service that provides machine learning-powered city recommendations for digital nomads. The service uses a trained neural network to predict city suitability scores based on factors like cost of living, safety, weather, and infrastructure.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone and navigate to the project:**
```bash
cd NomadPal_Model
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start the server:**
```bash
python3 app.py
```

The server will start on `http://localhost:5000`

## 📊 Model Details

The service uses a neural network trained on city data with the following features:
- **Cost factors**: Monthly cost, rent percentages, housing costs
- **Climate**: Average temperature (optimized for ~22°C)
- **Safety**: Safety scores and ratings
- **Infrastructure**: Internet speed, transport ratings
- **Lifestyle**: Nightlife, coworking spaces, lifestyle tags

The model generates a suitability score for each city, with higher scores indicating better matches for digital nomads.

## 🔌 API Endpoints

### 1. Health Check
Check if the service is running.

```bash
curl -X GET http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "NomadPal Model Service is running",
  "timestamp": "2025-08-13T17:30:00.000Z"
}
```

### 2. Get Top Cities
Get the top-ranked cities based on ML predictions.

```bash
# Get top 10 cities (default)
curl -X GET http://localhost:5000/cities/top

# Get top 5 cities
curl -X GET "http://localhost:5000/cities/top?limit=5"

# Get top 20 cities
curl -X GET "http://localhost:5000/cities/top?limit=20"
```

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "id": "cc36eea674cc479f8069caaad533bba9",
      "slug": "buenos-aires",
      "name": "Buenos Aires",
      "country": "Argentina",
      "description": null,
      "monthly_cost_usd": "1043.53",
      "avg_pay_rate_usd_hour": "11.76",
      "weather_avg_temp_c": "26.0",
      "safety_score": "72.15",
      "predicted_score": 1.2345,
      "last_updated": "2025-08-13T17:01:19.909499"
    }
  ],
  "total": 10,
  "limit": 10
}
```

### 3. Search Cities
Search cities by name or country.

```bash
# Search for cities containing "london"
curl -X GET "http://localhost:5000/cities/search?q=london"

# Search for cities in "thailand"
curl -X GET "http://localhost:5000/cities/search?q=thailand&limit=5"

# Search for "berlin" with limit
curl -X GET "http://localhost:5000/cities/search?q=berlin&limit=3"
```

**Response:**
```json
{
  "status": "success",
  "data": [...],
  "total": 5,
  "query": "london"
}
```

### 4. Get Cities by Country
Get cities filtered by a specific country.

```bash
# Get cities in Thailand
curl -X GET http://localhost:5000/cities/by-country/thailand

# Get cities in Japan with limit
curl -X GET "http://localhost:5000/cities/by-country/japan?limit=5"

# Get cities in Portugal
curl -X GET http://localhost:5000/cities/by-country/portugal
```

**Response:**
```json
{
  "status": "success",
  "data": [...],
  "total": 8,
  "country": "thailand"
}
```

### 5. Get Available Countries
Get a list of all countries in the dataset.

```bash
curl -X GET http://localhost:5000/countries
```

**Response:**
```json
{
  "status": "success",
  "data": [
    "Argentina",
    "Australia", 
    "Brazil",
    "Canada",
    "Germany",
    "Japan",
    "Thailand",
    "United States"
  ],
  "total": 45
}
```

### 6. Get Dataset Statistics
Get basic statistics about the dataset.

```bash
curl -X GET http://localhost:5000/stats
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "total_cities": 782,
    "total_countries": 45,
    "avg_monthly_cost": 1456.78,
    "avg_safety_score": 68.45,
    "avg_temperature": 23.12
  }
}
```

## 🧪 Testing Examples

### Basic Testing Workflow

1. **Start the server:**
```bash
python3 app.py
```

2. **Test health endpoint:**
```bash
curl -X GET http://localhost:5000/health
```

3. **Get top 3 cities:**
```bash
curl -X GET "http://localhost:5000/cities/top?limit=3" | jq
```

4. **Search for cities in Europe:**
```bash
curl -X GET "http://localhost:5000/cities/search?q=europe&limit=10" | jq
```

5. **Get cities in a specific country:**
```bash
curl -X GET http://localhost:5000/cities/by-country/portugal | jq
```

### Advanced Testing

**Test with different parameters:**
```bash
# Large dataset query
curl -X GET "http://localhost:5000/cities/top?limit=50"

# Multiple search terms
curl -X GET "http://localhost:5000/cities/search?q=beach"
curl -X GET "http://localhost:5000/cities/search?q=mountain"

# Case insensitive country search
curl -X GET http://localhost:5000/cities/by-country/THAILAND
curl -X GET http://localhost:5000/cities/by-country/ThAiLaNd
```

**Error handling tests:**
```bash
# Missing query parameter
curl -X GET http://localhost:5000/cities/search

# Invalid limit (will be capped at 50)
curl -X GET "http://localhost:5000/cities/top?limit=1000"
```

## 📁 Data Format

Each city object contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `slug` | string | URL-friendly city name |
| `name` | string | City name |
| `country` | string | Country name |
| `monthly_cost_usd` | string | Monthly living cost in USD |
| `safety_score` | string | Safety rating (0-100) |
| `weather_avg_temp_c` | string | Average temperature in Celsius |
| `internet_speed` | string | Internet speed in Mbps |
| `predicted_score` | number | ML-generated suitability score |
| `housing_*` | string | Various housing cost metrics |
| `cost_pct_*` | string | Cost breakdown percentages |
| `lifestyle_tags` | array | Lifestyle characteristics |
| `last_updated` | string | ISO timestamp |

## 🛠️ Development

### Running in Development Mode
```bash
export FLASK_ENV=development
python3 app.py
```

### Model Retraining
The model is automatically trained on server startup. To retrain:
```bash
# The model retrains automatically when you restart the server
python3 app.py
```

### Adding New Features
1. Modify the feature engineering in `load_data_and_train_model()`
2. Update the model architecture if needed
3. Restart the server to retrain

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors:**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**2. CSV File Not Found:**
```bash
# Ensure the CSV file exists
ls csv/FINALIZED_cities_data.csv
```

**3. Model Training Warnings:**
- Warnings about missing data are normal and handled automatically
- The model uses imputation for missing values

**4. CORS Issues:**
- CORS is enabled for all origins
- For production, configure specific origins in `app.py`

### Performance Notes
- First request may be slower due to model loading
- Subsequent requests use cached predictions
- Consider using Redis for production caching

## 📝 API Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (missing parameters) |
| 500 | Internal Server Error (model issues) |

## 🚀 Deployment to Fly.io

### Prerequisites
1. Install [flyctl](https://fly.io/docs/getting-started/installing-flyctl/)
2. Sign up for [Fly.io account](https://fly.io/app/sign-up)
3. Authenticate: `flyctl auth login`

### Manual Deployment

1. **Initialize the Fly app:**
```bash
flyctl launch --no-deploy
```

2. **Set database secrets (update with your MySQL credentials):**
```bash
flyctl secrets set DB_HOST="your-mysql-host"
flyctl secrets set DB_PORT="3306"
flyctl secrets set DB_USER="your-username"
flyctl secrets set DB_PASSWORD="your-password"
flyctl secrets set DB_NAME="nomadpal_db"
```

3. **Deploy the application:**
```bash
flyctl deploy
```

4. **Check the deployment:**
```bash
flyctl status
flyctl logs
```

### Automatic Deployment with GitHub Actions

1. **Set up GitHub repository secrets:**
   - Go to your GitHub repository settings
   - Add secret: `FLY_API_TOKEN` (get from `flyctl auth token`)

2. **Push to main branch:**
```bash
git add .
git commit -m "Deploy NomadPal Model Service"
git push origin main
```

3. **Monitor deployment:**
   - Check GitHub Actions tab for deployment status
   - Visit: `https://nomadpal-model.fly.dev/health`

### Production Configuration

The `fly.toml` file is configured for:
- **Region**: Singapore (sin) - change as needed
- **Memory**: 1GB - adjust based on data size
- **Auto-scaling**: Stops when idle, starts on demand
- **HTTPS**: Force HTTPS enabled

### API Endpoints (Live)

Once deployed, your API will be available at:
```
https://nomadpal-model.fly.dev/health
https://nomadpal-model.fly.dev/cities/top?limit=10
https://nomadpal-model.fly.dev/cities/search?q=thailand
https://nomadpal-model.fly.dev/countries
```

## 🔮 Future Enhancements

- [ ] User preference-based scoring
- [ ] Real-time data updates
- [ ] Caching with Redis
- [ ] API rate limiting
- [ ] Authentication
- [ ] Batch prediction endpoints
- [ ] Model versioning

---

**Note:** This service is now production-ready with MySQL integration and Fly.io deployment!