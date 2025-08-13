#!/bin/bash

# NomadPal Model API Test Script
# Make sure the server is running: python3 app.py

BASE_URL="http://localhost:5000"

echo "🚀 Testing NomadPal Model API"
echo "================================"

# Test 1: Health Check
echo "1️⃣ Testing Health Check..."
curl -s -X GET "$BASE_URL/health" | jq
echo -e "\n"

# Test 2: Get Top Cities
echo "2️⃣ Testing Top Cities (limit=3)..."
curl -s -X GET "$BASE_URL/cities/top?limit=3" | jq '.data[] | {name, country, predicted_score}'
echo -e "\n"

# Test 3: Search Cities
echo "3️⃣ Testing City Search (query=thailand)..."
curl -s -X GET "$BASE_URL/cities/search?q=thailand&limit=5" | jq '.data[] | {name, country, predicted_score}'
echo -e "\n"

# Test 4: Cities by Country
echo "4️⃣ Testing Cities by Country (portugal)..."
curl -s -X GET "$BASE_URL/cities/by-country/portugal" | jq '.data[] | {name, monthly_cost_usd, safety_score}'
echo -e "\n"

# Test 5: Get Countries
echo "5️⃣ Testing Get Countries..."
curl -s -X GET "$BASE_URL/countries" | jq '.data[0:10]'
echo -e "\n"

# Test 6: Get Stats
echo "6️⃣ Testing Dataset Statistics..."
curl -s -X GET "$BASE_URL/stats" | jq
echo -e "\n"

echo "✅ API Testing Complete!"
echo "================================"
