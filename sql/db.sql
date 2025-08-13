
CREATE TABLE users (
  id CHAR(24) PRIMARY KEY,
  email VARCHAR(191) NOT NULL UNIQUE,
  password_hash VARCHAR(191) NOT NULL,
  display_name VARCHAR(128),
  preferred_language VARCHAR(64),
  country_city VARCHAR(128),
  timezone VARCHAR(64),
  passport VARCHAR(128),
  visa_flexibility VARCHAR(128),
  preferred_regions VARCHAR(128),
  job_title VARCHAR(128),
  target_salary_usd DECIMAL(12,2),
  salary_currency CHAR(3) DEFAULT 'USD',
  sources VARCHAR(255),
  work_style VARCHAR(64),
  monthly_budget_min_usd DECIMAL(10,2),
  monthly_budget_max_usd DECIMAL(10,2),
  preferred_climate VARCHAR(64),
  internet_speed_requirement VARCHAR(64),
  lifestyle_priorities JSON,
  newsletter_consent BOOLEAN DEFAULT FALSE,
  research_consent BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE cities (
  id CHAR(24) PRIMARY KEY,
  slug VARCHAR(128) NOT NULL UNIQUE,
  name VARCHAR(128) NOT NULL,
  country VARCHAR(128) NOT NULL,
  description TEXT,
  visa_requirement TEXT,
  monthly_cost_usd DECIMAL(10,2),
  avg_pay_rate_usd_hour DECIMAL(8,2),
  weather_avg_temp_c DECIMAL(4,1),
  safety_score DECIMAL(5,2),
  nightlife_rating DECIMAL(4,2),
  transport_rating DECIMAL(4,2),
  housing_studio_usd_month DECIMAL(10,2),
  housing_one_bed_usd_month DECIMAL(10,2),
  housing_coliving_usd_month DECIMAL(10,2),
  climate_avg_temp_c DECIMAL(4,1),
  climate_summary VARCHAR(128),
  internet_speed VARCHAR (128),
  cost_pct_rent DECIMAL(5,2),
  cost_pct_dining DECIMAL(5,2),
  cost_pct_transport DECIMAL(5,2),
  cost_pct_groceries DECIMAL(5,2),
  cost_pct_coworking DECIMAL(5,2),
  cost_pct_other DECIMAL(5,2),
  travel_flight_from_usd DECIMAL(10,2),
  travel_local_transport_usd_week DECIMAL(10,2),
  travel_hotel_usd_week DECIMAL(10,2),
  lifestyle_tags JSON,
  currency CHAR(3) DEFAULT 'USD',
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE jobs (
  id CHAR(24) PRIMARY KEY,
  city_id CHAR(24),
  title VARCHAR(255) NOT NULL,
  company VARCHAR(255),
  location VARCHAR(255),
  category VARCHAR(128),
  job_type VARCHAR(64),
  posted_date DATE,
  min_salary DECIMAL(12,2),
  max_salary DECIMAL(12,2),
  salary_currency CHAR(3) DEFAULT 'USD',
  salary_period VARCHAR(32),
  description TEXT,
  source VARCHAR(128),
  source_url VARCHAR(512),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (city_id) REFERENCES cities(id)
);


CREATE TABLE saved_cities (
  user_id CHAR(24),
  city_id CHAR(24),
  PRIMARY KEY (user_id, city_id),
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (city_id) REFERENCES cities(id)
);

CREATE TABLE saved_jobs (
  user_id CHAR(24),
  job_id CHAR(24),
  status VARCHAR(64), -- e.g., Applied, Interested
  PRIMARY KEY (user_id, job_id),
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (job_id) REFERENCES jobs(id)
);