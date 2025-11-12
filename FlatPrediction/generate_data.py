"""
Cluj Flat Price Prediction Model
This module generates synthetic data for flat prices in Cluj-Napoca
based on realistic market criteria.
"""
import pandas as pd
import numpy as np

def generate_cluj_flat_data(n_samples=1000, random_state=42):
    """
    Generate synthetic flat price data for Cluj-Napoca.
    
    Features:
    - area_sqm: Living area in square meters
    - rooms: Number of rooms
    - bathrooms: Number of bathrooms
    - floor: Floor number
    - total_floors: Total floors in building
    - year_built: Year of construction
    - neighborhood_zone: Zone/neighborhood (1-5, where 5 is most desirable)
    - distance_to_center_km: Distance to city center
    - has_parking: Has parking spot (0/1)
    - has_balcony: Has balcony (0/1)
    - condition: Condition (1=needs renovation, 2=good, 3=excellent/renovated)
    
    Target:
    - price_eur: Price in EUR
    """
    np.random.seed(random_state)
    
    # Generate features
    area_sqm = np.random.normal(65, 25, n_samples)
    area_sqm = np.clip(area_sqm, 30, 200)
    
    rooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.15, 0.35, 0.30, 0.15, 0.05])
    bathrooms = np.random.choice([1, 2, 3], n_samples, p=[0.70, 0.25, 0.05])
    
    floor = np.random.randint(0, 11, n_samples)
    total_floors = floor + np.random.randint(1, 5, n_samples)
    
    year_built = np.random.choice(
        [1970, 1980, 1990, 2000, 2010, 2015, 2020],
        n_samples,
        p=[0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05]
    )
    
    neighborhood_zone = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.15, 0.25, 0.30, 0.20, 0.10])
    distance_to_center_km = np.random.exponential(3, n_samples)
    distance_to_center_km = np.clip(distance_to_center_km, 0.5, 15)
    
    has_parking = np.random.choice([0, 1], n_samples, p=[0.60, 0.40])
    has_balcony = np.random.choice([0, 1], n_samples, p=[0.30, 0.70])
    condition = np.random.choice([1, 2, 3], n_samples, p=[0.20, 0.50, 0.30])
    
    # Calculate price based on realistic factors
    # Base price per sqm in EUR (Cluj market 2023-2024 range: 1500-2500 EUR/sqm)
    base_price_per_sqm = 1800
    
    price_eur = (
        area_sqm * base_price_per_sqm *
        (1 + (neighborhood_zone - 3) * 0.15) *  # Zone impact
        (1 - distance_to_center_km * 0.02) *  # Distance penalty
        (1 + (year_built - 1990) * 0.003) *  # Age impact
        (1 + condition * 0.08) *  # Condition bonus
        (1 + has_parking * 0.08) *  # Parking bonus
        (1 + has_balcony * 0.03) *  # Balcony bonus
        (1 + (rooms - 2.5) * 0.05) *  # Rooms adjustment
        (1 - (floor == 0) * 0.05) *  # Ground floor penalty
        (1 - (floor == total_floors - 1) * 0.03)  # Top floor slight penalty
    )
    
    # Add some random noise
    noise = np.random.normal(1, 0.08, n_samples)
    price_eur = price_eur * noise
    price_eur = np.clip(price_eur, 40000, 400000)
    
    # Create DataFrame
    data = pd.DataFrame({
        'area_sqm': area_sqm,
        'rooms': rooms,
        'bathrooms': bathrooms,
        'floor': floor,
        'total_floors': total_floors,
        'year_built': year_built,
        'neighborhood_zone': neighborhood_zone,
        'distance_to_center_km': distance_to_center_km,
        'has_parking': has_parking,
        'has_balcony': has_balcony,
        'condition': condition,
        'price_eur': price_eur
    })
    
    return data

if __name__ == "__main__":
    # Generate and save data
    df = generate_cluj_flat_data(1000)
    df.to_csv('cluj_flat_prices.csv', index=False)
    print(f"Generated {len(df)} flat price records")
    print("\nDataset statistics:")
    print(df.describe())
    print("\nSample records:")
    print(df.head(10))
