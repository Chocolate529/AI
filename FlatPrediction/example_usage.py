"""
Quick Start Example for Cluj Flat Price Prediction
This script demonstrates the basic usage of the flat price prediction model.
"""
import sys
import os

# Add FlatPrediction to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FlatPrediction'))

from train_model import FlatPricePredictor

def main():
    print("=" * 70)
    print("Cluj Flat Price Prediction - Quick Start Example")
    print("=" * 70)
    
    # Load the pre-trained model
    try:
        predictor = FlatPricePredictor.load_model('FlatPrediction/flat_price_model.pkl')
        print("\n✓ Model loaded successfully!")
    except FileNotFoundError:
        print("\n✗ Model not found. Please run the following commands first:")
        print("  cd FlatPrediction")
        print("  python generate_data.py")
        print("  python train_model.py")
        return
    
    # Example flats with different characteristics
    examples = [
        {
            'name': 'Modern 2-room apartment in central area',
            'data': {
                'area_sqm': 65,
                'rooms': 2,
                'bathrooms': 1,
                'floor': 3,
                'total_floors': 10,
                'year_built': 2015,
                'neighborhood_zone': 4,
                'distance_to_center_km': 2.5,
                'has_parking': 1,
                'has_balcony': 1,
                'condition': 3
            }
        },
        {
            'name': 'Older 2-room flat in suburban area',
            'data': {
                'area_sqm': 50,
                'rooms': 2,
                'bathrooms': 1,
                'floor': 1,
                'total_floors': 4,
                'year_built': 1985,
                'neighborhood_zone': 2,
                'distance_to_center_km': 5.0,
                'has_parking': 0,
                'has_balcony': 1,
                'condition': 2
            }
        },
        {
            'name': 'Luxury 4-room apartment in premium area',
            'data': {
                'area_sqm': 120,
                'rooms': 4,
                'bathrooms': 2,
                'floor': 5,
                'total_floors': 8,
                'year_built': 2020,
                'neighborhood_zone': 5,
                'distance_to_center_km': 1.5,
                'has_parking': 1,
                'has_balcony': 1,
                'condition': 3
            }
        },
        {
            'name': 'Compact studio near city center',
            'data': {
                'area_sqm': 35,
                'rooms': 1,
                'bathrooms': 1,
                'floor': 2,
                'total_floors': 5,
                'year_built': 2012,
                'neighborhood_zone': 4,
                'distance_to_center_km': 1.0,
                'has_parking': 0,
                'has_balcony': 0,
                'condition': 2
            }
        }
    ]
    
    print("\n" + "-" * 70)
    print("Predicting prices for example flats:")
    print("-" * 70)
    
    for i, example in enumerate(examples, 1):
        flat = example['data']
        price = predictor.predict(flat)[0]
        price_per_sqm = price / flat['area_sqm']
        
        print(f"\n{i}. {example['name']}")
        print(f"   Area: {flat['area_sqm']} sqm")
        print(f"   Rooms: {flat['rooms']} | Bathrooms: {flat['bathrooms']}")
        print(f"   Floor: {flat['floor']}/{flat['total_floors']}")
        print(f"   Year: {flat['year_built']} | Zone: {flat['neighborhood_zone']}/5")
        print(f"   Distance to center: {flat['distance_to_center_km']} km")
        print(f"   Parking: {'Yes' if flat['has_parking'] else 'No'} | "
              f"Balcony: {'Yes' if flat['has_balcony'] else 'No'}")
        condition_map = {1: 'Needs renovation', 2: 'Good', 3: 'Excellent'}
        print(f"   Condition: {condition_map[flat['condition']]}")
        print(f"   → Predicted Price: {price:,.0f} EUR ({price_per_sqm:,.0f} EUR/sqm)")
    
    print("\n" + "=" * 70)
    print("To predict prices for your own flats:")
    print("  1. Use the interactive mode: python FlatPrediction/predict.py")
    print("  2. Or batch mode: python FlatPrediction/predict.py input.csv output.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
