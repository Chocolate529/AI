"""
Prediction Script for Cluj Flat Prices
Use this script to make predictions on new flat data.
"""
import pandas as pd
from train_model import FlatPricePredictor

def predict_single_flat():
    """
    Interactive prediction for a single flat.
    """
    print("=" * 60)
    print("Cluj Flat Price Predictor")
    print("=" * 60)
    
    # Load the trained model
    try:
        predictor = FlatPricePredictor.load_model('flat_price_model.pkl')
    except FileNotFoundError:
        print("Error: Model file 'flat_price_model.pkl' not found.")
        print("Please run 'python train_model.py' first to train the model.")
        return
    
    print("\nEnter flat characteristics:")
    print("-" * 60)
    
    # Get user input
    try:
        area_sqm = float(input("Area (square meters): "))
        rooms = int(input("Number of rooms: "))
        bathrooms = int(input("Number of bathrooms: "))
        floor = int(input("Floor number: "))
        total_floors = int(input("Total floors in building: "))
        year_built = int(input("Year built: "))
        neighborhood_zone = int(input("Neighborhood zone (1-5, where 5 is best): "))
        distance_to_center_km = float(input("Distance to city center (km): "))
        has_parking = int(input("Has parking? (0=No, 1=Yes): "))
        has_balcony = int(input("Has balcony? (0=No, 1=Yes): "))
        condition = int(input("Condition (1=needs renovation, 2=good, 3=excellent): "))
        
        # Create flat data
        flat_data = {
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
            'condition': condition
        }
        
        # Make prediction
        predicted_price = predictor.predict(flat_data)[0]
        
        # Display results
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Predicted Price: {predicted_price:,.0f} EUR")
        print(f"Price per sqm: {predicted_price/area_sqm:,.0f} EUR/sqm")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\nError: Invalid input - {e}")
    except KeyboardInterrupt:
        print("\n\nPrediction cancelled.")


def predict_from_csv(input_file, output_file='predictions.csv'):
    """
    Make predictions for multiple flats from a CSV file.
    
    Args:
        input_file: Path to CSV file with flat data
        output_file: Path to save predictions
    """
    # Load model
    try:
        predictor = FlatPricePredictor.load_model('flat_price_model.pkl')
    except FileNotFoundError:
        print("Error: Model file 'flat_price_model.pkl' not found.")
        print("Please run 'python train_model.py' first to train the model.")
        return
    
    # Load data
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    
    # Make predictions
    predictions = predictor.predict(df)
    
    # Add predictions to dataframe
    df['predicted_price_eur'] = predictions
    df['predicted_price_per_sqm'] = df['predicted_price_eur'] / df['area_sqm']
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    print(f"\nProcessed {len(df)} flats")
    print(f"Average predicted price: {predictions.mean():,.0f} EUR")
    print(f"Price range: {predictions.min():,.0f} - {predictions.max():,.0f} EUR")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Batch prediction mode
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
        predict_from_csv(input_file, output_file)
    else:
        # Interactive mode
        predict_single_flat()
