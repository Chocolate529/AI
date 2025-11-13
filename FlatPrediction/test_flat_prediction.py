"""
Test suite for the Cluj Flat Price Prediction Model
"""
import sys
import os
import pandas as pd
import numpy as np

# Add FlatPrediction to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from FlatPrediction.train_model import FlatPricePredictor
from FlatPrediction.generate_data import generate_cluj_flat_data


def test_data_generation():
    """Test data generation function."""
    print("Testing data generation...")
    df = generate_cluj_flat_data(n_samples=100, random_state=42)
    
    # Check shape
    assert len(df) == 100, "Should generate 100 samples"
    assert len(df.columns) == 12, "Should have 12 columns"
    
    # Check required columns
    required_cols = [
        'area_sqm', 'rooms', 'bathrooms', 'floor', 'total_floors',
        'year_built', 'neighborhood_zone', 'distance_to_center_km',
        'has_parking', 'has_balcony', 'condition', 'price_eur'
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check data ranges
    assert df['area_sqm'].min() >= 30, "Minimum area should be >= 30 sqm"
    assert df['area_sqm'].max() <= 200, "Maximum area should be <= 200 sqm"
    assert df['rooms'].min() >= 1, "Minimum rooms should be >= 1"
    assert df['rooms'].max() <= 5, "Maximum rooms should be <= 5"
    assert df['neighborhood_zone'].min() >= 1, "Zone should be >= 1"
    assert df['neighborhood_zone'].max() <= 5, "Zone should be <= 5"
    assert df['price_eur'].min() >= 40000, "Price should be >= 40,000 EUR"
    assert df['price_eur'].max() <= 400000, "Price should be <= 400,000 EUR"
    
    print("✓ Data generation test passed!")


def test_model_training():
    """Test model training and prediction."""
    print("\nTesting model training...")
    
    # Generate small dataset
    df = generate_cluj_flat_data(n_samples=200, random_state=42)
    
    # Train model
    predictor = FlatPricePredictor(model_type='random_forest')
    metrics = predictor.train(df, test_size=0.3)
    
    # Check metrics exist
    assert 'train' in metrics, "Should have train metrics"
    assert 'test' in metrics, "Should have test metrics"
    assert 'mae' in metrics['test'], "Should have MAE"
    assert 'rmse' in metrics['test'], "Should have RMSE"
    assert 'r2' in metrics['test'], "Should have R²"
    
    # Check model performance
    assert metrics['test']['r2'] > 0.5, "R² should be > 0.5 for reasonable performance"
    assert metrics['test']['mae'] < 100000, "MAE should be reasonable"
    
    print(f"  Test R²: {metrics['test']['r2']:.4f}")
    print(f"  Test MAE: {metrics['test']['mae']:,.0f} EUR")
    print("✓ Model training test passed!")


def test_prediction():
    """Test prediction on single flat."""
    print("\nTesting prediction...")
    
    # Generate and train
    df = generate_cluj_flat_data(n_samples=200, random_state=42)
    predictor = FlatPricePredictor(model_type='random_forest')
    predictor.train(df, test_size=0.3)
    
    # Test prediction
    test_flat = {
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
    
    price = predictor.predict(test_flat)[0]
    
    # Check prediction is reasonable
    assert price > 50000, "Price should be > 50,000 EUR"
    assert price < 350000, "Price should be < 350,000 EUR"
    
    print(f"  Predicted price for test flat: {price:,.0f} EUR")
    print("✓ Prediction test passed!")


def test_feature_importance():
    """Test feature importance extraction."""
    print("\nTesting feature importance...")
    
    df = generate_cluj_flat_data(n_samples=200, random_state=42)
    predictor = FlatPricePredictor(model_type='random_forest')
    predictor.train(df, test_size=0.3)
    
    importance_df = predictor.get_feature_importance()
    
    assert importance_df is not None, "Should return feature importance"
    assert len(importance_df) > 0, "Should have some features"
    assert 'feature' in importance_df.columns, "Should have feature names"
    assert 'importance' in importance_df.columns, "Should have importance values"
    
    # Check that area_sqm is among top features (it should be the most important)
    top_feature = importance_df.iloc[0]['feature']
    print(f"  Most important feature: {top_feature}")
    assert top_feature == 'area_sqm', "Area should be the most important feature"
    
    print("✓ Feature importance test passed!")


def test_batch_prediction():
    """Test batch prediction on multiple flats."""
    print("\nTesting batch prediction...")
    
    df = generate_cluj_flat_data(n_samples=200, random_state=42)
    predictor = FlatPricePredictor(model_type='random_forest')
    predictor.train(df, test_size=0.3)
    
    # Create test batch
    test_data = pd.DataFrame([
        {
            'area_sqm': 55, 'rooms': 2, 'bathrooms': 1, 'floor': 2,
            'total_floors': 4, 'year_built': 2010, 'neighborhood_zone': 3,
            'distance_to_center_km': 3.0, 'has_parking': 1, 'has_balcony': 1,
            'condition': 2
        },
        {
            'area_sqm': 90, 'rooms': 3, 'bathrooms': 2, 'floor': 5,
            'total_floors': 10, 'year_built': 2020, 'neighborhood_zone': 5,
            'distance_to_center_km': 2.0, 'has_parking': 1, 'has_balcony': 1,
            'condition': 3
        }
    ])
    
    predictions = predictor.predict(test_data)
    
    assert len(predictions) == 2, "Should predict for 2 flats"
    assert all(p > 50000 for p in predictions), "All prices should be > 50,000 EUR"
    assert all(p < 350000 for p in predictions), "All prices should be < 350,000 EUR"
    assert predictions[1] > predictions[0], "Larger, newer flat should cost more"
    
    print(f"  Predictions: {predictions[0]:,.0f} EUR, {predictions[1]:,.0f} EUR")
    print("✓ Batch prediction test passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Cluj Flat Price Prediction Model Tests")
    print("=" * 60)
    
    try:
        test_data_generation()
        test_model_training()
        test_prediction()
        test_feature_importance()
        test_batch_prediction()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
