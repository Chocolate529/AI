# Cluj Flat Price Prediction Model

A machine learning model to predict flat/apartment prices in Cluj-Napoca, Romania based on various property characteristics.

## Features

The model predicts flat prices based on the following criteria:
- **Area**: Living area in square meters
- **Rooms**: Number of rooms
- **Bathrooms**: Number of bathrooms
- **Floor**: Floor number
- **Total Floors**: Total floors in the building
- **Year Built**: Construction year
- **Neighborhood Zone**: Quality of neighborhood (1-5, where 5 is most desirable)
- **Distance to Center**: Distance to city center in kilometers
- **Parking**: Has parking spot (Yes/No)
- **Balcony**: Has balcony (Yes/No)
- **Condition**: Property condition (1=needs renovation, 2=good, 3=excellent/renovated)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Chocolate529/AI.git
cd AI
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Training Data

First, generate synthetic training data for Cluj flat prices:

```bash
cd FlatPrediction
python generate_data.py
```

This creates a `cluj_flat_prices.csv` file with 1000 sample records.

### 2. Train the Model

Train the machine learning model:

```bash
python train_model.py
```

This will:
- Train a Random Forest regression model
- Display evaluation metrics (MAE, RMSE, R²)
- Show feature importance
- Save visualization plots
- Save the trained model to `flat_price_model.pkl`

### 3. Make Predictions

#### Interactive Single Prediction

Run the prediction script interactively:

```bash
python predict.py
```

You'll be prompted to enter flat characteristics, and the model will predict the price.

#### Batch Predictions from CSV

Predict prices for multiple flats from a CSV file:

```bash
python predict.py input_flats.csv output_predictions.csv
```

## Example Prediction

```python
from train_model import FlatPricePredictor

# Load the trained model
predictor = FlatPricePredictor.load_model('flat_price_model.pkl')

# Define flat characteristics
flat = {
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

# Predict price
price = predictor.predict(flat)[0]
print(f"Predicted price: {price:,.0f} EUR")
```

## Model Performance

The Random Forest model typically achieves:
- **R² Score**: ~0.95-0.98 on test data
- **MAE**: ~5,000-10,000 EUR
- **RMSE**: ~8,000-15,000 EUR

These metrics indicate high prediction accuracy on the training data.

## Project Structure

```
FlatPrediction/
├── generate_data.py      # Generate synthetic training data
├── train_model.py        # Train and evaluate the model
├── predict.py            # Make predictions on new data
├── cluj_flat_prices.csv  # Generated training data
├── flat_price_model.pkl  # Trained model (after training)
├── feature_importance.png # Feature importance visualization
└── predictions_plot.png   # Actual vs predicted prices plot
```

## Model Details

The model uses **Random Forest Regression** with the following engineered features:
- Age of the property (derived from year built)
- Price per room ratio
- Top floor indicator
- Ground floor indicator

The most important features for price prediction are typically:
1. Area in square meters
2. Neighborhood zone
3. Year built / Age
4. Number of rooms
5. Distance to city center

## Data Source

The current implementation uses synthetically generated data that reflects realistic Cluj-Napoca real estate market conditions (2023-2024). For production use, replace with actual market data from real estate platforms or agencies.

## Future Improvements

- Integration with real Cluj real estate data APIs
- Additional features (heating type, orientation, elevator, etc.)
- Deep learning models for improved accuracy
- Web API for easy integration
- Deployment as a web service

## License

This project is open source and available under the MIT License.

## Author

Created for predicting flat prices in Cluj-Napoca, Romania.
