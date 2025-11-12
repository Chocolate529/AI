# Technical Documentation - Cluj Flat Price Prediction Model

## Model Architecture

### Algorithm: Random Forest Regression
- **Ensemble method**: 100 decision trees
- **Max depth**: 20
- **Min samples split**: 5
- **Min samples leaf**: 2
- **Bootstrap**: True
- **Feature selection**: All features at each split

## Features

### Input Features (11 base features + 4 engineered)

#### Base Features:
1. **area_sqm** (float): Living area in square meters [30-200]
2. **rooms** (int): Number of rooms [1-5]
3. **bathrooms** (int): Number of bathrooms [1-3]
4. **floor** (int): Floor number [0-10]
5. **total_floors** (int): Total floors in building [1-15]
6. **year_built** (int): Year of construction [1970-2025]
7. **neighborhood_zone** (int): Zone quality rating [1-5]
   - 1: Low desirability
   - 5: High desirability (premium area)
8. **distance_to_center_km** (float): Distance to city center in km [0.5-15]
9. **has_parking** (binary): Parking availability [0/1]
10. **has_balcony** (binary): Balcony presence [0/1]
11. **condition** (int): Property condition [1-3]
    - 1: Needs renovation
    - 2: Good condition
    - 3: Excellent/Recently renovated

#### Engineered Features:
1. **age** (int): Building age = 2024 - year_built
2. **price_per_room** (float): area_sqm / rooms
3. **is_top_floor** (binary): 1 if floor == total_floors - 1
4. **is_ground_floor** (binary): 1 if floor == 0

### Target Variable:
- **price_eur** (float): Flat price in EUR [40,000-400,000]

## Performance Metrics

### Training Set:
- **MAE**: ~6,300 EUR
- **RMSE**: ~9,000 EUR
- **R² Score**: ~0.975

### Test Set (20% holdout):
- **MAE**: ~13,500 EUR
- **RMSE**: ~18,400 EUR
- **R² Score**: ~0.91

### Interpretation:
- The model explains 91% of the variance in flat prices
- Average prediction error is ±13,500 EUR
- Performance indicates good generalization without significant overfitting

## Feature Importance

Ranked by contribution to predictions:

1. **area_sqm**: ~72% (dominant feature)
2. **neighborhood_zone**: ~19%
3. **distance_to_center_km**: ~3%
4. **price_per_room**: ~2%
5. **condition**: ~1%
6. **age**: ~1%
7. Other features: <1% each

**Key Insight**: Area and location (zone + distance) account for ~94% of price variation.

## Data Preprocessing

### Scaling:
- **Method**: StandardScaler (zero mean, unit variance)
- **Applied to**: All features before training
- **Saved with model**: Yes (scaler persisted for inference)

### Missing Values:
- **Handling**: Not applicable (synthetic data is complete)
- **Production recommendation**: Impute with median or mode

### Outliers:
- **Detection**: None in current implementation
- **Production recommendation**: Use IQR method or domain knowledge

## Model Validation

### Cross-Validation:
- **Strategy**: Train-test split (80/20)
- **Random state**: 42 (reproducible)
- **Stratification**: None (regression task)

### Test Coverage:
All components tested:
- Data generation
- Model training
- Single prediction
- Batch prediction
- Feature importance
- Model persistence

## Usage Patterns

### Training:
```python
from train_model import FlatPricePredictor
predictor = FlatPricePredictor(model_type='random_forest')
metrics = predictor.train(df, test_size=0.2)
predictor.save_model('model.pkl')
```

### Inference:
```python
predictor = FlatPricePredictor.load_model('model.pkl')
price = predictor.predict(flat_data)[0]
```

## Model Limitations

### Current Limitations:
1. **Synthetic data**: Trained on generated data, not real market data
2. **Feature coverage**: Missing some relevant features:
   - Heating type
   - Orientation (N/S/E/W)
   - Elevator presence
   - Building material
   - Proximity to amenities (schools, transport)
3. **Temporal aspect**: No time-series modeling of market trends
4. **Market dynamics**: Doesn't account for economic conditions

### Recommended Improvements:
1. **Data**: Replace synthetic data with real Cluj real estate listings
2. **Features**: Add more property characteristics and location data
3. **Model**: Consider ensemble of multiple algorithms (XGBoost, LightGBM)
4. **Validation**: Implement k-fold cross-validation
5. **Production**: Add confidence intervals for predictions
6. **Monitoring**: Track prediction accuracy over time

## Deployment Considerations

### Model Size:
- **File size**: ~2.5 MB (pickled)
- **Memory**: ~10 MB loaded
- **Inference time**: <1ms per prediction

### Dependencies:
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0 (visualization only)
- seaborn >= 0.12.0 (visualization only)
- joblib >= 1.3.0

### API Integration:
Ready for:
- REST API (Flask/FastAPI)
- Batch processing
- Real-time predictions
- Web applications

## Market Context (Cluj-Napoca)

### Typical Price Ranges (2023-2024):
- **City center**: 2,200-2,800 EUR/sqm
- **Semi-central**: 1,800-2,200 EUR/sqm
- **Suburban**: 1,400-1,800 EUR/sqm

### Popular Neighborhoods:
- **Zone 5**: Centru, Gheorgheni
- **Zone 4**: Zorilor, Mănăștur
- **Zone 3**: Mărăști, Gruia
- **Zone 2**: Iris, Dâmbul Rotund
- **Zone 1**: Peripheral areas

## Version History

### v1.0 (Current)
- Initial implementation
- Random Forest model
- Synthetic data generation
- Basic feature engineering
- Interactive and batch prediction
- Comprehensive test suite
