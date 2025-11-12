"""
Cluj Flat Price Prediction Model
This module trains a machine learning model to predict flat prices in Cluj-Napoca.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class FlatPricePredictor:
    """
    A machine learning model for predicting flat prices in Cluj-Napoca.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the predictor.
        
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError("model_type must be 'random_forest' or 'gradient_boosting'")
        
        self.feature_names = None
        
    def prepare_features(self, df):
        """
        Prepare features for modeling.
        
        Args:
            df: DataFrame with flat data
            
        Returns:
            X: Feature matrix
            feature_names: List of feature names
        """
        features = [
            'area_sqm', 'rooms', 'bathrooms', 'floor', 'total_floors',
            'year_built', 'neighborhood_zone', 'distance_to_center_km',
            'has_parking', 'has_balcony', 'condition'
        ]
        
        X = df[features].copy()
        
        # Create additional features
        X['age'] = 2024 - X['year_built']
        X['price_per_room'] = X['area_sqm'] / X['rooms']
        X['is_top_floor'] = (X['floor'] == X['total_floors'] - 1).astype(int)
        X['is_ground_floor'] = (X['floor'] == 0).astype(int)
        
        # Drop year_built as we now have age
        X = X.drop('year_built', axis=1)
        
        return X, X.columns.tolist()
    
    def train(self, df, test_size=0.2):
        """
        Train the model on the data.
        
        Args:
            df: DataFrame with flat data including 'price_eur' target
            test_size: Proportion of data to use for testing
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Prepare features
        X, self.feature_names = self.prepare_features(df)
        y = df['price_eur'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train': {
                'mae': mean_absolute_error(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'r2': r2_score(y_train, y_train_pred)
            },
            'test': {
                'mae': mean_absolute_error(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'r2': r2_score(y_test, y_test_pred)
            }
        }
        
        # Store test predictions for analysis
        self.y_test = y_test
        self.y_test_pred = y_test_pred
        
        return metrics
    
    def predict(self, flat_data):
        """
        Predict price for a flat or multiple flats.
        
        Args:
            flat_data: DataFrame or dict with flat characteristics
            
        Returns:
            Predicted price(s) in EUR
        """
        if isinstance(flat_data, dict):
            flat_data = pd.DataFrame([flat_data])
        
        X, _ = self.prepare_features(flat_data)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None
    
    def plot_feature_importance(self, save_path=None):
        """
        Plot feature importance.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        importance_df = self.get_feature_importance()
        if importance_df is None:
            print("Feature importance not available for this model type")
            return
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance for Flat Price Prediction')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        plt.close()
    
    def plot_predictions(self, save_path=None):
        """
        Plot actual vs predicted prices.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_test_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Price (EUR)')
        plt.ylabel('Predicted Price (EUR)')
        plt.title('Actual vs Predicted Flat Prices')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        plt.close()
    
    def save_model(self, filepath='flat_price_model.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath='flat_price_model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            FlatPricePredictor instance with loaded model
        """
        model_data = joblib.load(filepath)
        
        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}")
        return predictor


def main():
    """
    Main function to train and evaluate the model.
    """
    # Load data
    print("Loading data...")
    df = pd.read_csv('cluj_flat_prices.csv')
    print(f"Loaded {len(df)} records")
    
    # Train Random Forest model
    print("\n" + "="*50)
    print("Training Random Forest Model")
    print("="*50)
    rf_predictor = FlatPricePredictor(model_type='random_forest')
    rf_metrics = rf_predictor.train(df)
    
    print("\nRandom Forest Results:")
    print(f"Train MAE: {rf_metrics['train']['mae']:,.0f} EUR")
    print(f"Train RMSE: {rf_metrics['train']['rmse']:,.0f} EUR")
    print(f"Train R²: {rf_metrics['train']['r2']:.4f}")
    print(f"\nTest MAE: {rf_metrics['test']['mae']:,.0f} EUR")
    print(f"Test RMSE: {rf_metrics['test']['rmse']:,.0f} EUR")
    print(f"Test R²: {rf_metrics['test']['r2']:.4f}")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    importance_df = rf_predictor.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))
    
    # Save plots
    rf_predictor.plot_feature_importance('feature_importance.png')
    rf_predictor.plot_predictions('predictions_plot.png')
    
    # Save model
    rf_predictor.save_model('flat_price_model.pkl')
    
    # Example predictions
    print("\n" + "="*50)
    print("Example Predictions")
    print("="*50)
    
    example_flats = [
        {
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
        },
        {
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
        },
        {
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
    ]
    
    for i, flat in enumerate(example_flats, 1):
        price = rf_predictor.predict(flat)[0]
        print(f"\nFlat {i}:")
        print(f"  Area: {flat['area_sqm']} sqm, Rooms: {flat['rooms']}, "
              f"Year: {flat['year_built']}, Zone: {flat['neighborhood_zone']}")
        print(f"  Predicted Price: {price:,.0f} EUR")
        print(f"  Price per sqm: {price/flat['area_sqm']:,.0f} EUR/sqm")


if __name__ == "__main__":
    main()
