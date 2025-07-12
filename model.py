"""
Machine learning model module.
Implements LightGBM base model with TimeSeriesSplit and RandomForest meta-labeling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class BaseLightGBMModel:
    """Base LightGBM model for directional prediction."""
    
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        self.model = None
        self.feature_names = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the LightGBM model."""
        print("Training base LightGBM model...")
        
        # Convert labels to 0, 1, 2 for multiclass
        y_train_mapped = self._map_labels(y_train)
        
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train_mapped)
        
        # Validation data
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            y_val_mapped = self._map_labels(y_val)
            val_data = lgb.Dataset(X_val, label=y_val_mapped)
            valid_sets.append(val_data)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=100,
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        self.feature_names = X_train.columns.tolist()
        print(f"Model trained with {len(self.feature_names)} features")
        
    def predict(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get probabilities
        probs = self.model.predict(X)
        
        # Convert to DataFrame
        prob_df = pd.DataFrame(probs, columns=['prob_-1', 'prob_0', 'prob_1'])
        
        # Get class predictions
        predictions = np.argmax(probs, axis=1)
        predictions = self._unmap_labels(predictions)
        
        return predictions, prob_df
    
    def _map_labels(self, y):
        """Map labels from {-1, 0, 1} to {0, 1, 2}."""
        return y + 1
    
    def _unmap_labels(self, y):
        """Map labels from {0, 1, 2} back to {-1, 0, 1}."""
        return y - 1
    
    def get_feature_importance(self):
        """Get feature importance."""
        if self.model is None:
            return None
        
        importance = self.model.feature_importance()
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

class MetaLabelingModel:
    """Meta-labeling model using RandomForest."""
    
    def __init__(self, params=None):
        self.params = params or {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        self.model = RandomForestClassifier(**self.params)
        self.feature_names = None
        
    def prepare_meta_dataset(self, X, y, base_predictions, base_probabilities):
        """Prepare meta-labeling dataset."""
        print("Preparing meta-labeling dataset...")
        
        # Meta-label: 1 if base prediction == true label, 0 otherwise
        meta_labels = (base_predictions == y).astype(int)
        
        # Meta-features: base probabilities + market context
        meta_features = base_probabilities.copy()
        
        # Add market context features
        context_features = self._extract_context_features(X)
        meta_features = pd.concat([meta_features, context_features], axis=1)
        
        print(f"Meta-dataset prepared: {len(meta_features.columns)} features, {len(meta_labels)} samples")
        print(f"Meta-label distribution: {pd.Series(meta_labels).value_counts().to_dict()}")
        
        return meta_features, meta_labels
    
    def _extract_context_features(self, X):
        """Extract market context features."""
        context_features = pd.DataFrame(index=X.index)
        
        # Volatility context
        vol_cols = [col for col in X.columns if 'volatility' in col.lower()]
        if vol_cols:
            context_features['avg_volatility'] = X[vol_cols].mean(axis=1)
        
        # Volume context
        vol_cols = [col for col in X.columns if 'volume' in col.lower()]
        if vol_cols:
            context_features['avg_volume'] = X[vol_cols].mean(axis=1)
        
        # Price momentum
        return_cols = [col for col in X.columns if 'return' in col.lower()]
        if return_cols:
            context_features['avg_return'] = X[return_cols].mean(axis=1)
        
        # RSI context
        rsi_cols = [col for col in X.columns if 'rsi' in col.lower()]
        if rsi_cols:
            context_features['avg_rsi'] = X[rsi_cols].mean(axis=1)
        
        # Fill missing values
        context_features = context_features.fillna(0)
        
        return context_features
    
    def train(self, meta_features, meta_labels):
        """Train the meta-labeling model."""
        print("Training meta-labeling model...")
        
        self.model.fit(meta_features, meta_labels)
        self.feature_names = meta_features.columns.tolist()
        
        print(f"Meta-model trained with {len(self.feature_names)} features")
        
    def predict_proba(self, meta_features):
        """Predict probability of base model being correct."""
        return self.model.predict_proba(meta_features)[:, 1]  # Probability of class 1
    
    def get_feature_importance(self):
        """Get feature importance."""
        if self.model is None:
            return None
        
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

class TradingModel:
    """Complete trading model with base model and meta-labeling."""
    
    def __init__(self, base_params=None, meta_params=None):
        self.base_model = BaseLightGBMModel(base_params)
        self.meta_model = MetaLabelingModel(meta_params)
        
    def train_with_time_series_split(self, data, train_days=42, val_days=7):
        """Train model using TimeSeriesSplit."""
        print(f"Training with TimeSeriesSplit (train: {train_days} days, val: {val_days} days)")
        
        # Prepare features and labels
        feature_cols = [col for col in data.columns if col != 'label']
        X = data[feature_cols]
        y = data['label']
        
        # Calculate split points based on days
        total_periods = len(data)
        periods_per_day = total_periods // 56  # Approximate periods per day
        
        train_size = int(train_days * periods_per_day)
        val_size = int(val_days * periods_per_day)
        
        # Use only the most recent split for simplicity
        X_train = X.iloc[-train_size-val_size:-val_size]
        y_train = y.iloc[-train_size-val_size:-val_size]
        X_val = X.iloc[-val_size:]
        y_val = y.iloc[-val_size:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Train base model
        self.base_model.train(X_train, y_train, X_val, y_val)
        
        # Get base model predictions for meta-labeling
        base_pred_train, base_probs_train = self.base_model.predict(X_train)
        base_pred_val, base_probs_val = self.base_model.predict(X_val)
        
        # Prepare meta-labeling dataset
        meta_X_train, meta_y_train = self.meta_model.prepare_meta_dataset(
            X_train, y_train, base_pred_train, base_probs_train
        )
        
        # Train meta-labeling model
        self.meta_model.train(meta_X_train, meta_y_train)
        
        # Evaluate on validation set
        meta_X_val, meta_y_val = self.meta_model.prepare_meta_dataset(
            X_val, y_val, base_pred_val, base_probs_val
        )
        
        meta_probs_val = self.meta_model.predict_proba(meta_X_val)
        
        # Print evaluation metrics
        print("\nBase model performance:")
        print(f"Accuracy: {accuracy_score(y_val, base_pred_val):.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy_score(y_val, base_pred_val):.4f}")
        
        print("\nMeta-labeling performance:")
        print(f"Accuracy: {accuracy_score(meta_y_val, meta_probs_val > 0.5):.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy_score(meta_y_val, meta_probs_val > 0.5):.4f}")
        
        return {
            'base_model': self.base_model,
            'meta_model': self.meta_model,
            'validation_results': {
                'X_val': X_val,
                'y_val': y_val,
                'base_pred_val': base_pred_val,
                'base_probs_val': base_probs_val,
                'meta_probs_val': meta_probs_val
            }
        }
    
    def generate_signals(self, X, meta_threshold=0.55):
        """Generate trading signals."""
        # Get base predictions
        base_pred, base_probs = self.base_model.predict(X)
        
        # Get meta-features
        meta_features, _ = self.meta_model.prepare_meta_dataset(
            X, base_pred, base_pred, base_probs  # Use base_pred as dummy y
        )
        
        # Get meta-labeling probabilities
        meta_probs = self.meta_model.predict_proba(meta_features)
        
        # Generate final signals
        signals = np.zeros(len(X))
        
        # Long signals
        long_mask = (base_pred == 1) & (meta_probs > meta_threshold)
        signals[long_mask] = 1
        
        # Short signals
        short_mask = (base_pred == -1) & (meta_probs > meta_threshold)
        signals[short_mask] = -1
        
        # Flat otherwise (signals already initialized to 0)
        
        return signals, base_pred, base_probs, meta_probs

def train_trading_model(data, train_days=42, val_days=7):
    """Convenience function to train the complete trading model."""
    model = TradingModel()
    results = model.train_with_time_series_split(data, train_days, val_days)
    return model, results

if __name__ == "__main__":
    # Test the model
    from data import download_market_data
    from features import prepare_features_and_labels
    
    print("Testing trading model...")
    data, spreads = download_market_data(days=14)
    labeled_data = prepare_features_and_labels(data, spreads)
    
    model, results = train_trading_model(labeled_data, train_days=7, val_days=3)
    
    print("\nModel training completed!")
    print("Base model feature importance:")
    print(results['base_model'].get_feature_importance().head())
    
    print("\nMeta-model feature importance:")
    print(results['meta_model'].get_feature_importance().head())