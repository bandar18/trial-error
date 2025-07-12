import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class TradingModel:
    def __init__(self, train_days=42, val_days=7, n_splits=5):
        """
        Initialize the trading model.
        
        Args:
            train_days: Number of days for training
            val_days: Number of days for validation
            n_splits: Number of splits for TimeSeriesSplit
        """
        self.train_days = train_days
        self.val_days = val_days
        self.n_splits = n_splits
        self.base_model = None
        self.meta_model = None
        self.feature_names = None
        self.scaler = None
        
    def prepare_time_series_split(self, features, labels, prices):
        """
        Prepare data for time series cross-validation.
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            prices: Price Series
        
        Returns:
            List of train/val splits
        """
        print("Preparing time series splits...")
        
        # Convert to 5-minute periods
        minutes_per_day = 24 * 12  # 12 5-minute periods per hour
        train_periods = self.train_days * minutes_per_day
        val_periods = self.val_days * minutes_per_day # Corrected from minutes_periods to minutes_per_day
        
        splits = []
        total_samples = len(features)
        
        for i in range(self.n_splits):
            # Calculate split indices
            val_start = total_samples - val_periods - (self.n_splits - i - 1) * val_periods
            val_end = val_start + val_periods
            train_end = val_start
            
            if val_start < train_periods:
                continue  # Skip if not enough data
                
            train_start = max(0, val_start - train_periods)
            
            # Create splits
            train_idx = list(range(train_start, train_end))
            val_idx = list(range(val_start, val_end))
            
            splits.append((train_idx, val_idx))
        
        print(f"Created {len(splits)} time series splits")
        return splits
    
    def train_base_model(self, features, labels, splits):
        """
        Train the base LightGBM model using time series cross-validation.
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            splits: List of train/val splits
        
        Returns:
            Trained base model and validation predictions
        """
        print("Training base LightGBM model...")
        
        # LightGBM parameters
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        all_val_preds = []
        all_val_probs = []
        all_val_labels = []
        
        for i, (train_idx, val_idx) in enumerate(splits):
            print(f"  Split {i+1}/{len(splits)}")
            
            # Prepare data for this split
            X_train = features.iloc[train_idx]
            y_train = labels.iloc[train_idx]
            X_val = features.iloc[val_idx]
            y_val = labels.iloc[val_idx]
            
            # Convert labels to 0, 1, 2 for LightGBM
            y_train_encoded = y_train + 1  # -1, 0, 1 -> 0, 1, 2
            y_val_encoded = y_val + 1
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train_encoded)
            val_data = lgb.Dataset(X_val, label=y_val_encoded, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(stopping_rounds=10)]
            )
            
            # Get predictions
            val_probs = model.predict(X_val, num_iteration=model.best_iteration)
            val_preds = np.argmax(val_probs, axis=1) - 1  # Convert back to -1, 0, 1
            
            # Store results
            all_val_preds.extend(val_preds)
            all_val_probs.extend(val_probs)
            all_val_labels.extend(y_val.values)
            
            # Store the last model as the final model
            if i == len(splits) - 1:
                self.base_model = model
                self.feature_names = features.columns.tolist()
        
        # Convert to arrays
        all_val_preds = np.array(all_val_preds)
        all_val_probs = np.array(all_val_probs)
        all_val_labels = np.array(all_val_labels)
        
        # Calculate base model performance
        base_accuracy = accuracy_score(all_val_labels, all_val_preds)
        base_balanced_accuracy = balanced_accuracy_score(all_val_labels, all_val_preds)
        
        print(f"Base model validation accuracy: {base_accuracy:.4f}")
        print(f"Base model balanced accuracy: {base_balanced_accuracy:.4f}")
        
        return all_val_preds, all_val_probs, all_val_labels
    
    def create_meta_features(self, base_probs, features, labels, prices):
        """
        Create meta-features for meta-labeling.
        
        Args:
            base_probs: Base model probabilities
            features: Original features
            labels: True labels
            prices: Price series
        
        Returns:
            Meta-features DataFrame
        """
        print("Creating meta-features...")
        
        meta_features = pd.DataFrame(index=features.index)
        
        # Base model probabilities
        meta_features['base_prob_neg'] = base_probs[:, 0]  # P(y=-1)
        meta_features['base_prob_zero'] = base_probs[:, 1]  # P(y=0)
        meta_features['base_prob_pos'] = base_probs[:, 2]  # P(y=1)
        
        # Base model confidence
        meta_features['base_confidence'] = np.max(base_probs, axis=1)
        
        # Market context features
        meta_features['price_volatility'] = prices.pct_change().rolling(20).std()
        meta_features['price_trend'] = prices.rolling(20).mean().pct_change()
        
        # Volume features
        if 'SPY_volume_zscore' in features.columns:
            meta_features['volume_zscore'] = features['SPY_volume_zscore']
        
        # RSI features
        if 'SPY_rsi' in features.columns:
            meta_features['rsi'] = features['SPY_rsi']
            meta_features['rsi_extreme'] = ((features['SPY_rsi'] > 70) | (features['SPY_rsi'] < 30)).astype(int)
        
        # MACD features
        if 'macd' in features.columns:
            meta_features['macd'] = features['macd']
            meta_features['macd_signal'] = features['macd_signal']
            meta_features['macd_histogram'] = features['macd_histogram']
        
        # VIX features
        if 'vix_change' in features.columns:
            meta_features['vix_change'] = features['vix_change']
            meta_features['vix_volatility'] = features['vix_change'].rolling(20).std()
        
        # DXY features
        if 'dxy_change' in features.columns:
            meta_features['dxy_change'] = features['dxy_change']
        
        # Clean up
        meta_features = meta_features.fillna(method='ffill').fillna(0)
        
        print(f"Created {len(meta_features.columns)} meta-features")
        return meta_features
    
    def train_meta_model(self, meta_features, base_preds, true_labels):
        """
        Train the meta-model for meta-labeling.
        
        Args:
            meta_features: Meta-features DataFrame
            base_preds: Base model predictions
            true_labels: True labels
        
        Returns:
            Trained meta-model
        """
        print("Training meta-model...")
        
        # Create meta-labels: 1 if base prediction is correct, 0 otherwise
        meta_labels = (base_preds == true_labels).astype(int)
        
        # Remove samples where base model predicted 0 (no trade)
        valid_mask = base_preds != 0
        X_meta = meta_features[valid_mask]
        y_meta = meta_labels[valid_mask]
        
        if len(X_meta) == 0:
            print("Warning: No valid samples for meta-model training")
            return None
        
        # Train RandomForest meta-model
        self.meta_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.meta_model.fit(X_meta, y_meta)
        
        # Evaluate meta-model
        meta_preds = self.meta_model.predict(X_meta)
        meta_accuracy = accuracy_score(y_meta, meta_preds)
        
        print(f"Meta-model accuracy: {meta_accuracy:.4f}")
        print(f"Meta-model feature importance:")
        for feature, importance in zip(X_meta.columns, self.meta_model.feature_importances_):
            print(f"  {feature}: {importance:.4f}")
        
        return self.meta_model
    
    def generate_signals(self, features, meta_features, threshold=0.55):
        """
        Generate trading signals based on base model and meta-model.
        
        Args:
            features: Feature DataFrame
            meta_features: Meta-features DataFrame
            threshold: Confidence threshold for meta-model
        
        Returns:
            Series with trading signals (-1, 0, 1)
        """
        print("Generating trading signals...")
        
        if self.base_model is None or self.meta_model is None:
            raise ValueError("Models must be trained before generating signals")
        
        # Get base model predictions and probabilities
        base_probs = self.base_model.predict(features)
        base_preds = np.argmax(base_probs, axis=1) - 1  # Convert to -1, 0, 1
        
        # Get meta-model predictions
        meta_probs = self.meta_model.predict_proba(meta_features)
        meta_confidence = meta_probs[:, 1]  # P(correct)
        
        # Generate signals
        signals = pd.Series(0, index=features.index)
        
        # Long signal: base predicts +1 and meta confidence > threshold
        long_mask = (base_preds == 1) & (meta_confidence > threshold)
        signals[long_mask] = 1
        
        # Short signal: base predicts -1 and meta confidence > threshold
        short_mask = (base_preds == -1) & (meta_confidence > threshold)
        signals[short_mask] = -1
        
        # Signal statistics
        signal_counts = signals.value_counts()
        print(f"Signal distribution: {signal_counts.to_dict()}")
        
        return signals
    
    def train(self, features, labels, prices):
        """
        Complete training pipeline.
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            prices: Price Series
        
        Returns:
            Trained models and validation results
        """
        print("Starting model training pipeline...")
        
        # Prepare time series splits
        splits = self.prepare_time_series_split(features, labels, prices)
        
        if len(splits) == 0:
            raise ValueError("Not enough data for time series splits")
        
        # Train base model
        base_preds, base_probs, true_labels = self.train_base_model(features, labels, splits)
        
        # Create meta-features
        meta_features = self.create_meta_features(base_probs, features, labels, prices)
        
        # Train meta-model
        meta_model = self.train_meta_model(meta_features, base_preds, true_labels)
        
        # Generate signals for validation
        val_signals = self.generate_signals(features, meta_features)
        
        return {
            'base_model': self.base_model,
            'meta_model': self.meta_model,
            'base_preds': base_preds,
            'base_probs': base_probs,
            'true_labels': true_labels,
            'signals': val_signals,
            'meta_features': meta_features
        }

if __name__ == "__main__":
    # Test the model
    from data import download_data
    from features import prepare_features_and_labels
    
    # Download and prepare data
    data = download_data(days=14)  # Test with smaller dataset
    features, labels, prices = prepare_features_and_labels(data)
    
    # Train model
    model = TradingModel(train_days=7, val_days=3, n_splits=2)
    results = model.train(features, labels, prices)
    
    print("Training completed successfully!")