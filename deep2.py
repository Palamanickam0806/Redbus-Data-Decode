import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SeatCountPredictor:
    def __init__(self, model_type='GRU', sequence_length=16):
        """
        Initialize the seat count predictor
        
        Parameters:
        - model_type: 'GRU' or 'LSTM'
        - sequence_length: Number of time steps (DBD 30 to 15 = 16 steps)
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def load_and_preprocess_data(self, file_path='train.csv'):
        print("Loading data...")
        df = pd.read_csv(file_path)
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Display basic info
        print(f"\nData info:")
        print(f"Date range: {df['doj'].min()} to {df['doj'].max()}")
        print(f"DBD range: {df['dbd'].min()} to {df['dbd'].max()}")
        print(f"Unique routes: {df[['doj', 'srcid', 'destid']].drop_duplicates().shape[0]}")
        
        # Sort data properly
        df = df.sort_values(['doj', 'srcid', 'destid', 'dbd'], ascending=[True, True, True, False])
        
        # Encode categorical features
        categorical_cols = ['srcid', 'destid', 'srcid_region', 'destid_region', 'srcid_tier', 'destid_tier']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Feature engineering
        # df['days_to_journey'] = df['dbd']
        df['booking_velocity'] = df['cumsum_seatcount'] / (31 - df['dbd'])  # Booking rate
        df['search_to_booking_ratio'] = df['cumsum_searchcount'] / (df['cumsum_seatcount'] + 1)
        
        # Define feature columns
        self.feature_columns = [
            'dbd', 'cumsum_seatcount', 'cumsum_searchcount', 'booking_velocity', 'search_to_booking_ratio'
        ]
        
        # Add encoded categorical features
        for col in categorical_cols:
            if col in df.columns:
                self.feature_columns.append(f'{col}_encoded')
        
        print(f"\nFeature columns: {self.feature_columns}")
        
        self.df = df
        return df
    
    def create_sequences(self, df):
        """Create sequences for time series prediction"""
        print("Creating sequences...")
        
        sequences = []
        targets = []
        route_info = []
        
        # Group by route
        for (doj, srcid, destid), group in df.groupby(['doj', 'srcid', 'destid']):
            # Sort by DBD descending (30 to 15)
            group_sorted = group.sort_values('dbd', ascending=False)
            
            # Check if we have complete sequence
            if len(group_sorted) == self.sequence_length:
                # Extract features for the sequence
                sequence_features = group_sorted[self.feature_columns].values
                
                # Target is the final seat count
                target = group_sorted['final_seatcount'].iloc[0]
                
                sequences.append(sequence_features)
                targets.append(target)
                route_info.append((doj, srcid, destid))
        
        print(f"Created {len(sequences)} sequences")
        print(f"Sequence shape: {np.array(sequences).shape}")
        
        return np.array(sequences), np.array(targets), route_info
    
    def prepare_data_for_training(self,  val_size=0.2):
        """Prepare data for training with proper scaling"""
        print("Preparing data for training...")
        
        X, y, route_info = self.create_sequences(self.df)
        
        # Split data (route-wise to avoid data leakage)
        route_indices = np.arange(len(X))
        
        # First split: train+val vs test
        train_idx, val_idx = train_test_split(
            route_indices, test_size=val_size, random_state=42
        )
        
        
        # Split the data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"Train set: {X_train.shape[0]} sequences")
        print(f"Validation set: {X_val.shape[0]} sequences")
        
        # Scale features
        # Reshape for scaling (combine batch and sequence dimensions)
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
        
        # Fit scaler on training data
        self.scaler_features.fit(X_train_reshaped)
        
        # Transform all datasets
        X_train_scaled = self.scaler_features.transform(X_train_reshaped).reshape(X_train.shape)
        X_val_scaled = self.scaler_features.transform(X_val_reshaped).reshape(X_val.shape)
 
        
        # Scale targets
        y_train_scaled = self.scaler_target.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_target.transform(y_val.reshape(-1, 1)).flatten()

        
        self.X_train, self.X_val = X_train_scaled, X_val_scaled
        self.y_train, self.y_val = y_train_scaled, y_val_scaled
        self.y_train_orig, self.y_val_orig = y_train, y_val
        
        return (X_train_scaled, X_val_scaled, 
                y_train_scaled, y_val_scaled)
    
    def build_model(self, units=128, dropout_rate=0.3, learning_rate=0.001, n_layers=4):
        print(f"Building {self.model_type} model with {n_layers} RNN layers...")

        model = Sequential()

        # Add the first layer with input shape
        RNNLayer = LSTM if self.model_type == 'LSTM' else GRU
        model.add(RNNLayer(units, return_sequences=True, 
                        input_shape=(self.sequence_length, len(self.feature_columns))))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # Add intermediate layers
        for _ in range(n_layers - 2):
            model.add(RNNLayer(units, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # Add final RNN layer (no return_sequences)
        model.add(RNNLayer(units // 2, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # Dense layers
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout_rate / 2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))  # Output layer for regression

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        print(model.summary())
        self.model = model
        return model

    
    def train_model(self, epochs=100, batch_size=16, patience=15):
        """Train the model"""
        print("Training model...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            ModelCheckpoint('best_model_2.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Train the model
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        return history
    
    def evaluate_model(self):
        """Evaluate the model and calculate RMSE"""
        print("Evaluating model...")
        
        # Make predictions
        y_train_pred_scaled = self.model.predict(self.X_train)
        y_val_pred_scaled = self.model.predict(self.X_val)
        
        # Inverse transform predictions
        y_train_pred = self.scaler_target.inverse_transform(y_train_pred_scaled).flatten()
        y_val_pred = self.scaler_target.inverse_transform(y_val_pred_scaled).flatten()
        
        # Calculate RMSE
        train_rmse = np.sqrt(mean_squared_error(self.y_train_orig, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(self.y_val_orig, y_val_pred))
        
        print(f"\nRMSE Results:")
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        
        # Store predictions for analysis
        self.predictions = {
            'train': {'actual': self.y_train_orig, 'predicted': y_train_pred},
            'val': {'actual': self.y_val_orig, 'predicted': y_val_pred}
        }
        
        return {'train_rmse': train_rmse, 'val_rmse': val_rmse}
    
    def plot_results(self):
        """Plot training history and prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training history
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE history
        axes[0, 1].plot(self.history.history['mae'], label='Train MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        

        
        plt.tight_layout()
        # plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='seat_prediction_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

def main():
    """Main function to run the training pipeline"""
    print("="*50)
    print("SEAT COUNT PREDICTION MODEL TRAINING")
    print("="*50)
    
    # Initialize predictor
    predictor = SeatCountPredictor(model_type='GRU', sequence_length=16)
    
    # Load and preprocess data
    df = predictor.load_and_preprocess_data('train.csv')
    
    # Prepare data for training
    predictor.prepare_data_for_training( val_size=0.2)
    
    # Build model
    predictor.build_model(units=128, dropout_rate=0.3, learning_rate=0.001 , n_layers=4)
    
    # Train model
    history = predictor.train_model(epochs=100, batch_size=32, patience=15)
    
    # Evaluate model
    results = predictor.evaluate_model()
    
    # Plot results
    predictor.plot_results()
    
    # Save model
    predictor.save_model('seat_prediction_model.h5')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("Model saved as 'seat_prediction_model.h5'")
    print("Results plotted in 'model_results.png'")

if __name__ == "__main__":
    main()


# RMSE Results:
# Train RMSE: 477.6501
# Validation RMSE: 530.9964