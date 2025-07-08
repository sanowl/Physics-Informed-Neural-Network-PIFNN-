import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

class BatteryDataProcessor:
    """Data processor for extracting features from battery data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_ic_curve(self, voltage, current, time, gaussian_sigma=2):
        """Extract Incremental Capacity (IC) curve from charging data"""
        # Calculate capacity using current and time
        dt = np.diff(time)
        dq = current[1:] * dt
        capacity = np.cumsum(dq)
        
        # Calculate dQ/dV using finite difference
        dv = np.diff(voltage)
        # Avoid division by zero
        dv[dv == 0] = 1e-6
        ic = dq / dv
        
        # Apply Gaussian filtering to smooth the curve
        ic_smooth = gaussian_filter1d(ic, sigma=gaussian_sigma)
        
        return ic_smooth, capacity[:-1], voltage[1:-1]
    
    def extract_dt_curve(self, temperature, L=40):
        """Extract Differential Temperature (DT) curve"""
        dt_curve = np.zeros(len(temperature) - L)
        for i in range(len(temperature) - L):
            dt_curve[i] = (temperature[i + L] - temperature[i]) / L
        
        # Apply Gaussian filtering
        dt_smooth = gaussian_filter1d(dt_curve, sigma=2)
        return dt_smooth
    
    def extract_features(self, voltage, current, temperature, time):
        """Extract all features from battery data"""
        # Extract IC curve
        ic_curve, capacity, voltage_ic = self.extract_ic_curve(voltage, current, time)
        
        # Extract DT curve
        dt_curve = self.extract_dt_curve(temperature)
        
        # Extract features
        features = {}
        
        # Peak of IC curve (P-IC)
        features['peak_ic'] = np.max(ic_curve) if len(ic_curve) > 0 else 0
        
        # DT curve features
        if len(dt_curve) > 0:
            # Left peak of DT curve
            first_half = dt_curve[:len(dt_curve)//2]
            features['dt_left_peak'] = np.max(first_half) if len(first_half) > 0 else 0
            
            # Middle valley of DT curve
            middle_third = dt_curve[len(dt_curve)//3:2*len(dt_curve)//3]
            features['dt_middle_valley'] = np.min(middle_third) if len(middle_third) > 0 else 0
            
            # Voltage difference between left peak and middle valley
            features['dt_peak_valley_diff'] = features['dt_left_peak'] - features['dt_middle_valley']
        else:
            features['dt_left_peak'] = 0
            features['dt_middle_valley'] = 0
            features['dt_peak_valley_diff'] = 0
            
        return features

class PIFNN(nn.Module):
    """Physics-Informed Feedforward Neural Network"""
    
    def __init__(self, input_size=4, hidden_sizes=[8, 8, 8], output_size=1):
        super(PIFNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # SOH is between 0 and 1
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class PIFNNTrainer:
    """Trainer for Physics-Informed Neural Network"""
    
    def __init__(self, model, learning_rate=0.001, omega1=0.01, omega2=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.omega1 = omega1  # Weight for physics loss in training
        self.omega2 = omega2  # Weight for physics loss in testing
        self.train_losses = []
        self.physics_losses = []
        
    def physics_constraint_loss(self, features, predictions):
        """Calculate physics constraint loss based on P-IC monotonicity"""
        # Extract P-IC feature (assuming it's the first feature)
        p_ic = features[:, 0]
        
        # Calculate gradient of predictions with respect to P-IC
        # Use finite difference approximation
        if len(predictions) > 1:
            # Sort by P-IC to ensure proper ordering
            sorted_indices = torch.argsort(p_ic)
            p_ic_sorted = p_ic[sorted_indices]
            pred_sorted = predictions[sorted_indices]
            
            # Calculate differences
            delta_p_ic = p_ic_sorted[1:] - p_ic_sorted[:-1]
            delta_pred = pred_sorted[1:] - pred_sorted[:-1]
            
            # Avoid division by zero
            delta_p_ic = torch.clamp(delta_p_ic, min=1e-6)
            
            # Calculate gradient (should be positive for monotonic relationship)
            gradient = delta_pred / delta_p_ic
            
            # Physics constraint: gradient should be positive
            constraint_violation = torch.clamp(-gradient, min=0)
            physics_loss = torch.mean(constraint_violation ** 2)
        else:
            physics_loss = torch.tensor(0.0, requires_grad=True)
            
        return physics_loss
    
    def train_step(self, features, targets):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(features)
        
        # Data loss (MSE)
        data_loss = nn.MSELoss()(predictions.squeeze(), targets)
        
        # Physics loss
        physics_loss = self.physics_constraint_loss(features, predictions)
        
        # Total loss
        total_loss = data_loss + self.omega1 * physics_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), data_loss.item(), physics_loss.item()
    
    def train(self, train_loader, epochs=2000):
        """Train the model"""
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_data_loss = 0
            epoch_physics_loss = 0
            
            for features, targets in train_loader:
                loss, data_loss, physics_loss = self.train_step(features, targets)
                epoch_loss += loss
                epoch_data_loss += data_loss
                epoch_physics_loss += physics_loss
            
            avg_loss = epoch_loss / len(train_loader)
            avg_data_loss = epoch_data_loss / len(train_loader)
            avg_physics_loss = epoch_physics_loss / len(train_loader)
            
            self.train_losses.append(avg_data_loss)
            self.physics_losses.append(avg_physics_loss)
            
            if epoch % 200 == 0:
                print(f'Epoch {epoch}, Total Loss: {avg_loss:.6f}, '
                      f'Data Loss: {avg_data_loss:.6f}, Physics Loss: {avg_physics_loss:.6f}')
    
    def secondary_training(self, train_loader, test_loader, epochs=400):
        """Secondary training phase with test set physics constraints"""
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Training set loss
            for features, targets in train_loader:
                predictions = self.model(features)
                data_loss = nn.MSELoss()(predictions.squeeze(), targets)
                physics_loss_train = self.physics_constraint_loss(features, predictions)
                train_loss = data_loss + self.omega1 * physics_loss_train
                
                # Test set physics loss
                test_physics_loss = 0
                for test_features, _ in test_loader:
                    test_predictions = self.model(test_features)
                    test_physics_loss += self.physics_constraint_loss(test_features, test_predictions)
                
                test_physics_loss /= len(test_loader)
                
                # Total loss
                total_loss = train_loss + self.omega2 * test_physics_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
            
            if epoch % 100 == 0:
                print(f'Secondary Training Epoch {epoch}, Loss: {epoch_loss/len(train_loader):.6f}')

class BatterySOHEstimator:
    """Complete Battery SOH Estimation System"""
    
    def __init__(self):
        self.processor = BatteryDataProcessor()
        self.model = None
        self.trainer = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, battery_data):
        """Prepare data for training"""
        features_list = []
        soh_list = []
        
        for cycle_data in battery_data:
            # Extract features
            features = self.processor.extract_features(
                cycle_data['voltage'],
                cycle_data['current'],
                cycle_data['temperature'],
                cycle_data['time']
            )
            
            # Convert to list
            feature_vector = [
                features['peak_ic'],
                features['dt_left_peak'],
                features['dt_middle_valley'],
                features['dt_peak_valley_diff']
            ]
            
            features_list.append(feature_vector)
            soh_list.append(cycle_data['soh'])
        
        return np.array(features_list), np.array(soh_list)
    
    def train_model(self, features, soh_values, test_size=0.4, 
                   primary_epochs=1600, secondary_epochs=400):
        """Train the PIFNN model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, soh_values, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model and trainer
        self.model = PIFNN(input_size=4)
        self.trainer = PIFNNTrainer(self.model)
        
        # Primary training
        print("Starting primary training...")
        self.trainer.train(train_loader, epochs=primary_epochs)
        
        # Secondary training
        print("Starting secondary training...")
        self.trainer.secondary_training(train_loader, test_loader, epochs=secondary_epochs)
        
        # Evaluate
        train_predictions = self.predict(X_train_scaled)
        test_predictions = self.predict(X_test_scaled)
        
        train_mae = mean_absolute_error(y_train, train_predictions)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        
        print(f"\nTraining Results:")
        print(f"Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")
        
        return {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def predict(self, features):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            predictions = self.model(features)
            return predictions.numpy().flatten()
    
    def plot_results(self, results):
        """Plot training results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(self.trainer.train_losses)
        axes[0, 0].set_title('Training Data Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Physics loss
        axes[0, 1].plot(self.trainer.physics_losses)
        axes[0, 1].set_title('Physics Constraint Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Train predictions
        axes[1, 0].scatter(results['y_train'], results['train_predictions'], alpha=0.6)
        axes[1, 0].plot([0, 1], [0, 1], 'r--')
        axes[1, 0].set_title('Training Predictions')
        axes[1, 0].set_xlabel('True SOH')
        axes[1, 0].set_ylabel('Predicted SOH')
        axes[1, 0].grid(True)
        
        # Test predictions
        axes[1, 1].scatter(results['y_test'], results['test_predictions'], alpha=0.6)
        axes[1, 1].plot([0, 1], [0, 1], 'r--')
        axes[1, 1].set_title('Test Predictions')
        axes[1, 1].set_xlabel('True SOH')
        axes[1, 1].set_ylabel('Predicted SOH')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage and synthetic data generation
def generate_synthetic_battery_data(num_cycles=200, num_cells=3):
    """Generate synthetic battery data for demonstration"""
    battery_data = []
    
    for cell in range(num_cells):
        cell_data = []
        np.random.seed(42 + cell)
        
        for cycle in range(num_cycles):
            # Simulate battery degradation
            soh = 1.0 - (cycle / num_cycles) * 0.3 + np.random.normal(0, 0.01)
            soh = max(0.7, min(1.0, soh))  # Clamp between 0.7 and 1.0
            
            # Generate synthetic voltage, current, temperature, time data
            time_points = np.linspace(0, 3600, 100)  # 1 hour charging
            voltage = 3.0 + np.linspace(0, 1.2, 100) + np.random.normal(0, 0.01, 100)
            current = np.ones(100) * 2.0 + np.random.normal(0, 0.1, 100)
            temperature = 25 + np.sin(time_points / 1000) * 5 + np.random.normal(0, 0.5, 100)
            
            cycle_data = {
                'voltage': voltage,
                'current': current,
                'temperature': temperature,
                'time': time_points,
                'soh': soh
            }
            cell_data.append(cycle_data)
        
        battery_data.extend(cell_data)
    
    return battery_data

# Demonstration
if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic battery data...")
    battery_data = generate_synthetic_battery_data(num_cycles=100, num_cells=3)
    
    # Initialize estimator
    estimator = BatterySOHEstimator()
    
    # Prepare data
    print("Preparing data...")
    features, soh_values = estimator.prepare_data(battery_data)
    
    print(f"Data shape: {features.shape}")
    print(f"SOH range: {soh_values.min():.3f} - {soh_values.max():.3f}")
    
    # Train model
    print("Training PIFNN model...")
    results = estimator.train_model(features, soh_values)
    
    # Plot results
    estimator.plot_results(results)
    
    print("\nPIFNN Training Complete!")
    print(f"Final Test MAE: {results['test_mae']:.4f}")
    print(f"Final Test RMSE: {results['test_rmse']:.4f}")