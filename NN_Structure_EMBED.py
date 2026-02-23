import json
import os
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import time, pickle
import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Add, Input, Embedding, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.utils import plot_model

# Import the pre_train data
# from Pre_Train_data_Ver2_1 import FETCH_ERP_SCREW # Removed for GitHub

class NaNInfMonitor(Callback):
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        for metric, value in logs.items():
            if np.isnan(value) or np.isinf(value):
                print(f"Warning: {metric} is {value} at batch {batch}. Stopping training.")
                self.model.stop_training = True

class PackingRatioMetric(Callback):
    def __init__(self, X_train, y_train, screw_volume, scale_target=False, scaler=None):
        super(PackingRatioMetric, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.screw_volume = screw_volume
        self.scale_target = scale_target
        self.scaler = scaler

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_train, verbose=2)
        if self.scale_target and self.scaler is not None:
            preds = self.scaler.inverse_transform(preds)
            y_true = self.scaler.inverse_transform(self.y_train.reshape(-1, 1))
        else:
            y_true = self.y_train
        for i in range(5):
            print(f"Epoch {epoch+1}, Sample {i}: Pred = {preds[i][0]:.2f}, True = {y_true[i][0]:.2f}")
        mean_packing_ratio = np.mean(preds.flatten()) * 100
        logs["packing_ratio"] = mean_packing_ratio

class Neural_Network_Arch:
    def __init__(self, config_path='config.json', load_erp_data=True):
        self.today = datetime.date.today()
        self.SAVE_FILE = os.path.join(os.path.dirname(__file__), "models")
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.target_scaler = None
        self.container_volumes = None
        self.config = self._load_config(os.path.join(self.SAVE_FILE, config_path))

        self.Train_data = None

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config['model_config']
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            raise

    def _build_model(self, input_dim_numeric, categorical_dims):
        try:
            # Numeric input
            numeric_input = Input(shape=(input_dim_numeric,), name='numeric_input')
            
            # Categorical inputs with embedding layers
            embedding_inputs = []
            embedding_layers = []
            for feature, n_unique in categorical_dims.items():
                input_layer = Input(shape=(1,), name=f'screw_input_{feature}')
                embedding_dim = min(50, (n_unique + 1) // 2)  # Heuristic for embedding dimension
                embedding_layer = Embedding(input_dim=n_unique, output_dim=embedding_dim, name=f'embedding_{feature}')(input_layer)
                embedding_layer = Flatten()(embedding_layer)
                embedding_inputs.append(input_layer)
                embedding_layers.append(embedding_layer)
            
            # Concatenate numeric and embedding layers
            x = Concatenate()([numeric_input] + embedding_layers)
            
            # Dense layers with residual connections
            x = Dense(128, activation='relu', 
                     kernel_regularizer=tf.keras.regularizers.l2(self.config['l2_regularization']))(x)
            x = BatchNormalization()(x)
            x = Dropout(self.config['dropout_rate'])(x)
            x1 = Dense(64, activation='relu')(x)
            x1 = BatchNormalization()(x1)
            x1 = Dense(64, activation='relu')(x1)
            x1 = BatchNormalization()(x1)
            x_proj1 = Dense(64, activation='linear')(x)
            x = Add()([x_proj1, x1])
            x = Dropout(self.config['dropout_rate'])(x)
            x2 = Dense(32, activation='relu')(x)
            x2 = BatchNormalization()(x2)
            x2 = Dense(32, activation='relu')(x2)
            x2 = BatchNormalization()(x2)
            x_proj2 = Dense(32, activation='linear')(x)
            x = Add()([x_proj2, x2])
            x = Dropout(self.config['dropout_rate'])(x)
            outputs = Dense(1, activation='linear')(x)
            
            # Define model
            model = Model(inputs=[numeric_input] + embedding_inputs, outputs=outputs)
            
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.config['initial_learning_rate']
            )
            
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.Huber(delta=self.config['huber_delta']),
                metrics=['mae', 'mse']
            )
            
            return model
        
        except Exception as e:
            print(f"Error in _build_model: {str(e)}")
            raise

    def _prepare_feature_NN(self, save_files=True):
        numeric_features = ['Screw volume in the box', 'Diameter', 'Length']
        categorical_features = ["pdc_4", "pdc_5", 'Screw_Type', 'Head_Type']

        # ---- 1. Handle missing values ----
        if self.Train_data[numeric_features].isna().any().any() or np.isinf(self.Train_data[numeric_features]).any().any():
            print("Warning: NaN or Inf values detected in numeric features before preprocessing.")
        if self.Train_data[categorical_features].isna().any().any():
            print("Warning: NaN values detected in categorical features before preprocessing.")

        self.Train_data[numeric_features] = self.Train_data[numeric_features].fillna(self.Train_data[numeric_features].median())
        self.Train_data[categorical_features] = self.Train_data[categorical_features].fillna('Unknown')


        # ---- 2. Numeric scaling ----
        if self.scaler is None:
            self.scaler = RobustScaler()
            X_numeric = self.scaler.fit_transform(self.Train_data[numeric_features])
        else:
            X_numeric = self.scaler.transform(self.Train_data[numeric_features])

        # ---- 3. Categorical integer encoding for embeddings ----
        X_categorical = []
        if not hasattr(self, "label_encoders") or self.label_encoders is None:
            self.label_encoders = {}
            for col in categorical_features:
                unique_values = self.Train_data[col].unique()
                self.label_encoders[col] = {val: idx for idx, val in enumerate(unique_values)}
                mapped_values = self.Train_data[col].map(self.label_encoders[col]).fillna(len(self.label_encoders[col])).astype(int)
                X_categorical.append(mapped_values.values)
        else:
            for col in categorical_features:
                mapped_values = self.Train_data[col].map(self.label_encoders[col]).fillna(len(self.label_encoders[col])).astype(int)
                X_categorical.append(mapped_values.values)


        # ---- 4. Target variable ----
        y = self.Train_data['Packing_Ratio'].values if 'Packing_Ratio' in self.Train_data else None
        if self.config.get('scale_target', False) and y is not None:
            if self.target_scaler is None:
                self.target_scaler = RobustScaler()
                y = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            else:
                y = self.target_scaler.transform(y.reshape(-1, 1)).flatten()

        # ---- 5. Save scaler, target_scaler, and encoders ----
        if save_files:
            os.makedirs(self.SAVE_FILE, exist_ok=True)
            with open(os.path.join(self.SAVE_FILE, f'scaler_{self.today}.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
            if self.config.get('scale_target', False) and y is not None:
                with open(os.path.join(self.SAVE_FILE, f'target_scaler_{self.today}.pkl'), 'wb') as f:
                    pickle.dump(self.target_scaler, f)
            with open(os.path.join(self.SAVE_FILE, f'label_encoders_{self.today}.pkl'), 'wb') as f:
                pickle.dump(self.label_encoders, f)

        # ---- 6. Check for NaN or Inf ----
        if np.isnan(X_numeric).any() or np.isinf(X_numeric).any():
            print("Warning: NaN or Inf values detected in numeric features.")
        for x_cat in X_categorical:
            if np.isnan(x_cat).any() or np.isinf(x_cat).any():
                print("Warning: NaN or Inf values detected in categorical features.")
        if y is not None and (np.isnan(y).any() or np.isinf(y).any()):
            print("Warning: NaN or Inf values detected in target.")

        # ---- 7. Debug statistics ----
        if y is not None:
            print(self.Train_data[['Screw volume in the box', 'decision box/master volume']].describe())
            print("Expected Packing Ratio (Screw volume / Box volume):")
            print((self.Train_data['Screw volume in the box'] / self.Train_data['decision box/master volume']).describe())

        # ---- 8. Return ----
        # Return numeric features, list of categorical integer arrays, target, scaler, encoders, original screw volume
        return [X_numeric] + X_categorical, y, self.scaler, self.label_encoders, self.Train_data['Screw volume in the box'].values

    # _predict_container_volume method removed for GitHub (Training logic)

if __name__ == "__main__":
    start = time.time()
    bot = Neural_Network_Arch()
    bot._predict_container_volume()
    end = time.time()
    print(f"耗時{end - start}")