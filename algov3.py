#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseball Statistics Prediction Algorithm

This module provides functionality to predict baseball player and team statistics
using Logistic Regression with Gradient Boosting based on historical performance data.
"""

import glob
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import os
import logging
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Union, Optional
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("baseball_predictions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

class BaseballPredictor:
    """
    A class for predicting baseball statistics using machine learning models.
    
    This class processes historical baseball data from batting, pitching, and game results
    to predict player and team performance metrics for future games.
    """
    
    def __init__(self, data_path: str = '.'):
        """
        Initialize the BaseballPredictor with data paths.
        
        Args:
            data_path (str): Path to the directory containing the CSV data files.
        """
        self.data_path = data_path
        self.batting_data = None
        self.pitching_data = None
        self.linescore_data = None
        self.batting_models = {}
        self.pitching_models = {}
        self.team_models = {}
        self.scaler = StandardScaler()
        self.logger = logger
        
    def load_data(self, target_date: Optional[str] = None) -> None:
        """
        Load the baseball data from CSV files.
        
        This method loads batting, pitching, and linescore data from their respective
        CSV files and performs initial data preprocessing. If target_date is provided,
        only data up to that date will be loaded.
        
        Args:
            target_date (Optional[str]): If provided, only load data up to this date
        
        Raises:
            FileNotFoundError: If any of the required CSV files cannot be found.
            ValueError: If the loaded data is empty or invalid.
        """
        try:
            # Load the data files
            batting_file = os.path.join(self.data_path, '2024_batting.csv')
            pitching_file = os.path.join(self.data_path, '2024_pitching.csv')
            linescore_file = os.path.join(self.data_path, '2024_linescore.csv')
            
            self.logger.info(f"Loading data from {batting_file}, {pitching_file}, and {linescore_file}")
            
            # Read the CSV files
            self.batting_data = pd.read_csv(batting_file)
            self.pitching_data = pd.read_csv(pitching_file)
            self.linescore_data = pd.read_csv(linescore_file)

            self.linescore_data = self.linescore_data[~self.linescore_data.iloc[:, 0].astype(str).str.startswith('WP')]
            self.linescore_data = self.linescore_data[~self.linescore_data.iloc[:, 0].astype(str).str.startswith('Winning')]

            # Remove the extra 0 from the Date column
            if 'Date' in self.batting_data.columns:
                self.batting_data['Date'] = self.batting_data['Date'].astype(str).str[:-1]
            if 'Date' in self.pitching_data.columns:
                self.pitching_data['Date'] = self.pitching_data['Date'].astype(str).str[:-1]
            if 'Date' in self.linescore_data.columns:
                self.linescore_data['Date'] = self.linescore_data['Date'].astype(str).str[:-1]
            
            self.batting_data['Date'] = pd.to_datetime(self.batting_data['Date'], errors='coerce')
            self.pitching_data['Date'] = pd.to_datetime(self.pitching_data['Date'], errors='coerce')
            self.linescore_data['Date'] = pd.to_datetime(self.linescore_data['Date'], errors='coerce')
            
            # Filter data by target date if provided
            if target_date:
                target_date_dt = pd.to_datetime(target_date)
                self.logger.info(f"Filtering data to include only entries up to {target_date}")
                
                self.batting_data = self.batting_data[self.batting_data['Date'] < target_date_dt]
                self.pitching_data = self.pitching_data[self.pitching_data['Date'] < target_date_dt]
                self.linescore_data = self.linescore_data[self.linescore_data['Date'] < target_date_dt]
                
                # Log the data size after filtering
                self.logger.info(f"After filtering: {len(self.batting_data)} batting records, " 
                                f"{len(self.pitching_data)} pitching records, "
                                f"{len(self.linescore_data)} game records")
            
            # Validate data
            self.validate_data()
            
            # Preprocess the data
            self.preprocess_data()
            
            self.logger.info("Data loaded and preprocessed successfully")
            
        except FileNotFoundError as e:
            self.logger.error(f"Error loading data: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"Invalid data: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during data loading: {e}")
            raise
    
    def validate_data(self) -> None:
        """
        Validate the loaded data for completeness and correctness.
        
        Raises:
            ValueError: If any data validation check fails.
        """
        # Check if dataframes are empty
        if self.batting_data.empty:
            raise ValueError("Batting data is empty")
        if self.pitching_data.empty:
            raise ValueError("Pitching data is empty")
        if self.linescore_data.empty:
            raise ValueError("Linescore data is empty")
        
        # Check for required columns in batting data
        required_batting_cols = ['Batting', 'Team', 'AB', 'R', 'H', 'RBI', 'Date', 'Home']
        missing_cols = [col for col in required_batting_cols if col not in self.batting_data.columns]
        if missing_cols:
            raise ValueError(f"Batting data missing required columns: {missing_cols}")
        
        # Check for required columns in pitching data
        required_pitching_cols = ['Pitching', 'Team', 'IP', 'H', 'R', 'ER', 'Date', 'Home']
        missing_cols = [col for col in required_pitching_cols if col not in self.pitching_data.columns]
        if missing_cols:
            raise ValueError(f"Pitching data missing required columns: {missing_cols}")
        
        # Check for required columns in linescore data
        required_linescore_cols = ['AwayTeam', 'HomeTeam', 'R', 'Date']
        missing_cols = [col for col in required_linescore_cols if col not in self.linescore_data.columns]
        if missing_cols:
            raise ValueError(f"Linescore data missing required columns: {missing_cols}")
        
        # Check data types
        try:
            pd.to_datetime(self.batting_data['Date'])
            pd.to_datetime(self.pitching_data['Date'])
            pd.to_datetime(self.linescore_data['Date'])
        except:
            raise ValueError("Date columns cannot be converted to datetime format")

    def preprocess_data(self) -> None:
        """
        Preprocess the loaded data for model training.
        
        This includes handling missing values, feature engineering, and data transformation.
        """
        # Convert date columns to datetime
        self.batting_data['Date'] = pd.to_datetime(self.batting_data['Date'])
        self.pitching_data['Date'] = pd.to_datetime(self.pitching_data['Date'])
        self.linescore_data['Date'] = pd.to_datetime(self.linescore_data['Date'])
        
        # Sort data by date
        self.batting_data = self.batting_data.sort_values('Date')
        self.pitching_data = self.pitching_data.sort_values('Date')
        self.linescore_data = self.linescore_data.sort_values('Date')
        
        # Handle missing values in batting data
        numeric_cols = self.batting_data.select_dtypes(include=['number']).columns
        self.batting_data[numeric_cols] = self.batting_data[numeric_cols].fillna(0)
        
        # Handle missing values in pitching data
        numeric_cols = self.pitching_data.select_dtypes(include=['number']).columns
        self.pitching_data[numeric_cols] = self.pitching_data[numeric_cols].fillna(0)
        
        # Add to _preprocess_data method
        # Feature engineering for batting data
        self.batting_data['AVG'] = self.batting_data['H'] / self.batting_data['AB']
        self.batting_data['AVG'] = self.batting_data['AVG'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Calculate OBP if all required columns exist
        if all(col in self.batting_data.columns for col in ['H', 'BB', 'HBP', 'AB', 'SF']):
            self.batting_data['OBP'] = (self.batting_data['H'] + self.batting_data['BB'] + self.batting_data['HBP']) / \
                                    (self.batting_data['AB'] + self.batting_data['BB'] + self.batting_data['HBP'] + self.batting_data['SF'])
            self.batting_data['OBP'] = self.batting_data['OBP'].replace([np.inf, -np.inf], np.nan).fillna(0)
        # Fallback if some columns are missing
        elif all(col in self.batting_data.columns for col in ['H', 'BB', 'AB']):
            # Simplified OBP without HBP and SF
            self.batting_data['OBP'] = (self.batting_data['H'] + self.batting_data['BB']) / \
                                    (self.batting_data['AB'] + self.batting_data['BB'])
            self.batting_data['OBP'] = self.batting_data['OBP'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Feature engineering for pitching data
        self.pitching_data['ERA'] = (9 * self.pitching_data['ER']) / self.pitching_data['IP']
        self.pitching_data['ERA'] = self.pitching_data['ERA'].replace([np.inf, -np.inf], np.nan).fillna(0)
        self.pitching_data['WHIP'] = (self.pitching_data['H'] + self.pitching_data['BB']) / self.pitching_data['IP']
        self.pitching_data['WHIP'] = self.pitching_data['WHIP'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Create team performance metrics
        self.create_team_performance_metrics()
        
        self.logger.info("Data preprocessing completed")
    
    def init_models(self):
        """
        Implement more sophisticated model architectures with better handling of baseball-specific data patterns.
        
        This method adds XGBoost models with custom objective functions and LightGBM models
        that can better capture non-linear relationships in baseball statistics.
        """
        try:
            import xgboost as xgb
            import lightgbm as lgb
            from sklearn.ensemble import StackingRegressor, VotingRegressor
            from sklearn.svm import SVR
            from sklearn.neural_network import MLPRegressor
            
            self.logger.info("Initializing advanced model architectures")
                       
            # Define the base models for ensemble methods
            # Define models with optimized, diverse hyperparameters
            batting_base_models = [
                ('gbr', GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=4, 
                                                min_samples_split=5, subsample=0.8, random_state=42)),
                ('xgb', xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, 
                                        gamma=0.1, colsample_bytree=0.8, reg_alpha=0.1, random_state=42)),
                ('lgb', lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, 
                                        num_leaves=31, feature_fraction=0.9, random_state=42)),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, 
                                            max_features='sqrt', random_state=42)),
                ('et', ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42))
            ]
                    
            pitching_base_models = [
                ('gbr', GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=4, 
                                                min_samples_split=5, subsample=0.8, random_state=42)),
                ('xgb', xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, 
                                        gamma=0.1, colsample_bytree=0.8, reg_alpha=0.1, random_state=42)),
                ('lgb', lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, 
                                        num_leaves=31, feature_fraction=0.9, random_state=42)),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, 
                                            max_features='sqrt', random_state=42)),
                ('et', ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42))
            ]
            
            team_base_models = [
                ('lr', LogisticRegression(C=1.0, class_weight='balanced', random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)),
                ('lgb', lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))
            ]
            
            # Create meta-learners for stacking
            batting_meta_model = SVR(kernel='rbf')
            pitching_meta_model = SVR(kernel='rbf')
            
            # Create ensemble models
            self.batting_ensemble = VotingRegressor(estimators=batting_base_models, weights=[1, 1.5, 1.5, 1, 0.8])
            self.pitching_ensemble = VotingRegressor(estimators=batting_base_models, weights=[1, 1.5, 1.5, 1, 0.8])
            self.team_ensemble = VotingClassifier(estimators=team_base_models, voting='soft')
            
            # Create stacking models
            self.batting_stacking = StackingRegressor(estimators=batting_base_models, final_estimator=batting_meta_model)
            self.pitching_stacking = StackingRegressor(estimators=pitching_base_models, final_estimator=pitching_meta_model)
            
            # Multi-task Neural Network for multiple stat prediction
            self.batting_nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            self.pitching_nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

            
            self.logger.info("Advanced model architectures initialized successfully")
            
        except ImportError as e:
            self.logger.error(f"Missing required packages for advanced models: {e}")
            self.logger.info("Install with: pip install xgboost lightgbm scikit-learn>=0.24")
        except Exception as e:
            self.logger.error(f"Error initializing advanced models: {e}")

    def train_models(self) -> None:
        """
        Train the prediction models using the preprocessed data.
        
        This method trains separate models for batting statistics, pitching statistics,
        and team performance predictions.
        """
        try:
            self.logger.info("Training prediction models")

            #initialize ensemble models
            #self.init_models()
            
            # Train batting models
            self.train_batting_models()
            
            # Train pitching models
            self.train_pitching_models()
            
            # Train team models
            self.train_team_models()
            
            self.logger.info("All models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            raise
    
    def train_batting_models(self) -> None:
        """
        Train models for predicting batting statistics.
        
        This includes models for hits, home runs, RBIs, and other batting metrics.
        """
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import StackingRegressor, VotingRegressor
        from sklearn.svm import SVR
        from sklearn.neural_network import MLPRegressor
        self.logger.info("Training batting models")
        if self.batting_data is not None:
            self.logger.info("Training advanced batting models")
            # Define the batting metrics to predict
            batting_metrics = ['H', 'R', 'RBI', 'HR', 'AVG', 'OBP', 'BB', 'SO']  # Added OBP here
            
            for metric in batting_metrics:
                try:
                    # Skip if metric doesn't exist in the data
                    if metric not in self.batting_data.columns:
                        self.logger.warning(f"Metric {metric} not found in batting data, skipping")
                        continue
                    
                    # Prepare features and target (similar to _train_batting_models)
                    X = self.batting_data[['AB', 'BB', 'SO', 'PA', 'Pit', 'Str', 'HR', 'Double', 'Triple', 'SB', 'GDP', 'Home',
                          'hit_streak', 'rolling_5_avg', 'rolling_10_avg',  # Add streak features
                          'opponent_defense', 'ballpark_factor',            # Add context features
                          'day_of_week', 'game_in_season',                  # Add time features
                          'ISO', 'BABIP']].copy()                           # Add advanced metrics
                    
                    # Add derived metrics
                    X['BB_per_PA'] = X['BB'] / X['PA'].replace(0, 0.1)
                    X['SO_per_PA'] = X['SO'] / X['PA'].replace(0, 0.1)
                    X['Contact_Rate'] = (X['AB'] - X['SO']) / X['AB'].replace(0, 0.1)
                    X['XBH_per_AB'] = (X['HR'] + X['Double'] + X['Triple']) / X['AB'].replace(0, 0.1)
                    X['Str_per_Pit'] = X['Str'] / X['Pit'].replace(0, 1)
                    X['Power_Factor'] = (X['HR'] * 4 + X['Triple'] * 3 + X['Double'] * 2) / X['AB'].replace(0, 0.1)
                    
                    # Handle any NaN or infinite values
                    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    y = self.batting_data[metric].copy()
                    
                    # Convert categorical features
                    X = pd.get_dummies(X, columns=['Home'], drop_first=True)
                    
                    # Handle missing values
                    X = X.fillna(0)
                    y = y.fillna(0)
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Create NEW model instances for each metric
                    ensemble_model = VotingRegressor(estimators=[
                        ('gbr', GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=4, 
                                                        min_samples_split=5, subsample=0.8, random_state=42)),
                        ('xgb', xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, 
                                                gamma=0.1, colsample_bytree=0.8, reg_alpha=0.1, random_state=42)),
                        ('lgb', lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, 
                                                num_leaves=31, feature_fraction=0.9, random_state=42)),
                        ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, 
                                                    max_features='sqrt', random_state=42)),
                        ('et', ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42))
                    ], weights=[1, 1.5, 1.5, 1, 0.8])
                    
                    stacking_model = StackingRegressor(
                        estimators=[
                            ('gbr', GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=4)),
                            ('rf', RandomForestRegressor(n_estimators=100)),
                            ('xgb', xgb.XGBRegressor(n_estimators=150))
                        ],
                        final_estimator=SVR(kernel='rbf')
                    )
                    
                    nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                    
                    # Train the models for this specific metric
                    ensemble_model.fit(X_train, y_train)
                    stacking_model.fit(X_train, y_train)
                    nn_model.fit(X_train, y_train)
                    
                    # Store in models dictionary
                    self.batting_models[metric] = {
                        'ensemble': ensemble_model,
                        'stacking': stacking_model,
                        'nn': nn_model
                    }
                    
                    # Evaluate performance
                    y_pred = ensemble_model.predict(X_test)
                    mse = np.mean((y_pred - y_test) ** 2)
                    rmse = np.sqrt(mse)
                    self.logger.info(f"Advanced batting model for {metric} - RMSE: {rmse:.4f}")

                    for metric, model in self.batting_models.items():
                        print(f"DEBUG: Batting model for {metric} is now {type(model)}")
                    
                except Exception as e:
                    self.logger.error(f"Error training advanced batting model for {metric}: {e}")
                    continue
        
    def train_pitching_models(self) -> None:
        """
        Train models for predicting pitching statistics.
        
        This includes models for ERA, WHIP, strikeouts, and other pitching metrics.
        """
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import StackingRegressor, VotingRegressor
        from sklearn.svm import SVR
        from sklearn.neural_network import MLPRegressor
        self.logger.info("Training pitching models")
        
        # Define the pitching metrics to predict
        if self.pitching_data is not None:
            self.logger.info("Training advanced pitching models")
            
            # Define the pitching metrics to predict (same as in _train_pitching_models)
            pitching_metrics = ['IP', 'H', 'R', 'ER', 'ERA', 'WHIP', 'SO', 'BB', 'HR', 'GB', 'FB', 'Pit', 'BF']  # Added ERA and WHIP here
            
            for metric in pitching_metrics:
                try:
                    # Skip if metric doesn't exist in the data
                    if metric not in self.pitching_data.columns:
                        self.logger.warning(f"Metric {metric} not found in pitching data, skipping")
                        continue
                    
                    # Prepare features and target (similar to _train_pitching_models)
                    X = self.pitching_data[['IP', 'SO', 'BB', 'HR', 'GB', 'FB', 'Str', 'Pit', 'BF', 'Home',
                        'quality_start_streak', 'rolling_3_ERA',            # Streak features
                        'opponent_offense', 'ballpark_factor',              # Context features
                        'day_of_week', 'game_in_season', 'days_rest',       # Time features
                        'FIP', 'BABIP_against']].copy()      
                    
                    # Calculate derived metrics
                    X['K_per_9'] = 9 * X['SO'] / X['IP'].replace(0, 0.1)
                    X['BB_per_9'] = 9 * X['BB'] / X['IP'].replace(0, 0.1)
                    X['HR_per_9'] = 9 * X['HR'] / X['IP'].replace(0, 0.1)
                    X['Str_pct'] = X['Str'] / X['Pit'].replace(0, 1)
                    X['GB_FB_ratio'] = X['GB'] / X['FB'].replace(0, 1)
                    
                    # Handle any NaN or infinite values
                    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    y = self.pitching_data[metric].copy()
                    
                    # Add team encoding
                    X = pd.get_dummies(X, columns=['Home'], drop_first=True)
                    
                    # Handle missing values
                    X = X.fillna(0)
                    y = y.fillna(0)
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Create NEW model instances for each metric
                    ensemble_model = VotingRegressor(estimators=[
                        ('gbr', GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=4, 
                                                        min_samples_split=5, subsample=0.8, random_state=42)),
                        ('xgb', xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, 
                                                gamma=0.1, colsample_bytree=0.8, reg_alpha=0.1, random_state=42)),
                        ('lgb', lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, 
                                                num_leaves=31, feature_fraction=0.9, random_state=42)),
                        ('rf', RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=2, 
                                                    max_features='sqrt', random_state=42)),
                        ('et', ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42))
                    ], weights=[1, 1.5, 1.5, 1, 0.8])
                    
                    stacking_model = StackingRegressor(
                        estimators=[
                            ('gbr', GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=4)),
                            ('rf', RandomForestRegressor(n_estimators=100)),
                            ('xgb', xgb.XGBRegressor(n_estimators=150))
                        ],
                        final_estimator=SVR(kernel='rbf')
                    )
                    
                    nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                    
                    # Train the models for this specific metric
                    ensemble_model.fit(X_train, y_train)
                    stacking_model.fit(X_train, y_train)
                    nn_model.fit(X_train, y_train)
                    
                    # Store in models dictionary
                    self.pitching_models[metric] = {
                        'ensemble': ensemble_model,
                        'stacking': stacking_model,
                        'nn': nn_model
                    }
                    
                    # Evaluate performance
                    y_pred = ensemble_model.predict(X_test)
                    mse = np.mean((y_pred - y_test) ** 2)
                    rmse = np.sqrt(mse)
                    self.logger.info(f"Advanced pitching model for {metric} - RMSE: {rmse:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training advanced pitching model for {metric}: {e}")
                    continue
    
    def train_team_models(self) -> None:
        """
        Train team-specific models for predicting win probability.
        
        This enhanced version includes streak data, head-to-head statistics,
        ballpark factors, and other advanced metrics to improve prediction accuracy.
        """
        self.logger.info("Training team win prediction models with enhanced features")
        
        try:
            # Initialize dictionary to store team-specific models
            self.team_models = {}
            
            # Create team performance metrics if not already created
            if not hasattr(self, 'team_records'):
                self.create_team_performance_metrics()
            
            # Initialize ballpark factors if not already created
            if not hasattr(self, 'ballpark_factors'):
                self.add_ballpark_factors()
            
            # Calculate team streaks
            team_streaks = {}
            for team in self.team_records.index:
                team_games = []
                # Process linescore data to get recent team results
                i = 0
                while i < len(self.linescore_data) - 1:
                    try:
                        away_row = self.linescore_data.iloc[i]
                        home_row = self.linescore_data.iloc[i+1]
                        
                        if away_row['Date'] != home_row['Date'] or away_row['AwayTeam'] != home_row['AwayTeam'] or away_row['HomeTeam'] != home_row['HomeTeam']:
                            i += 1
                            continue
                        
                        # Check if team played in this game
                        if team in away_row['AwayTeam'] or team in home_row['HomeTeam']:
                            game_date = home_row['Date']
                            is_home = (team == home_row['HomeTeam'])
                            away_runs = float(away_row.get('R', 0))
                            home_runs = float(home_row.get('R', 0))
                            team_win = (is_home and home_runs > away_runs) or (not is_home and away_runs > home_runs)
                            
                            team_games.append({
                                'date': game_date,
                                'win': team_win
                            })
                        
                        i += 2
                    except Exception as e:
                        self.logger.warning(f"Error processing game streak at index {i}: {e}")
                        i += 1
                
                # Sort games by date (oldest to newest)
                team_games.sort(key=lambda x: x['date'])
                
                # Calculate current win streak and recent form
                win_streak = 0
                recent_wins = 0
                recent_games = min(10, len(team_games))
                
                # Calculate streak (consecutive most recent wins)
                for i in range(len(team_games)-1, -1, -1):
                    if team_games[i]['win']:
                        win_streak += 1
                    else:
                        break
                
                # Calculate recent form (win % in last 10 games)
                for i in range(len(team_games)-1, max(len(team_games)-11, -1), -1):
                    if team_games[i]['win']:
                        recent_wins += 1
                
                team_streaks[team] = {
                    'win_streak': win_streak,
                    'recent_win_pct': recent_wins / recent_games if recent_games > 0 else 0.5
                }
            
            # Process all teams in team_records
            for team_name in self.team_records.index:
                self.logger.info(f"Training model for team: {team_name}")
                
                # Get games where this team played
                team_games = []
                
                # Process the linescore data two rows at a time
                i = 0
                while i < len(self.linescore_data) - 1:
                    try:
                        # Get the away team row and home team row
                        away_row = self.linescore_data.iloc[i]
                        home_row = self.linescore_data.iloc[i+1]
                        
                        # Skip if not a valid game pair
                        if away_row['Date'] != home_row['Date'] or away_row['AwayTeam'] != home_row['AwayTeam'] or away_row['HomeTeam'] != home_row['HomeTeam']:
                            i += 1
                            continue
                        
                        # Check if this team played in this game
                        if team_name in away_row['AwayTeam'] or team_name in home_row['HomeTeam']:
                            # Get team names
                            home_team = home_row['HomeTeam']
                            away_team = away_row['AwayTeam']
                            game_date = home_row['Date']
                            
                            # Get the scores
                            try:
                                away_runs = float(away_row.get('R', 0))
                                home_runs = float(home_row.get('R', 0))
                                
                                # Get team records
                                home_record = self.team_records.loc[home_team].copy() if home_team in self.team_records.index else None
                                away_record = self.team_records.loc[away_team].copy() if away_team in self.team_records.index else None
                                
                                if home_record is not None and away_record is not None:
                                    # Get head-to-head statistics
                                    h2h_stats = self.get_head_to_head_stats(home_team, away_team, game_date)
                                    
                                    # Create features
                                    is_home = (team_name == home_team)
                                    team_win = (is_home and home_runs > away_runs) or (not is_home and away_runs > home_runs)
                                    
                                    # Create enhanced feature set
                                    features = {
                                        'TeamWinPct': self.team_records.loc[team_name, 'WinPct'],
                                        'OpponentWinPct': self.team_records.loc[away_team if is_home else home_team, 'WinPct'],
                                        'IsHome': 1 if is_home else 0,
                                        'HomeTeamHomeWinPct': home_record['HomeWinPct'],
                                        'AwayTeamAwayWinPct': away_record['AwayWinPct'],
                                        
                                        # Add head-to-head statistics
                                        'H2HWinPct': h2h_stats.get('blended_win_pct', 0.5) if is_home else (1 - h2h_stats.get('blended_win_pct', 0.5)),
                                        'H2HGameCount': h2h_stats.get('total_games', 0),
                                        'H2HRunDiff': h2h_stats.get('run_differential', 0) if is_home else -h2h_stats.get('run_differential', 0),
                                        
                                        # Add streak features
                                        'WinStreak': team_streaks.get(team_name, {}).get('win_streak', 0),
                                        'RecentForm': team_streaks.get(team_name, {}).get('recent_win_pct', 0.5),
                                        
                                        # Add run production/prevention metrics
                                        # 'RunsPerGame': home_record['rpg'] if is_home else away_record['rpg'],
                                        # 'RunsAllowedPerGame': home_record['rapg'] if is_home else away_record['rapg'],
                                        # 'RunDifferential': (home_record['runs_scored'] - home_record['runs_allowed']) if is_home 
                                        #                 else (away_record['runs_scored'] - away_record['runs_allowed']),
                                        
                                        # Add ballpark factors
                                        #'BallparkFactor': self.ballpark_factors.get(home_team, 1.0) if is_home else 1.0,
                                        
                                        # Add time-based features
                                        'DayOfWeek': game_date.dayofweek if isinstance(game_date, pd.Timestamp) else 0,
                                        'GameInSeason': (game_date - self.linescore_data['Date'].min()).days + 1 if isinstance(game_date, pd.Timestamp) else 0,
                                        
                                        # Target variable
                                        'TeamWin': 1 if team_win else 0
                                    }
                                    
                                    team_games.append(features)
                            except (ValueError, TypeError) as e:
                                self.logger.warning(f"Error processing game for team model: {e}")
                        
                        # Move to the next game (skip two rows)
                        i += 2
                    except Exception as e:
                        self.logger.warning(f"Error processing game at index {i}: {e}")
                        i += 1
                
                # Create DataFrame from team games
                if len(team_games) < 10:
                    self.logger.warning(f"Not enough games ({len(team_games)}) to train model for {team_name}")
                    continue
                    
                games_df = pd.DataFrame(team_games)
                
                # Prepare features and target
                X = games_df.drop('TeamWin', axis=1)
                y = games_df['TeamWin']
                
                # Handle missing values
                X = X.fillna(0)
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Initialize model library imports if needed
                import xgboost as xgb
                import lightgbm as lgb
                
                # Train the ensemble model with enhanced configuration
                team_ensemble = VotingClassifier(estimators=[
                    ('lr', LogisticRegression(C=1.0, class_weight='balanced', random_state=42)),
                    ('xgb', xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)),
                    ('lgb', lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))
                ], voting='soft')
                
                team_ensemble.fit(X_train, y_train)
                
                # Store in models dictionary
                self.team_models[team_name] = team_ensemble
                
                # Evaluate performance
                y_pred = team_ensemble.predict(X_test)
                accuracy = np.mean(y_pred == y_test)
                self.logger.info(f"Team win prediction model for {team_name} - Accuracy: {accuracy:.4f}")
                
                # Feature importance analysis (if available)
                try:
                    for name, clf in team_ensemble.named_estimators_.items():
                        if hasattr(clf, 'feature_importances_'):
                            importances = clf.feature_importances_
                            indices = np.argsort(importances)[::-1]
                            feature_names = X.columns
                            
                            self.logger.info(f"Top 5 features for {name} model:")
                            for i in range(min(5, len(indices))):
                                self.logger.info(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
                except:
                    pass
                
            self.logger.info(f"Trained team models for {len(self.team_models)} teams with enhanced features")
                
        except Exception as e:
            self.logger.error(f"Error training team models: {e}")
            raise
    
    def save_models(self, model_dir: str = 'models') -> None:
        """
        Save the trained models to disk.
        
        Args:
            model_dir (str): Directory to save the models in.
        """
        try:
            # Create the model directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Save batting models
            for metric, model in self.batting_models.items():
                joblib.dump(model, os.path.join(model_dir, f'batting_{metric}.pkl'))
            
            # Save pitching models
            for metric, model in self.pitching_models.items():
                joblib.dump(model, os.path.join(model_dir, f'pitching_{metric}.pkl'))
            
            # Save team models
            for model_name, model in self.team_models.items():
                joblib.dump(model, os.path.join(model_dir, f'team_{model_name}.pkl'))
            
            self.logger.info(f"Models saved to {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self, model_dir: str = 'models') -> None:
        """
        Load the trained models from disk.
        
        Args:
            model_dir (str): Directory to load the models from.
        """
        try:
            # Check if the model directory exists
            if not os.path.exists(model_dir):
                self.logger.warning(f"Model directory {model_dir} does not exist")
                return
            
            # Load batting models
            self.batting_models = {}
            for model_file in glob.glob(os.path.join(model_dir, 'batting_*.pkl')):
                metric = os.path.basename(model_file).replace('batting_', '').replace('.pkl', '')
                self.batting_models[metric] = joblib.load(model_file)
            
            # Load pitching models
            self.pitching_models = {}
            for model_file in glob.glob(os.path.join(model_dir, 'pitching_*.pkl')):
                metric = os.path.basename(model_file).replace('pitching_', '').replace('.pkl', '')
                self.pitching_models[metric] = joblib.load(model_file)
            
            # Load team models
            self.team_models = {}
            for model_file in glob.glob(os.path.join(model_dir, 'team_*.pkl')):
                model_name = os.path.basename(model_file).replace('team_', '').replace('.pkl', '')
                self.team_models[model_name] = joblib.load(model_file)
            
            self.logger.info(f"Models loaded from {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

    def get_player_stats(self, player_name: str, target_date: str, 
                        last_n_games: int = 163) -> Dict[str, Dict[str, float]]:
        """
        Get historical performance statistics for a specific player.
        
        Args:
            player_name (str): Name of the player to get statistics for.
            target_date (str): The reference date for retrieving historical data.
            last_n_games (int): Number of games to include in the statistics.
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing player statistics
                for overall, home, and away games.
        """
        try:
            target_date = pd.to_datetime(target_date)
            
            # Check if the player is a batter or pitcher
            batter_data = self.batting_data[
                (self.batting_data['Batting'].str.contains(player_name, case=True, na=False)) & 
                (self.batting_data['Date'] < target_date)
            ].sort_values('Date', ascending=False)
            
            pitcher_data = self.pitching_data[
                (self.pitching_data['Pitching'].str.contains(player_name, case=True, na=False)) & 
                (self.pitching_data['Date'] < target_date)
            ].sort_values('Date', ascending=False)
            
            result = {
                'overall': {},
                'home': {},
                'away': {}
            }
            
            # Process batter data
            if not batter_data.empty:
                # Overall stats (last N games)
                overall_data = batter_data.head(last_n_games)
                
                # Home stats (last N home games)
                home_data = batter_data[batter_data['Home'] == True].head(last_n_games)
                
                # Away stats (last N away games)
                away_data = batter_data[batter_data['Home'] == False].head(last_n_games)
                
                # Calculate overall stats
                if not overall_data.empty:
                    # Calculate BB (walks) and other OBP components if available
                    bb = overall_data['BB'].sum() if 'BB' in overall_data.columns else 0
                    hbp = overall_data['HBP'].sum() if 'HBP' in overall_data.columns else 0
                    sf = overall_data['SF'].sum() if 'SF' in overall_data.columns else 0
                    
                    # Calculate OBP
                    obp_denominator = overall_data['AB'].sum() + bb + hbp + sf
                    obp = (overall_data['H'].sum() + bb + hbp) / obp_denominator if obp_denominator > 0 else 0
                    
                    result['overall'] = {
                        'games': len(overall_data),
                        'AB': overall_data['AB'].sum(),
                        'H': overall_data['H'].sum(),
                        'BB': bb,
                        'SO': overall_data['SO'].sum() if 'SO' in overall_data.columns else 0,
                        'PA': overall_data['PA'].sum() if 'PA' in overall_data.columns else overall_data['AB'].sum() + bb,
                        'R': overall_data['R'].sum(),
                        'RBI': overall_data['RBI'].sum(),
                        'HR': overall_data['HR'].sum() if 'HR' in overall_data.columns else 0,
                        'Double': overall_data['Double'].sum() if 'Double' in overall_data.columns else 0,
                        'Triple': overall_data['Triple'].sum() if 'Triple' in overall_data.columns else 0,
                        'SB': overall_data['SB'].sum() if 'SB' in overall_data.columns else 0,
                        'GDP': overall_data['GDP'].sum() if 'GDP' in overall_data.columns else 0,
                        'Pit': overall_data['Pit'].sum() if 'Pit' in overall_data.columns else 0,
                        'Str': overall_data['Str'].sum() if 'Str' in overall_data.columns else 0,
                        'AVG': overall_data['H'].sum() / overall_data['AB'].sum() if overall_data['AB'].sum() > 0 else 0,
                        'OBP': obp
                    }
                    
                    # Add these features to 'overall' dictionary
                    result['overall']['hit_streak'] = batter_data['hit_streak'].iloc[0] if 'hit_streak' in batter_data.columns else 0
                    result['overall']['rolling_5_avg'] = batter_data['rolling_5_avg'].iloc[0] if 'rolling_5_avg' in batter_data.columns else 0
                    result['overall']['rolling_10_avg'] = batter_data['rolling_10_avg'].iloc[0] if 'rolling_10_avg' in batter_data.columns else 0
                    
                    # Add ballpark factors and opponent adjustments
                    result['overall']['ballpark_factor'] = batter_data['ballpark_factor'].iloc[0] if 'ballpark_factor' in batter_data.columns else 1.0
                    result['overall']['opponent_defense'] = batter_data['opponent_defense'].iloc[0] if 'opponent_defense' in batter_data.columns else 1.0
                    
                    # Add advanced metrics
                    result['overall']['ISO'] = batter_data['ISO'].iloc[0] if 'ISO' in batter_data.columns else 0
                    result['overall']['BABIP'] = batter_data['BABIP'].iloc[0] if 'BABIP' in batter_data.columns else 0


                # Calculate home stats
                if not home_data.empty:
                    # Calculate BB (walks) and other OBP components if available
                    bb = home_data['BB'].sum() if 'BB' in home_data.columns else 0
                    hbp = home_data['HBP'].sum() if 'HBP' in home_data.columns else 0
                    sf = home_data['SF'].sum() if 'SF' in home_data.columns else 0
                    
                    # Calculate OBP
                    obp_denominator = home_data['AB'].sum() + bb + hbp + sf
                    obp = (home_data['H'].sum() + bb + hbp) / obp_denominator if obp_denominator > 0 else 0
                    
                    result['home'] = {
                        'games': len(home_data),
                        'AB': home_data['AB'].sum(),
                        'H': home_data['H'].sum(),
                        'BB': bb,
                        'SO': home_data['SO'].sum() if 'SO' in home_data.columns else 0,
                        'PA': home_data['PA'].sum() if 'PA' in home_data.columns else home_data['Abatter_dataB'].sum() + bb,
                        'R': home_data['R'].sum(),
                        'RBI': home_data['RBI'].sum(),
                        'HR': home_data['HR'].sum() if 'HR' in home_data.columns else 0,
                        'Double': home_data['Double'].sum() if 'Double' in home_data.columns else 0,
                        'Triple': home_data['Triple'].sum() if 'Triple' in home_data.columns else 0,
                        'SB': home_data['SB'].sum() if 'SB' in overall_data.columns else 0,
                        'GDP': home_data['GDP'].sum() if 'GDP' in home_data.columns else 0,
                        'Pit': home_data['Pit'].sum() if 'Pit' in home_data.columns else 0,
                        'Str': home_data['Str'].sum() if 'Str' in home_data.columns else 0,
                        'AVG': home_data['H'].sum() / home_data['AB'].sum() if home_data['AB'].sum() > 0 else 0,
                        'OBP': obp
                    }


        

                # Calculate away stats
                if not away_data.empty:
                    # Calculate BB (walks) and other OBP components if available
                    bb = away_data['BB'].sum() if 'BB' in away_data.columns else 0
                    hbp = away_data['HBP'].sum() if 'HBP' in away_data.columns else 0
                    sf = away_data['SF'].sum() if 'SF' in away_data.columns else 0
                    
                    # Calculate OBP
                    obp_denominator = away_data['AB'].sum() + bb + hbp + sf
                    obp = (away_data['H'].sum() + bb + hbp) / obp_denominator if obp_denominator > 0 else 0
                    
                    result['away'] = {
                        'games': len(away_data),
                        'AB': away_data['AB'].sum(),
                        'H': away_data['H'].sum(),
                        'BB': bb,
                        'SO': away_data['SO'].sum() if 'SO' in away_data.columns else 0,
                        'PA': away_data['PA'].sum() if 'PA' in away_data.columns else away_data['AB'].sum() + bb,
                        'R': away_data['R'].sum(),
                        'RBI': away_data['RBI'].sum(),
                        'HR': away_data['HR'].sum() if 'HR' in away_data.columns else 0,
                        'Double': away_data['Double'].sum() if 'Double' in away_data.columns else 0,
                        'Triple': away_data['Triple'].sum() if 'Triple' in away_data.columns else 0,
                        'SB': away_data['SB'].sum() if 'SB' in away_data.columns else 0,
                        'GDP': away_data['GDP'].sum() if 'GDP' in away_data.columns else 0,
                        'Pit': away_data['Pit'].sum() if 'Pit' in away_data.columns else 0,
                        'Str': away_data['Str'].sum() if 'Str' in away_data.columns else 0,
                        'AVG': away_data['H'].sum() / away_data['AB'].sum() if away_data['AB'].sum() > 0 else 0,
                        'OBP': obp
                    }
            
            # Process pitcher data
            elif not pitcher_data.empty:
                # Overall stats (last N games)
                overall_data = pitcher_data.head(last_n_games)
                
                # Home stats (last N home games)
                home_data = pitcher_data[pitcher_data['Home'] == True].head(last_n_games)
                
                # Away stats (last N away games)
                away_data = pitcher_data[pitcher_data['Home'] == False].head(last_n_games)
                
                # Calculate overall stats
                if not overall_data.empty:
                    result['overall'] = {
                        'games': len(overall_data),
                        'IP': overall_data['IP'].sum(),
                        'H': overall_data['H'].sum(),
                        'R': overall_data['R'].sum(),
                        'ER': overall_data['ER'].sum(),
                        'SO': overall_data['SO'].sum() if 'SO' in overall_data.columns else 0,
                        'BB': overall_data['BB'].sum() if 'BB' in overall_data.columns else 0,
                        'HR': overall_data['HR'].sum() if 'HR' in overall_data.columns else 0,
                        'GB': overall_data['GB'].sum() if 'GB' in overall_data.columns else 0,
                        'FB': overall_data['FB'].sum() if 'FB' in overall_data.columns else 0,
                        'Str': overall_data['Str'].sum() if 'Str' in overall_data.columns else 0,
                        'Pit': overall_data['Pit'].sum() if 'Pit' in overall_data.columns else 0,
                        'BF': overall_data['BF'].sum() if 'BF' in overall_data.columns else 0,
                        'ERA': (9 * overall_data['ER'].sum() / overall_data['IP'].sum()) if overall_data['IP'].sum() > 0 else 0,
                        'WHIP': ((overall_data['H'].sum() + overall_data['BB'].sum()) / overall_data['IP'].sum()) 
                            if overall_data['IP'].sum() > 0 and 'BB' in overall_data.columns else 0
                    }

                    result['overall']['quality_start_streak'] = pitcher_data['quality_start_streak'].iloc[0] if 'quality_start_streak' in pitcher_data.columns else 0
                    result['overall']['rolling_3_ERA'] = pitcher_data['rolling_3_ERA'].iloc[0] if 'rolling_3_ERA' in pitcher_data.columns else 0
                    
                    # Add ballpark factors and opponent adjustments
                    result['overall']['ballpark_factor'] = pitcher_data['ballpark_factor'].iloc[0] if 'ballpark_factor' in pitcher_data.columns else 1.0
                    result['overall']['opponent_offense'] = pitcher_data['opponent_offense'].iloc[0] if 'opponent_offense' in pitcher_data.columns else 1.0
                    
                    # Add advanced metrics
                    result['overall']['FIP'] = pitcher_data['FIP'].iloc[0] if 'FIP' in pitcher_data.columns else 0
                    result['overall']['BABIP_against'] = pitcher_data['BABIP_against'].iloc[0] if 'BABIP_against' in pitcher_data.columns else 0
                    
                    # Add rest days
                    result['overall']['days_rest'] = pitcher_data['days_rest'].iloc[0] if 'days_rest' in pitcher_data.columns else 0
                
                # Calculate home stats
                if not home_data.empty:
                    result['home'] = {
                        'games': len(home_data),
                        'IP': home_data['IP'].sum(),
                        'H': home_data['H'].sum(),
                        'R': home_data['R'].sum(),
                        'ER': home_data['ER'].sum(),
                        'SO': home_data['SO'].sum() if 'SO' in home_data.columns else 0,
                        'BB': home_data['BB'].sum() if 'BB' in home_data.columns else 0,
                        'HR': home_data['HR'].sum() if 'HR' in home_data.columns else 0,
                        'GB': home_data['GB'].sum() if 'GB' in home_data.columns else 0,
                        'FB': home_data['FB'].sum() if 'FB' in home_data.columns else 0,
                        'Str': home_data['Str'].sum() if 'Str' in home_data.columns else 0,
                        'Pit': home_data['Pit'].sum() if 'Pit' in home_data.columns else 0,
                        'BF': home_data['BF'].sum() if 'BF' in home_data.columns else 0,
                        'ERA': (9 * home_data['ER'].sum() / home_data['IP'].sum()) if home_data['IP'].sum() > 0 else 0,
                        'WHIP': ((home_data['H'].sum() + home_data['BB'].sum()) / home_data['IP'].sum()) 
                            if home_data['IP'].sum() > 0 and 'BB' in home_data.columns else 0
                    }
                
                # Calculate away stats
                if not away_data.empty:
                    result['away'] = {
                        'games': len(away_data),
                        'IP': away_data['IP'].sum(),
                        'H': away_data['H'].sum(),
                        'R': away_data['R'].sum(),
                        'ER': away_data['ER'].sum(),
                        'SO': away_data['SO'].sum() if 'SO' in away_data.columns else 0,
                        'BB': away_data['BB'].sum() if 'BB' in away_data.columns else 0,
                        'HR': away_data['HR'].sum() if 'HR' in away_data.columns else 0,
                        'GB': away_data['GB'].sum() if 'GB' in away_data.columns else 0,
                        'FB': away_data['FB'].sum() if 'FB' in away_data.columns else 0,
                        'Str': away_data['Str'].sum() if 'Str' in away_data.columns else 0,
                        'Pit': away_data['Pit'].sum() if 'Pit' in away_data.columns else 0,
                        'BF': away_data['BF'].sum() if 'BF' in away_data.columns else 0,
                        'ERA': (9 * away_data['ER'].sum() / away_data['IP'].sum()) if away_data['IP'].sum() > 0 else 0,
                        'WHIP': ((away_data['H'].sum() + away_data['BB'].sum()) / away_data['IP'].sum()) 
                            if away_data['IP'].sum() > 0 and 'BB' in away_data.columns else 0
                    }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting player stats for {player_name}: {e}")
            return {'overall': {}, 'home': {}, 'away': {}}
    
    def get_team_stats(self, team_name: str, target_date: str) -> Dict[str, float]:
        """
        Get team performance statistics up to a specific date.
        
        Args:
            team_name (str): Name of the team to get statistics for.
            target_date (str): The reference date for retrieving historical data.
            
        Returns:
            Dict[str, float]: Dictionary containing team statistics.
        """
        try:
            target_date = pd.to_datetime(target_date)
            
            # Filter linescore data to include only games before the target date
            filtered_linescore = self.linescore_data[self.linescore_data['Date'] < target_date]
            
            # Calculate team performance metrics
            wins = 0
            losses = 0
            home_wins = 0
            home_losses = 0
            away_wins = 0
            away_losses = 0
            runs_scored = 0
            runs_allowed = 0
            
            # Process the linescore data two rows at a time
            i = 0
            while i < len(filtered_linescore) - 1:
                try:
                    # Get the away team row and home team row
                    away_row = filtered_linescore.iloc[i]
                    home_row = filtered_linescore.iloc[i+1]
                    
                    # Verify that these rows are for the same game
                    if away_row['Date'] != home_row['Date'] or away_row['AwayTeam'] != home_row['AwayTeam'] or away_row['HomeTeam'] != home_row['HomeTeam']:
                        # These rows don't represent the same game, skip to next row
                        #self.logger.warning(f"Mismatched game data at index {i}, skipping")
                        i += 1
                        continue
                    
                    # Get team names
                    home_team = home_row['HomeTeam']
                    away_team = away_row['AwayTeam']
                    
                    # Skip if this game doesn't involve the team we're looking for
                    if team_name not in home_team and team_name not in away_team :
                        i += 2
                        continue
                    
                    # Get the scores - for away team from first row
                    away_runs = away_row.get('R', None)
                    # Get the scores - for home team from second row
                    home_runs = home_row.get('R', None)
                    
                    # If we can't find the runs columns, try to infer them
                    if home_runs is None or away_runs is None:
                        # For away row, look for numeric values
                        if isinstance(away_row, pd.Series):
                            away_numeric_cols = [col for col, val in away_row.items() if isinstance(val, (int, float)) and not pd.isna(val)]
                            if len(away_numeric_cols) >= 2:
                                away_runs = away_row[away_numeric_cols[-2]]  # Use R column (usually second-to-last numeric)
                        
                        # For home row, look for numeric values
                        if isinstance(home_row, pd.Series):
                            home_numeric_cols = [col for col, val in home_row.items() if isinstance(val, (int, float)) and not pd.isna(val)]
                            if len(home_numeric_cols) >= 2:
                                home_runs = home_row[home_numeric_cols[-2]]  # Use R column (usually second-to-last numeric)
                    
                    try:
                        home_runs = float(home_runs)
                        away_runs = float(away_runs)
                        
                        if team_name in home_team:
                            # Team is playing at home
                            runs_scored += home_runs
                            runs_allowed += away_runs
                            
                            if home_runs > away_runs:
                                wins += 1
                                home_wins += 1
                            else:
                                losses += 1
                                home_losses += 1
                        else:
                            # Team is playing away
                            runs_scored += away_runs
                            runs_allowed += home_runs
                            
                            if away_runs > home_runs:
                                wins += 1
                                away_wins += 1
                            else:
                                losses += 1
                                away_losses += 1
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error processing team stats game: {e}")
                    
                    # Move to the next game (skip two rows)
                    i += 2
                    
                except Exception as e:
                    self.logger.warning(f"Error processing game at index {i}: {e}")
                    i += 1  # Move to next row if there's an error
            
            # Calculate win percentages
            win_pct = wins / (wins + losses) if (wins + losses) > 0 else 0
            home_win_pct = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0
            away_win_pct = away_wins / (away_wins + away_losses) if (away_wins + away_losses) > 0 else 0
            
            # Calculate runs per game
            games_played = wins + losses
            rpg = runs_scored / games_played if games_played > 0 else 0
            rapg = runs_allowed / games_played if games_played > 0 else 0
            
            return {
                'wins': wins,
                'losses': losses,
                'win_pct': win_pct,
                'home_wins': home_wins,
                'home_losses': home_losses,
                'home_win_pct': home_win_pct,
                'away_wins': away_wins,
                'away_losses': away_losses,
                'away_win_pct': away_win_pct,
                'runs_scored': runs_scored,
                'runs_allowed': runs_allowed,
                'run_differential': runs_scored - runs_allowed,
                'rpg': rpg,
                'rapg': rapg
            }
            
        except Exception as e:
            self.logger.error(f"Error getting team stats for {team_name}: {e}")
            return {}
    
    def prepare_player_features(self, player_name: str, player_stats: Dict, is_home_game: bool, is_batter: bool) -> Dict:
        """
        Prepare feature dictionary for player prediction based on their stats.
        
        Args:
            player_name: Name of the player
            player_stats: Dictionary of player statistics 
            is_home_game: Whether the player is playing at home
            is_batter: Whether the player is a batter (False for pitcher)
            
        Returns:
            Dictionary of features for model input
        """
        features_dict = {}
        
        # Create appropriate feature vector based on player type
        if is_batter:
            features_dict = {
                'AB': player_stats.get('AB', 0) / max(player_stats.get('games', 1), 1),
                'H': player_stats.get('H', 0) / max(player_stats.get('games', 1), 1),
                'BB': player_stats.get('BB', 0) / max(player_stats.get('games', 1), 1),
                'SO': player_stats.get('SO', 0) / max(player_stats.get('games', 1), 1),
                'PA': player_stats.get('PA', player_stats.get('AB', 0) + player_stats.get('BB', 0)) / max(player_stats.get('games', 1), 1),
                'R': player_stats.get('R', 0) / max(player_stats.get('games', 1), 1),
                'RBI': player_stats.get('RBI', 0) / max(player_stats.get('games', 1), 1),
                'HR': player_stats.get('HR', 0) / max(player_stats.get('games', 1), 1),
                'Double': player_stats.get('Double', 0) / max(player_stats.get('games', 1), 1),
                'Triple': player_stats.get('Triple', 0) / max(player_stats.get('games', 1), 1),
                'SB': player_stats.get('SB', 0) / max(player_stats.get('games', 1), 1),
                'GDP': player_stats.get('GDP', 0) / max(player_stats.get('games', 1), 1),
                'Pit': player_stats.get('Pit', 0) / max(player_stats.get('games', 1), 1),
                'Str': player_stats.get('Str', 0) / max(player_stats.get('games', 1), 1),
                'AVG': player_stats.get('AVG', 0),
                'OBP': player_stats.get('OBP', 0),
                'Home_True': 1 if is_home_game else 0
            }
            
            # Add streak and enhanced features
            streak_data = self.get_player_streak_data(player_name, is_batter=True)
            features_dict.update(streak_data)
            
            # Calculate derived metrics
            pa = max(features_dict['PA'], 0.1)
            ab = max(features_dict['AB'], 0.1)
            pit = max(features_dict['Pit'], 1)

            features_dict['BB_per_PA'] = features_dict['BB'] / pa
            features_dict['SO_per_PA'] = features_dict['SO'] / pa
            features_dict['Contact_Rate'] = (features_dict['AB'] - features_dict['SO']) / ab
            features_dict['XBH_per_AB'] = (features_dict['HR'] + features_dict['Double'] + features_dict['Triple']) / ab
            features_dict['Str_per_Pit'] = features_dict['Str'] / pit
            features_dict['Power_Factor'] = (features_dict['HR'] * 4 + features_dict['Triple'] * 3 + features_dict['Double'] * 2) / ab
            
        else:  # Pitcher
            features_dict = {
                'IP': player_stats.get('IP', 0) / max(player_stats.get('games', 1), 1),
                'H': player_stats.get('H', 0) / max(player_stats.get('games', 1), 1),
                'R': player_stats.get('R', 0) / max(player_stats.get('games', 1), 1),
                'ER': player_stats.get('ER', 0) / max(player_stats.get('games', 1), 1),
                'SO': player_stats.get('SO', 0) / max(player_stats.get('games', 1), 1),
                'BB': player_stats.get('BB', 0) / max(player_stats.get('games', 1), 1),
                'HR': player_stats.get('HR', 0) / max(player_stats.get('games', 1), 1),
                'GB': player_stats.get('GB', 0) / max(player_stats.get('games', 1), 1),
                'FB': player_stats.get('FB', 0) / max(player_stats.get('games', 1), 1),
                'Str': player_stats.get('Str', 0) / max(player_stats.get('games', 1), 1),
                'Pit': player_stats.get('Pit', 0) / max(player_stats.get('games', 1), 1),
                'BF': player_stats.get('BF', 0) / max(player_stats.get('games', 1), 1),
                'ERA': player_stats.get('ERA', 0),
                'WHIP': player_stats.get('WHIP', 0),
                'Home_True': 1 if is_home_game else 0
            }
            
            # Add streak and enhanced features
            streak_data = self.get_player_streak_data(player_name, is_batter=False)
            features_dict.update(streak_data)
            
            # Calculate derived metrics
            ip = max(player_stats.get('IP', 0), 0.1)
            pit = max(player_stats.get('Pit', 0), 1)
            fb = max(player_stats.get('FB', 0), 1)

            features_dict['K_per_9'] = 9 * player_stats.get('SO', 0) / ip
            features_dict['BB_per_9'] = 9 * player_stats.get('BB', 0) / ip
            features_dict['HR_per_9'] = 9 * player_stats.get('HR', 0) / ip
            features_dict['Str_pct'] = player_stats.get('Str', 0) / pit
            features_dict['GB_FB_ratio'] = player_stats.get('GB', 0) / fb
        
        return features_dict
    
    def get_player_streak_data(self, player_name: str, is_batter: bool) -> Dict:
        """
        Get streak and enhanced features for a player.
        
        Args:
            player_name: Name of the player
            is_batter: Whether the player is a batter (False for pitcher)
            
        Returns:
            Dictionary of streak features
        """
        streak_features = {}
        
        if is_batter:
            # Get batting streak data
            streak_data = self.batting_data[
                (self.batting_data['Batting'] == player_name) 
            ].sort_values('Date', ascending=False)
            
            if not streak_data.empty:
                streak_features['hit_streak'] = streak_data['hit_streak'].iloc[0] if 'hit_streak' in streak_data.columns else 0
                streak_features['rolling_5_avg'] = streak_data['rolling_5_avg'].iloc[0] if 'rolling_5_avg' in streak_data.columns else 0
                streak_features['rolling_10_avg'] = streak_data['rolling_10_avg'].iloc[0] if 'rolling_10_avg' in streak_data.columns else 0
                streak_features['opponent_defense'] = streak_data['opponent_defense'].iloc[0] if 'opponent_defense' in streak_data.columns else 1.0
                streak_features['ballpark_factor'] = streak_data['ballpark_factor'].iloc[0] if 'ballpark_factor' in streak_data.columns else 1.0
                streak_features['ISO'] = streak_data['ISO'].iloc[0] if 'ISO' in streak_data.columns else 0
                streak_features['BABIP'] = streak_data['BABIP'].iloc[0] if 'BABIP' in streak_data.columns else 0
                
        else:
            # Get pitching streak data
            streak_data = self.pitching_data[
                (self.pitching_data['Pitching'] == player_name)
            ].sort_values('Date', ascending=False)
            
            if not streak_data.empty:
                streak_features['quality_start_streak'] = streak_data['quality_start_streak'].iloc[0] if 'quality_start_streak' in streak_data.columns else 0
                streak_features['rolling_3_ERA'] = streak_data['rolling_3_ERA'].iloc[0] if 'rolling_3_ERA' in streak_data.columns else 0
                streak_features['opponent_offense'] = streak_data['opponent_offense'].iloc[0] if 'opponent_offense' in streak_data.columns else 1.0
                streak_features['ballpark_factor'] = streak_data['ballpark_factor'].iloc[0] if 'ballpark_factor' in streak_data.columns else 1.0
                streak_features['FIP'] = streak_data['FIP'].iloc[0] if 'FIP' in streak_data.columns else 0
                streak_features['BABIP_against'] = streak_data['BABIP_against'].iloc[0] if 'BABIP_against' in streak_data.columns else 0
                streak_features['days_rest'] = streak_data['days_rest'].iloc[0] if 'days_rest' in streak_data.columns else 0
        
        return streak_features

    def get_prediction_model(self, model_dict: Dict, metric: str):
        """
        Select the best model to use for prediction.
        
        Args:
            model_dict: Dictionary of available models
            metric: Statistical metric to predict
            
        Returns:
            The selected model
        """
        if isinstance(model_dict, dict):
            # Choose best model in order of preference
            if 'ensemble' in model_dict:
                return model_dict['ensemble']
            elif 'stacking' in model_dict:
                return model_dict['stacking']
            elif 'nn' in model_dict:
                return model_dict['nn']
            else:
                # Use first available model
                return next(iter(model_dict.values()))
        else:
            # If model_dict is already a model, return it
            return model_dict
        
    def match_features_to_model(self, features_df: pd.DataFrame, model) -> pd.DataFrame:
        """
        Ensure features match the model's expected input.
        
        Args:
            features_df: DataFrame of features
            model: Model to match features for
            
        Returns:
            DataFrame with features matched to model requirements
        """
        if hasattr(model, 'feature_names_in_'):
            required_features = model.feature_names_in_
            
            # Add missing features with zero values
            for feature in required_features:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            
            # Ensure correct column order and remove extra columns
            return features_df[required_features]
        
        # If model doesn't specify required features, return as is
        return features_df
        
    def calculate_confidence_interval(self, prediction: float, model, features: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate confidence interval for a prediction.
        
        Args:
            prediction: The predicted value
            model: The model used for prediction
            features: Features used for prediction
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Calculate standard deviation
        std_dev = np.std(model.predict(features))
        
        # For ensemble models, use predictions from component models if available
        if hasattr(model, 'estimators_'):
            predictions = []
            for estimator in model.estimators_:
                try:
                    predictions.append(estimator.predict(features)[0])
                except:
                    pass
            
            if predictions:
                std_dev = np.std(predictions)
        
        # Calculate confidence interval (95% confidence)
        lower_bound = max(0, prediction - 1.96 * std_dev)
        upper_bound = prediction + 1.96 * std_dev
        
        return (lower_bound, upper_bound)
    
    def predict_player_performance(self, player_name: str, is_home_game: bool, 
                              opponent: str, target_date: str) -> Dict[str, Dict[str, float]]:
        """
        Predict player performance for an upcoming game.
        
        Args:
            player_name: Name of the player to predict performance for
            is_home_game: Whether the player's team is the home team
            opponent: Name of the opposing team
            target_date: The date of the game to predict
            
        Returns:
            Dictionary containing predicted performance metrics and confidence intervals
        """
        try:
            # Input validation
            if not player_name or not opponent:
                self.logger.error("Missing required parameters: player_name or opponent")
                return {'prediction': {}, 'confidence': {}}
            
            # Determine player type
            is_batter = False
            is_pitcher = False
            player_team = None
            
            # Check if player is a batter
            batter_info = self.batting_data[self.batting_data['Batting'].str.contains(player_name, case=True, na=False)]
            if not batter_info.empty:
                is_batter = True
                player_team = batter_info['Team'].iloc[0]
            
            # Check if player is a pitcher
            pitcher_info = self.pitching_data[self.pitching_data['Pitching'].str.contains(player_name, case=True, na=False)]
            if not pitcher_info.empty:
                is_pitcher = True
                player_team = pitcher_info['Team'].iloc[0]
            
            # Handle unknown players
            if not is_batter and not is_pitcher:
                self.logger.warning(f"Player {player_name} not found in batting or pitching data")
                return {'prediction': {}, 'confidence': {}, 'error': 'player_not_found'}
            
            # Get player's historical statistics
            full_player_stats = self.get_player_stats(player_name, target_date)
            
            # Select appropriate stats based on home/away and sample size
            if is_home_game:
                player_stats = full_player_stats['home']
                if full_player_stats['overall']['games'] <= 5:
                    player_stats = full_player_stats['overall']
                if player_stats is None:
                    player_stats = full_player_stats['overall']
            else:
                player_stats = full_player_stats['away']
                if full_player_stats['overall']['games'] <= 5:
                    player_stats = full_player_stats['overall']
                if player_stats is None:
                    player_stats = full_player_stats['overall']
            
            # Prepare results dictionary
            prediction_result = {
                'prediction': {},
                'confidence': {}
            }
            
            # Prepare feature dictionary
            features_dict = self.prepare_player_features(player_name, player_stats, is_home_game, is_batter)
            
            # Make predictions based on player type
            model_type = 'batting_models' if is_batter else 'pitching_models'
            model_dict = getattr(self, model_type, {})
            
            for metric, models in model_dict.items():
                # Skip IP prediction for pitchers
                if metric == 'IP' and is_pitcher:
                    prediction_result['prediction'][metric] = player_stats.get('IP', 0) / player_stats.get('games', 1)
                    prediction_result['confidence'][metric] = (
                        prediction_result['prediction'][metric] * 0.7,
                        prediction_result['prediction'][metric] * 1.3
                    )
                    continue
                
                # Get appropriate model for this metric
                model = self.get_prediction_model(models, metric)
                
                # Create feature DataFrame
                features = pd.DataFrame([features_dict])
                
                # Match features to model requirements
                features = self.match_features_to_model(features, model)
                
                # Make prediction
                try:
                    prediction = model.predict(features)[0]
                    
                    # Convert numpy types to Python types
                    if isinstance(prediction, (np.integer, np.floating)):
                        prediction = float(prediction)
                    
                    # Calculate confidence interval
                    confidence_interval = self.calculate_confidence_interval(prediction, model, features)
                    
                    # Store prediction and confidence interval
                    prediction_result['prediction'][metric] = prediction
                    prediction_result['confidence'][metric] = confidence_interval
                    
                except Exception as e:
                    self.logger.warning(f"Error predicting {metric} for {player_name}: {e}")
            
            # Add metadata for debugging and analysis
            prediction_result['metadata'] = {
                'player': player_name,
                'team': player_team,
                'position': 'batter' if is_batter else 'pitcher',
                'game_date': target_date,
                'is_home': is_home_game,
                'opponent': opponent,
                'sample_size': player_stats.get('games', 0)
            }
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Error predicting player performance for {player_name}: {e}")
            return {'prediction': {}, 'confidence': {}, 'error': str(e)}
    
    def get_head_to_head_stats(self, home_team: str, away_team: str, target_date: str) -> Dict[str, float]:
        """
        Get head-to-head statistics between two teams up to a specific date.
        
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            target_date (str): The reference date for retrieving historical data
            
        Returns:
            Dict[str, float]: Dictionary containing head-to-head statistics
        """
        try:
            target_date = pd.to_datetime(target_date)
            
            # Filter linescore data to include only games before the target date
            filtered_linescore = self.linescore_data[self.linescore_data['Date'] < target_date]
            
            # Initialize head-to-head stats
            home_wins = 0
            away_wins = 0
            home_runs_scored = 0
            away_runs_scored = 0
            recent_games = []  # To track most recent matchups (more weight)
            
            # Process the linescore data two rows at a time
            i = 0
            while i < len(filtered_linescore) - 1:
                try:
                    # Get the away team row and home team row
                    away_row = filtered_linescore.iloc[i]
                    home_row = filtered_linescore.iloc[i+1]
                    
                    # Verify that these rows are for the same game
                    if away_row['Date'] != home_row['Date'] or away_row['AwayTeam'] != home_row['AwayTeam'] or away_row['HomeTeam'] != home_row['HomeTeam']:
                        # These rows don't represent the same game, skip to next row
                        i += 1
                        continue
                    
                    # Check if this is a game between the two teams we're interested in
                    is_matchup = (
                        (home_row['HomeTeam'] == home_team and away_row['AwayTeam'] == away_team) or
                        (home_row['HomeTeam'] == away_team and away_row['AwayTeam'] == home_team)
                    )
                    
                    if is_matchup:
                        game_date = home_row['Date']
                        curr_home_team = home_row['HomeTeam']
                        curr_away_team = away_row['AwayTeam']
                        
                        # Get the scores
                        try:
                            away_runs = float(away_row.get('R', 0))
                            home_runs = float(home_row.get('R', 0))
                            
                            # Store the matchup result
                            result = {
                                'date': game_date,
                                'home_team': curr_home_team,
                                'away_team': curr_away_team,
                                'home_runs': home_runs,
                                'away_runs': away_runs,
                                'home_win': home_runs > away_runs
                            }
                            recent_games.append(result)
                            
                            # Update stats based on which team was home
                            if curr_home_team == home_team and curr_away_team == away_team:
                                # Regular orientation (home_team at home)
                                if home_runs > away_runs:
                                    home_wins += 1
                                else:
                                    away_wins += 1
                                home_runs_scored += home_runs
                                away_runs_scored += away_runs
                            else:
                                # Reverse orientation (home_team was away)
                                if home_runs > away_runs:
                                    away_wins += 1
                                else:
                                    home_wins += 1
                                home_runs_scored += away_runs
                                away_runs_scored += home_runs
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Error processing H2H game scores: {e}")
                    
                    # Move to the next game (skip two rows)
                    i += 2
                    
                except Exception as e:
                    self.logger.warning(f"Error processing game at index {i}: {e}")
                    i += 1  # Move to next row if there's an error
            
            # Calculate statistics
            total_games = home_wins + away_wins
            home_win_pct = home_wins / total_games if total_games > 0 else 0.5
            
            # Calculate run differential
            home_rpg = home_runs_scored / total_games if total_games > 0 else 0
            away_rpg = away_runs_scored / total_games if total_games > 0 else 0
            
            # Calculate recency-weighted win percentage (more weight to recent games)
            # Sort games by date
            recent_games.sort(key=lambda x: x['date'])
            
            # Calculate weighted win percentage if we have games
            weighted_home_wins = 0
            total_weight = 0
            
            for i, game in enumerate(recent_games):
                # Exponential weight based on recency
                weight = 1.5 ** i  # More recent games get higher weight
                total_weight += weight
                
                # Check if home team won (accounting for possible venue switch)
                home_team_won = (
                    (game['home_team'] == home_team and game['home_win']) or
                    (game['away_team'] == home_team and not game['home_win'])
                )
                
                if home_team_won:
                    weighted_home_wins += weight
            
            # Calculate weighted win percentage
            weighted_home_win_pct = weighted_home_wins / total_weight if total_weight > 0 else 0.5
            
            # Blend regular and weighted win percentages, favoring weighted if we have enough games
            blended_win_pct = home_win_pct
            if total_games >= 5:
                recency_weight = min(0.7, 0.3 + (total_games / 20) * 0.4)  # 30% for 5 games, up to 70% for 20+ games
                blended_win_pct = (home_win_pct * (1 - recency_weight)) + (weighted_home_win_pct * recency_weight)
            
            return {
                'home_wins': home_wins,
                'away_wins': away_wins,
                'total_games': total_games,
                'home_win_pct': home_win_pct,
                'weighted_home_win_pct': weighted_home_win_pct,
                'blended_win_pct': blended_win_pct,
                'home_rpg': home_rpg,
                'away_rpg': away_rpg,
                'run_differential': home_rpg - away_rpg,
                'recent_games': recent_games[-5:] if recent_games else []  # Include last 5 games
            }
                
        except Exception as e:
            self.logger.error(f"Error getting head-to-head stats for {home_team} vs {away_team}: {e}")
            return {'home_win_pct': 0.5, 'total_games': 0}
        
    def get_best_team_model(self, team: str):
        """
        Select the best available model for a team, with fallbacks.
        
        Args:
            team (str): Team name to find a model for
            
        Returns:
            Model object or None if no suitable model found
        """
        # Direct match
        if team in self.team_models:
            self.logger.info(f"Using exact match model for {team}")
            return self.team_models[team]
        
        # Try partial name matching
        for model_team in self.team_models:
            if team in model_team or model_team in team:
                self.logger.info(f"Using partial match model for {team} (matched with {model_team})")
                return self.team_models[model_team]
        
        # Try league average model as fallback
        if 'league_average' in self.team_models:
            self.logger.info(f"Using league average model for {team} (no specific model found)")
            return self.team_models['league_average']
        
        # If no suitable model found, return None
        self.logger.warning(f"No suitable model found for {team}")
        return None
    
    def calculate_momentum_factor(self, team_name: str, target_date: str) -> float:
        """
        Calculate a momentum factor based on recent performance.
        
        Args:
            team_name (str): Name of the team
            target_date (str): Target date for prediction
            
        Returns:
            float: Momentum factor (1.0 is neutral, higher values mean positive momentum)
        """
        momentum = 1.0
        
        # Use team streaks if available
        if hasattr(self, 'team_streaks') and team_name in self.team_streaks:
            win_streak = self.team_streaks[team_name].get('win_streak', 0)
            recent_form = self.team_streaks[team_name].get('recent_win_pct', 0.5)
            
            # Adjust for win streak (capped at 0.1 adjustment)
            streak_factor = min(0.1, win_streak * 0.02)
            
            # Adjust for recent form (difference from 0.5)
            form_factor = (recent_form - 0.5) * 0.2
            
            momentum += streak_factor + form_factor
        
        # Calculate run differentials over last N games if available
        if hasattr(self, 'team_records') and team_name in self.team_records.index:
            run_diff = self.team_records.loc[team_name, 'run_differential']
            run_diff_per_game = run_diff / max(1, self.team_records.loc[team_name, 'Wins'] + self.team_records.loc[team_name, 'Losses'])
            
            # Add small adjustment based on run differential
            if abs(run_diff_per_game) > 0.5:
                run_diff_factor = min(0.1, abs(run_diff_per_game) * 0.02) * (1 if run_diff_per_game > 0 else -1)
                momentum += run_diff_factor
        
        return momentum

    def calculate_prediction_confidence(self, home_win_prob: float, model_count: int, 
                               h2h_games: int, total_games: int) -> str:
        """
        Calculate prediction confidence level based on multiple factors.
        
        Args:
            home_win_prob (float): Predicted home win probability
            model_count (int): Number of models used for prediction
            h2h_games (int): Number of head-to-head games in history
            total_games (int): Total games played this season
            
        Returns:
            str: Confidence level description
        """
        # Base confidence on deviation from 50%
        prob_confidence = abs(home_win_prob - 0.5)
        
        # Scale based on number of models used
        model_factor = min(1.0, model_count / 2)
        
        # Scale based on head-to-head history
        h2h_factor = min(1.0, h2h_games / 10)
        
        # Scale based on total games played this season
        season_factor = min(1.0, total_games / 40)
        
        # Calculate weighted confidence score
        confidence_score = (
            prob_confidence * 0.5 +
            model_factor * 0.2 + 
            h2h_factor * 0.2 +
            season_factor * 0.1
        )
        
        # Convert to confidence levels
        if confidence_score < 0.15:
            return "low"
        elif confidence_score < 0.3:
            return "medium-low"
        elif confidence_score < 0.45:
            return "medium"
        elif confidence_score < 0.6:
            return "medium-high"
        else:
            return "high"
        
    def generate_prediction_explanation(self, home_team: str, away_team: str, 
                                   home_win_prob: float, h2h_stats: Dict) -> str:
        """
        Generate human-readable explanation for the prediction.
        
        Args:
            home_team (str): Home team name
            away_team (str): Away team name
            home_win_prob (float): Predicted home win probability
            h2h_stats (Dict): Head-to-head statistics dictionary
            
        Returns:
            str: Explanation of prediction factors
        """
        explanation = []
        
        # Basic win probability statement
        win_pct = home_win_prob * 100
        if win_pct > 50:
            explanation.append(f"{home_team} has a {win_pct:.1f}% chance to win at home against {away_team}.")
        else:
            explanation.append(f"{away_team} has a {(100-win_pct):.1f}% chance to win on the road against {home_team}.")
        
        # Add head-to-head context
        h2h_games = h2h_stats.get('total_games', 0)
        if h2h_games >= 3:
            h2h_win_pct = h2h_stats.get('home_win_pct', 0.5) * 100
            explanation.append(f"In {h2h_games} previous matchups, {home_team} has won {h2h_win_pct:.1f}% of games against {away_team}.")
        
        # Add team form context
        if hasattr(self, 'team_streaks'):
            home_streak = self.team_streaks.get(home_team, {}).get('win_streak', 0)
            away_streak = self.team_streaks.get(away_team, {}).get('win_streak', 0)
            
            if home_streak > 2:
                explanation.append(f"{home_team} is on a {home_streak}-game winning streak.")
            elif home_streak == 0:
                explanation.append(f"{home_team} has not won their last game.")
                
            if away_streak > 2:
                explanation.append(f"{away_team} is on a {away_streak}-game winning streak.")
            elif away_streak == 0:
                explanation.append(f"{away_team} has not won their last game.")
        
        # Add team stat comparison if available
        if hasattr(self, 'team_records'):
            if home_team in self.team_records.index and away_team in self.team_records.index:
                home_win_pct = self.team_records.loc[home_team, 'WinPct'] * 100
                away_win_pct = self.team_records.loc[away_team, 'WinPct'] * 100
                
                if abs(home_win_pct - away_win_pct) > 10:
                    better_team = home_team if home_win_pct > away_win_pct else away_team
                    worse_team = away_team if home_win_pct > away_win_pct else home_team
                    win_diff = abs(home_win_pct - away_win_pct)
                    explanation.append(f"{better_team} has a {win_diff:.1f}% better overall win percentage than {worse_team}.")
        
        return " ".join(explanation)
 
    def predict_game_winner(self, home_team: str, away_team: str, 
            target_date: str) -> Dict[str, float]:
        """
        Predict the winner of an upcoming game using team-specific models with enhanced features.
        
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            target_date (str): The date of the game to predict
            
        Returns:
            Dict: Enhanced prediction results with detailed information
        """
        try:
            # Get team statistics
            home_team_stats = self.get_team_stats(home_team, target_date)
            away_team_stats = self.get_team_stats(away_team, target_date)
            
            # Get head-to-head statistics up to the target date
            self.h2h_stats = self.get_head_to_head_stats(home_team, away_team, target_date)
            
            if not home_team_stats or not away_team_stats:
                self.logger.warning(f"Missing team stats for {home_team} or {away_team}")
                return {
                    'home_win_probability': 0.5,
                    'away_win_probability': 0.5,
                    'confidence': 'low',
                    'predicted_winner': home_team if random.random() > 0.5 else away_team
                }
            
            # Create enhanced feature vectors with all available metrics
            home_features = pd.DataFrame({
                # Basic team stats
                'TeamWinPct': [home_team_stats.get('win_pct', 0)],
                'OpponentWinPct': [away_team_stats.get('win_pct', 0)],
                'IsHome': [1],
                'HomeTeamHomeWinPct': [home_team_stats.get('home_win_pct', 0)],
                'AwayTeamAwayWinPct': [away_team_stats.get('away_win_pct', 0)],
                
                # Head-to-head metrics
                'H2HWinPct': [self.h2h_stats.get('blended_win_pct', 0.5)],
                'H2HGameCount': [self.h2h_stats.get('total_games', 0)],
                'H2HRunDiff': [self.h2h_stats.get('run_differential', 0)],
                
                # Run production/prevention
                'RunsPerGame': [home_team_stats.get('rpg', 0)],
                'RunsAllowedPerGame': [home_team_stats.get('rapg', 0)],
                'RunDifferential': [home_team_stats.get('run_differential', 0)],
                
                # Team streaks if available
                'WinStreak': [self.team_streaks.get(home_team, {}).get('win_streak', 0) if hasattr(self, 'team_streaks') else 0],
                'RecentForm': [self.team_streaks.get(home_team, {}).get('recent_win_pct', 0.5) if hasattr(self, 'team_streaks') else 0.5],
                
                # Ballpark factor
                'BallparkFactor': [self.ballpark_factors.get(home_team, 1.0) if hasattr(self, 'ballpark_factors') else 1.0]
            })
            
            # Create similar enhanced feature vector for away team
            away_features = pd.DataFrame({
                'TeamWinPct': [away_team_stats.get('win_pct', 0)],
                'OpponentWinPct': [home_team_stats.get('win_pct', 0)],
                'IsHome': [0],
                'HomeTeamHomeWinPct': [home_team_stats.get('home_win_pct', 0)],
                'AwayTeamAwayWinPct': [away_team_stats.get('away_win_pct', 0)],
                'H2HWinPct': [1 - self.h2h_stats.get('blended_win_pct', 0.5)],
                'H2HGameCount': [self.h2h_stats.get('total_games', 0)],
                'H2HRunDiff': [-self.h2h_stats.get('run_differential', 0)],
                'RunsPerGame': [away_team_stats.get('rpg', 0)],
                'RunsAllowedPerGame': [away_team_stats.get('rapg', 0)],
                'RunDifferential': [away_team_stats.get('run_differential', 0)],
                'WinStreak': [self.team_streaks.get(away_team, {}).get('win_streak', 0) if hasattr(self, 'team_streaks') else 0],
                'RecentForm': [self.team_streaks.get(away_team, {}).get('recent_win_pct', 0.5) if hasattr(self, 'team_streaks') else 0.5],
                'BallparkFactor': [self.ballpark_factors.get(home_team, 1.0) if hasattr(self, 'ballpark_factors') else 1.0]
            })
            
            # Get best models for home and away teams
            home_model = self.get_best_team_model(home_team)
            away_model = self.get_best_team_model(away_team)
            
            # Initialize win probability arrays and model tracking
            home_win_probs = []
            model_weights = []
            used_models = []
            
            # Use home team model if available
            if home_model:
                try:
                    # Need to ensure features match what the model expects
                    home_team_win_prob = home_model.predict_proba(home_features)[0][1]
                    home_win_probs.append(home_team_win_prob)
                    
                    # Weight based on model quality
                    model_weight = 1.0
                    if home_team in self.team_models:  # Exact match gets higher weight
                        model_weight = 1.5
                    model_weights.append(model_weight)
                    used_models.append(f"home_{home_team}")
                    
                    self.logger.info(f"Home team model predicts: {home_team_win_prob:.4f}")
                except Exception as e:
                    self.logger.warning(f"Error using home team model: {e}")
            
            # Use away team model if available
            if away_model:
                try:
                    away_team_win_prob = away_model.predict_proba(away_features)[0][1]
                    # Convert to home win probability
                    away_home_win_prob = 1 - away_team_win_prob
                    home_win_probs.append(away_home_win_prob)
                    
                    # Weight based on model quality
                    model_weight = 1.0
                    if away_team in self.team_models:  # Exact match gets higher weight
                        model_weight = 1.5
                    model_weights.append(model_weight)
                    used_models.append(f"away_{away_team}")
                    
                    self.logger.info(f"Away team model predicts: {away_home_win_prob:.4f}")
                except Exception as e:
                    self.logger.warning(f"Error using away team model: {e}")
            
            # Apply momentum adjustments
            home_momentum = self.calculate_momentum_factor(home_team, target_date)
            away_momentum = self.calculate_momentum_factor(away_team, target_date)
            momentum_advantage = home_momentum - away_momentum
            
            # Calculate head-to-head adjustment
            h2h_adjustment = 0
            h2h_games = self.h2h_stats.get('total_games', 0)
            
            if h2h_games >= 3:
                h2h_win_pct = self.h2h_stats.get('blended_win_pct', 0.5)
                h2h_advantage = h2h_win_pct - 0.5
                
                # Adjustment strength increases with more games but caps at a certain level
                h2h_weight = min(0.2, 0.05 + (h2h_games / 30) * 0.15)
                h2h_adjustment = h2h_advantage * h2h_weight
                
                self.logger.info(f"H2H adjustment: {h2h_adjustment:.4f} (based on {h2h_games} games)")
            
            # Calculate final win probability
            if home_win_probs:
                # Calculate weighted average of model predictions
                base_home_win_prob = sum(p * w for p, w in zip(home_win_probs, model_weights)) / sum(model_weights)
                
                # Apply momentum and head-to-head adjustments
                adjusted_home_win_prob = base_home_win_prob
                adjusted_home_win_prob += momentum_advantage * 0.05  # Scale momentum effect
                adjusted_home_win_prob += h2h_adjustment
                
                # Ensure probability stays in valid range
                home_win_prob = min(0.95, max(0.05, adjusted_home_win_prob))
            else:
                # Fallback if no models available - use head-to-head and momentum
                self.logger.info("No team models available, using heuristic")
                
                # Start with neutral probability and apply adjustments
                home_win_prob = 0.54  # Slight home field advantage
                
                # Apply head-to-head adjustment
                if h2h_games >= 3:
                    home_win_prob = (home_win_prob * 0.7) + (self.h2h_stats.get('blended_win_pct', 0.5) * 0.3)
                
                # Apply momentum adjustment
                home_win_prob += momentum_advantage * 0.05
                
                # Apply win percentage difference
                win_pct_diff = home_team_stats.get('win_pct', 0.5) - away_team_stats.get('win_pct', 0.5)
                home_win_prob += win_pct_diff * 0.2
                
                # Apply home/away record adjustment
                home_home_advantage = home_team_stats.get('home_win_pct', 0.5) - 0.5
                away_road_disadvantage = 0.5 - away_team_stats.get('away_win_pct', 0.5)
                home_win_prob += (home_home_advantage + away_road_disadvantage) * 0.1
                
                # Ensure probability stays in valid range
                home_win_prob = min(0.95, max(0.05, home_win_prob))
            
            # Determine predicted winner
            predicted_winner = home_team if home_win_prob > 0.5 else away_team
            win_probability = home_win_prob if predicted_winner == home_team else (1 - home_win_prob)
            
            # Calculate win margin estimation based on win probability
            probability_diff = abs(home_win_prob - 0.5)
            base_margin = probability_diff * 6  # 6 runs for a 100% certain win
            run_diff_component = abs(home_team_stats.get('run_differential', 0) - away_team_stats.get('run_differential', 0)) / 10
            estimated_margin = max(1, (base_margin + run_diff_component))
            
            if home_win_prob > 0.5:
                win_margin = estimated_margin
            else:
                win_margin = -estimated_margin
                
            # Calculate confidence
            confidence_level = self.calculate_prediction_confidence(
                home_win_prob, 
                len(home_win_probs),
                h2h_games,
                home_team_stats.get('wins', 0) + home_team_stats.get('losses', 0)
            )
            
            # Generate natural language explanation
            explanation = self.generate_prediction_explanation(
                home_team, away_team, home_win_prob, self.h2h_stats
            )
            
            # Return enhanced prediction results
            return {
                'home_win_probability': home_win_prob,
                'away_win_probability': 1 - home_win_prob,
                'predicted_winner': predicted_winner,
                'win_probability': win_probability,
                'win_margin': win_margin,
                'confidence': confidence_level,
                'h2h_stats': self.h2h_stats,
                'team_form': {
                    'home_team': {
                        'win_streak': self.team_streaks.get(home_team, {}).get('win_streak', 0) if hasattr(self, 'team_streaks') else 0,
                        'recent_form': self.team_streaks.get(home_team, {}).get('recent_win_pct', 0.5) if hasattr(self, 'team_streaks') else 0.5
                    },
                    'away_team': {
                        'win_streak': self.team_streaks.get(away_team, {}).get('win_streak', 0) if hasattr(self, 'team_streaks') else 0,
                        'recent_form': self.team_streaks.get(away_team, {}).get('recent_win_pct', 0.5) if hasattr(self, 'team_streaks') else 0.5
                    }
                },
                'explanation': explanation,
                'model_contributions': {
                    'models_used': used_models,
                    'base_prediction': base_home_win_prob if home_win_probs else None,
                    'momentum_adjustment': momentum_advantage * 0.05,
                    'h2h_adjustment': h2h_adjustment,
                    'home_momentum': home_momentum,
                    'away_momentum': away_momentum
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting game winner for {home_team} vs {away_team}: {e}")
            return {
                'home_win_probability': 0.5,
                'away_win_probability': 0.5,
                'predicted_winner': home_team if random.random() > 0.5 else away_team,
                'confidence': 'low',
                'error': str(e)
            }
    
    def _extract_batting_data(self, batters):
        """Extract normalized batting data from player predictions"""
        batting_data = []
        for player in batters:
            if 'player_name' not in player:
                continue
                
            row = {
                'name': player['player_name'],
                'position': player.get('position', 'Unknown'),
                'order': player.get('batting_order', 0),
                'R': player.get('pred_R', 0),
                'H': player.get('pred_H', 0),
                'RBI': player.get('pred_RBI', 0),
                'HR': player.get('pred_HR', 0),
                'AVG': player.get('pred_AVG', 0),
                'OBP': player.get('pred_OBP', 0)
            }
            batting_data.append(row)
        return batting_data

    def _extract_pitching_data(self, pitchers):
        """Extract normalized pitching data from player predictions"""
        pitching_data = []
        for player in pitchers:
            if 'player_name' not in player:
                continue
                
            row = {
                'name': player['player_name'],
                'IP': player.get('pred_IP', 0),
                'H': player.get('pred_H', 0),
                'R': player.get('pred_R', 0),
                'ER': player.get('pred_ER', 0),
                'ERA': player.get('pred_ERA', 0),
                'WHIP': player.get('pred_WHIP', 0),
                'SO': player.get('pred_SO', 0),
                'BB': player.get('pred_BB', 0)
            }
            pitching_data.append(row)
        return pitching_data

    def _calculate_expected_runs(self, batting_data, opposing_pitching_data):
        """Calculate expected runs with weighted contributions"""
        # Direct contributions
        direct_runs = sum(player.get('R', 0) for player in batting_data)
        
        # Add production from RBIs (with compensation for double-counting)
        rbi_component = 0.7 * sum(player.get('RBI', 0) for player in batting_data)
        
        # Add HR contribution (already in R and RBI but helps model accuracy)
        hr_component = 0.2 * sum(player.get('HR', 0) for player in batting_data)
        
        # Base expected runs
        expected_runs = (direct_runs + rbi_component + hr_component) / 3
        
        # Adjust for opposing pitching quality
        pitching_factor = 1.0
        if opposing_pitching_data:
            ip_values = [p.get('IP', 0) for p in opposing_pitching_data]
            era_values = [p.get('ERA', 0) for p in opposing_pitching_data if p.get('ERA', 0) > 0]
            
            if era_values and sum(ip_values) > 0:
                # Weight ERA by IP
                weighted_era = sum(era * ip for era, ip in zip(era_values, ip_values)) / sum(ip_values)
                # League average ERA is around 4.00
                pitching_factor = weighted_era / 4.00
                
                # Cap adjustment factor
                pitching_factor = max(0.7, min(1.3, pitching_factor))
        
        # Apply pitching adjustment
        expected_runs *= pitching_factor
        
        return expected_runs
    
    def _calculate_matchup_factors(self, home_team, away_team, home_pitching, away_pitching, 
                             home_batting, away_batting):
        """Calculate team-specific matchup adjustments"""
        matchup_factors = {}
        
        # Power vs. Strikeout pitchers (home batting vs away pitching)
        home_power = sum(p.get('HR', 0) for p in home_batting) / max(1, len(home_batting))
        away_strikeouts = 0
        
        if away_pitching:
            # Extract strikeout tendency
            so_values = [p.get('SO', 0) for p in away_pitching]
            ip_values = [p.get('IP', 0) for p in away_pitching]
            
            if so_values and sum(ip_values) > 0:
                # Calculate K/9
                away_strikeouts = sum(so_values) * 9 / sum(ip_values)
        
        # High HR + low K = advantage for home
        power_vs_k_factor = 1 + min(0.1, max(-0.1, (home_power - 0.1) * (1 - (away_strikeouts / 9)) * 0.3))
        matchup_factors['home_power_vs_away_k'] = power_vs_k_factor
        
        # Same for away team
        away_power = sum(p.get('HR', 0) for p in away_batting) / max(1, len(away_batting))
        home_strikeouts = 0
        
        if home_pitching:
            so_values = [p.get('SO', 0) for p in home_pitching]
            ip_values = [p.get('IP', 0) for p in home_pitching]
            
            if so_values and sum(ip_values) > 0:
                home_strikeouts = sum(so_values) * 9 / sum(ip_values)
        
        power_vs_k_factor = 1 + min(0.1, max(-0.1, (away_power - 0.1) * (1 - (home_strikeouts / 9)) * 0.3))
        matchup_factors['away_power_vs_home_k'] = power_vs_k_factor
        
        return matchup_factors
    
    def _identify_key_contributors(self, home_batting, away_batting, home_pitching, away_pitching):
        """Identify key contributors to the expected outcome"""
        key_contributors = {'home': [], 'away': []}
        
        # Find top batters by combined R+RBI+HR
        for team, data in [('home', home_batting), ('away', away_batting)]:
            for player in data:
                contribution = player.get('R', 0) + player.get('RBI', 0) + (player.get('HR', 0) * 1.5)
                if contribution > 0:
                    key_contributors[team].append({
                        'name': player['name'],
                        'value': contribution,
                        'metric': 'R+RBI+HR',
                        'type': 'batter'
                    })
        
        # Find top pitchers by IP and low ERA
        for team, data in [('home', home_pitching), ('away', away_pitching)]:
            for player in data:
                ip = player.get('IP', 0)
                era = player.get('ERA', 5.0)
                if ip > 0:
                    # Lower ERA is better, so invert for contribution
                    contribution = ip * (6.0 - min(6.0, era)) / 2
                    key_contributors[team].append({
                        'name': player['name'],
                        'value': contribution,
                        'metric': 'IP/ERA',
                        'type': 'pitcher'
                    })
        
        # Sort contributors by value
        for team in key_contributors:
            key_contributors[team] = sorted(key_contributors[team], key=lambda x: x['value'], reverse=True)[:3]
        
        return key_contributors
    
    def _calculate_confidence_score(self, team_model_confidence, model_agreement, player_data_quality):
        """Calculate an integrated confidence score for game prediction"""
        # Base confidence from team model
        confidence_map = {
            'low': 0.3,
            'medium-low': 0.4,
            'medium': 0.5,
            'medium-high': 0.7,
            'high': 0.9
        }
        
        confidence = confidence_map.get(team_model_confidence, 0.5)
        
        # Adjust for model agreement
        if model_agreement:
            confidence *= 1.2
        else:
            confidence *= 0.8
        
        # Adjust for player data quality
        confidence *= player_data_quality
        
        # Ensure confidence is in valid range
        confidence = max(0.1, min(1.0, confidence))
        
        # Map back to descriptive confidence
        if confidence < 0.35:
            return 'low'
        elif confidence < 0.45:
            return 'medium-low'
        elif confidence < 0.65:
            return 'medium'
        elif confidence < 0.8:
            return 'medium-high'
        else:
            return 'high'
        
    def _generate_game_analysis_explanation(self, home_team, away_team, winner, win_prob, 
                                     score_prediction, key_contributors, model_agreement, 
                                     matchup_factors):
        """Generate a detailed explanation of the game analysis in natural language"""
        explanation = []
        
        # Explain the overall prediction
        explanation.append(f"{winner} is projected to win with {win_prob*100:.1f}% probability.")
        
        # Add score prediction
        explanation.append(f"Projected score: {score_prediction}")
        
        # Explain key contributors
        home_contributors = key_contributors.get('home', [])
        away_contributors = key_contributors.get('away', [])
        
        if home_contributors:
            top_player = home_contributors[0]
            explanation.append(f"Key contributor for {home_team}: {top_player['name']} with expected "
                            f"{top_player['metric']} of {top_player['value']:.1f}.")
        
        if away_contributors:
            top_player = away_contributors[0]
            explanation.append(f"Key contributor for {away_team}: {top_player['name']} with expected "
                            f"{top_player['metric']} of {top_player['value']:.1f}.")
        
        # Add context about model agreement
        if model_agreement:
            explanation.append("Both team-level and player-level models agree on this prediction.")
        else:
            explanation.append("Note: Team and player models show different projections, reducing confidence.")
        
        # Add matchup insights
        significant_factors = {k: v for k, v in matchup_factors.items() if abs(v - 1.0) > 0.05}
        if significant_factors:
            most_significant = max(significant_factors.items(), key=lambda x: abs(x[1] - 1.0))
            factor_name = most_significant[0].replace('_', ' ').title()
            factor_impact = (most_significant[1] - 1.0) * 100
            
            if factor_impact > 0:
                explanation.append(f"Matchup advantage: {factor_name} provides a {factor_impact:.1f}% boost.")
            else:
                explanation.append(f"Matchup challenge: {factor_name} causes a {abs(factor_impact):.1f}% reduction.")
        
        return " ".join(explanation)
    
    def analyze_game_outcome(self, game_prediction: Dict[str, any], 
                    home_players: List[Dict], away_players: List[Dict],
                    home_team: str, away_team: str) -> Dict[str, any]:
        """
        Enhanced game analysis with robust error handling and improved metrics.
        
        Args:
            game_prediction: Team model's game prediction dictionary
            home_players: List of home team player prediction dictionaries
            away_players: List of away team player prediction dictionaries
            home_team: Name of the home team
            away_team: Name of the away team
            
        Returns:
            Dict: Enhanced game prediction with detailed analysis
        """
        try:
            # Early validation of input data
            if not home_players or not away_players:
                self.logger.warning(f"Missing player data for {home_team} vs {away_team}")
                return {
                    'home_team': home_team,
                    'away_team': away_team,
                    'predicted_winner': game_prediction.get('predicted_winner', home_team if random.random() > 0.5 else away_team),
                    'error': 'insufficient_player_data',
                    'original_prediction': game_prediction
                }
                
            # Separate batters and pitchers
            home_batters = [p for p in home_players if p.get('position') != 'P']
            home_pitchers = [p for p in home_players if p.get('position') == 'P']
            away_batters = [p for p in away_players if p.get('position') != 'P']
            away_pitchers = [p for p in away_players if p.get('position') == 'P']
            
            # Calculate player data quality (used for confidence calculation)
            home_data_quality = min(1.0, len(home_batters) / 9) * min(1.0, len(home_pitchers) / 1)
            away_data_quality = min(1.0, len(away_batters) / 9) * min(1.0, len(away_pitchers) / 1)
            player_data_quality = (home_data_quality + away_data_quality) / 2
            
            # Process batting and pitching data using helper methods
            home_batting_data = self._extract_batting_data(home_batters)
            away_batting_data = self._extract_batting_data(away_batters)
            home_pitching_data = self._extract_pitching_data(home_pitchers)
            away_pitching_data = self._extract_pitching_data(away_pitchers)
            
            # Calculate expected runs with enhanced methodology
            home_expected_runs = self._calculate_expected_runs(home_batting_data, away_pitching_data)
            away_expected_runs = self._calculate_expected_runs(away_batting_data, home_pitching_data)
            
            # Incorporate head-to-head history
            h2h_stats = self.h2h_stats if hasattr(self, 'h2h_stats') else game_prediction.get('h2h_stats', {})
            h2h_adjustment = 0
            
            if h2h_stats and h2h_stats.get('total_games', 0) >= 3:
                h2h_win_pct = h2h_stats.get('blended_win_pct', 0.5)
                h2h_home_advantage = h2h_win_pct - 0.5
                
                # Scale by recency and sample size
                recency_weight = min(0.15, 0.05 + (h2h_stats.get('total_games', 0) / 20) * 0.1)
                h2h_adjustment = h2h_home_advantage * recency_weight
                
                # Factor in run differential as well
                run_diff = h2h_stats.get('run_differential', 0)
                if abs(run_diff) > 1.0:
                    run_diff_adjustment = min(0.1, abs(run_diff) * 0.02) * (1 if run_diff > 0 else -1)
                    h2h_adjustment += run_diff_adjustment
            
            # Calculate matchup-specific factors
            matchup_factors = self._calculate_matchup_factors(home_team, away_team, 
                                                        home_pitching_data, away_pitching_data,
                                                        home_batting_data, away_batting_data)
            
            # Apply matchup factors to expected runs
            for factor, value in matchup_factors.items():
                if 'home' in factor:
                    home_expected_runs *= value
                elif 'away' in factor:
                    away_expected_runs *= value
            
            # Player model win probability
            player_model_home_win_prob = 0.5
            if home_expected_runs != away_expected_runs:
                # Calculate win probability based on Pythagoras expectation formula
                run_diff = home_expected_runs - away_expected_runs
                player_model_home_win_prob = 0.5 + (run_diff / (abs(run_diff) + 4)) * 0.5
                
                # Ensure probability is within valid range
                player_model_home_win_prob = max(0.05, min(0.95, player_model_home_win_prob))
            
            # Get team model probability
            team_model_home_win_prob = game_prediction.get('home_win_probability', 0.5)
            
            # Determine model agreement
            player_model_winner = home_team if player_model_home_win_prob > 0.5 else away_team
            team_model_winner = game_prediction.get('predicted_winner', 
                                                home_team if team_model_home_win_prob > 0.5 else away_team)
            model_agreement = player_model_winner == team_model_winner
            
            # Weighted model combination with adaptive weights
            if model_agreement:
                # Models agree, weight based on confidence
                team_weight = 0.6
                player_weight = 0.4
            else:
                # Models disagree, adjust weights based on data quality
                team_quality = game_prediction.get('confidence', 'medium')
                team_quality_map = {'low': 0.3, 'medium-low': 0.4, 'medium': 0.5, 'medium-high': 0.7, 'high': 0.8}
                team_quality_factor = team_quality_map.get(team_quality, 0.5)
                
                # If player data quality is high, give it more weight in disagreement
                team_weight = min(0.5, max(0.3, team_quality_factor))
                player_weight = 1 - team_weight
            
            # Calculate combined probability
            combined_home_win_prob = (team_model_home_win_prob * team_weight) + (player_model_home_win_prob * player_weight) + h2h_adjustment
            
            # Ensure probability is within valid range
            combined_home_win_prob = max(0.05, min(0.95, combined_home_win_prob))
            
            # Determine final prediction
            final_winner = home_team if combined_home_win_prob > 0.5 else away_team
            final_win_prob = combined_home_win_prob if final_winner == home_team else (1 - combined_home_win_prob)
            
            # Calculate expected score (rounded to nearest integer)
            home_score = max(0, round(home_expected_runs))
            away_score = max(0, round(away_expected_runs))
            
            # Ensure score is consistent with winner
            if (home_score > away_score and final_winner == away_team) or (home_score < away_score and final_winner == home_team):
                if final_winner == home_team:
                    home_score = away_score + 1
                else:
                    away_score = home_score + 1
            
            # Calculate confidence level
            confidence = self._calculate_confidence_score(
                game_prediction.get('confidence', 'medium'),
                model_agreement,
                player_data_quality
            )
            
            # Identify key contributors to predicted outcome
            key_contributors = self._identify_key_contributors(
                home_batting_data, away_batting_data,
                home_pitching_data, away_pitching_data
            )
            
            # Generate detailed explanation
            explanation = self._generate_game_analysis_explanation(
                home_team, away_team,
                final_winner, final_win_prob,
                f"{away_team} {away_score}, {home_team} {home_score}",
                key_contributors,
                model_agreement,
                matchup_factors
            )
            
            # Create detailed analysis result
            analysis_result = {
                'home_team': home_team,
                'away_team': away_team,
                'home_batting_stats': home_batting_data,
                'away_batting_stats': away_batting_data,
                'home_pitching_stats': home_pitching_data,
                'away_pitching_stats': away_pitching_data,
                'home_runs_scored': home_expected_runs,
                'home_runs_allowed': away_expected_runs,
                'away_runs_scored': away_expected_runs,
                'away_runs_allowed': home_expected_runs,
                'player_model_winner': player_model_winner,
                'player_model_home_win_prob': player_model_home_win_prob,
                'team_model_winner': team_model_winner,
                'team_model_home_win_prob': team_model_home_win_prob,
                'h2h_adjustment': h2h_adjustment,
                'matchup_factors': matchup_factors,
                'combined_home_win_prob': combined_home_win_prob,
                'predicted_winner': final_winner,
                'win_probability': final_win_prob,
                'predicted_score': f"{away_team} {away_score}, {home_team} {home_score}",
                'agreement': model_agreement,
                'confidence': confidence,
                'key_contributors': key_contributors,
                'explanation': explanation,
                'original_prediction': game_prediction
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing game outcome: {e}")
            return {
                'error': str(e),
                'home_team': home_team,
                'away_team': away_team,
                'predicted_winner': game_prediction.get('predicted_winner', 'Unknown'),
                'original_prediction': game_prediction
            }

    def create_team_performance_metrics(self) -> None:
            """
            Create team performance metrics from the linescore data.
            
            This includes win-loss records for home and away games.
            """
            # Create a dataframe to track team wins and losses
            teams = set(self.linescore_data['HomeTeam'].unique()) | set(self.linescore_data['AwayTeam'].unique())
            self.team_records = pd.DataFrame(index=list(teams))
            
            # Initialize columns
            self.team_records['Wins'] = 0
            self.team_records['Losses'] = 0
            self.team_records['HomeWins'] = 0
            self.team_records['HomeLosses'] = 0
            self.team_records['AwayWins'] = 0
            self.team_records['AwayLosses'] = 0
            
            # Process the linescore data two rows at a time
            # Each game consists of away team (first row) and home team (second row)
            i = 0
            while i < len(self.linescore_data) - 1:
                try:
                    # Get the away team row and home team row
                    away_row = self.linescore_data.iloc[i]
                    home_row = self.linescore_data.iloc[i+1]
                    
                    # Verify that these rows are for the same game
                    if away_row['Date'] != home_row['Date'] or away_row['AwayTeam'] != home_row['AwayTeam'] or away_row['HomeTeam'] != home_row['HomeTeam']:
                        # These rows don't represent the same game, skip to next row
                        #self.logger.warning(f"Mismatched game data at index {i}, skipping")
                        i += 1
                        continue
                    
                    # Get team names
                    home_team = home_row['HomeTeam']
                    away_team = away_row['AwayTeam']
                    
                    # Get the scores - for away team from first row
                    away_runs = away_row.get('R', None)
                    # Get the scores - for home team from second row
                    home_runs = home_row.get('R', None)
                    
                    # If we can't find the runs columns, try to infer them
                    if home_runs is None or away_runs is None:
                        # For away row, look for numeric values
                        if isinstance(away_row, pd.Series):
                            away_numeric_cols = [col for col, val in away_row.items() if isinstance(val, (int, float)) and not pd.isna(val)]
                            if len(away_numeric_cols) >= 2:
                                away_runs = away_row[away_numeric_cols[-2]]  # Use R column (usually second-to-last numeric)
                        
                        # For home row, look for numeric values
                        if isinstance(home_row, pd.Series):
                            home_numeric_cols = [col for col, val in home_row.items() if isinstance(val, (int, float)) and not pd.isna(val)]
                            if len(home_numeric_cols) >= 2:
                                home_runs = home_row[home_numeric_cols[-2]]  # Use R column (usually second-to-last numeric)
                    
                    try:
                        home_runs = float(home_runs)
                        away_runs = float(away_runs)
                        
                        if home_runs > away_runs:
                            # Home team won
                            self.team_records.at[home_team, 'Wins'] += 1
                            self.team_records.at[home_team, 'HomeWins'] += 1
                            self.team_records.at[away_team, 'Losses'] += 1
                            self.team_records.at[away_team, 'AwayLosses'] += 1
                        else:
                            # Away team won
                            self.team_records.at[away_team, 'Wins'] += 1
                            self.team_records.at[away_team, 'AwayWins'] += 1
                            self.team_records.at[home_team, 'Losses'] += 1
                            self.team_records.at[home_team, 'HomeLosses'] += 1
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error processing game scores for {home_team} vs {away_team}: {e}")
                    
                    # Move to the next game (skip two rows)
                    i += 2
                except Exception as e:
                    self.logger.warning(f"Error processing game at index {i}: {e}")
                    i += 1  # Move to next row if there's an error
            
            # Calculate win percentages
            self.team_records['WinPct'] = self.team_records['Wins'] / (self.team_records['Wins'] + self.team_records['Losses'])
            self.team_records['HomeWinPct'] = self.team_records['HomeWins'] / (self.team_records['HomeWins'] + self.team_records['HomeLosses'])
            self.team_records['AwayWinPct'] = self.team_records['AwayWins'] / (self.team_records['AwayWins'] + self.team_records['AwayLosses'])
            
            # Fill NaN values with 0 (teams with no games yet)
            self.team_records = self.team_records.fillna(0)
            
            self.logger.info("Team performance metrics created")

    def enhance_features(self):
        """
        Implement advanced feature engineering specific to baseball statistics.
        
        This method adds derived features that capture complex baseball relationships,
        opponent-specific adjustments, time-series elements, and external factors.
        """
        try:
            self.logger.info("Enhancing feature engineering")
            
            # Add historical streaks and trends
            self.add_streak_features()
            
            # Add opponent adjustments
            self.add_opponent_adjustments()
            
            # Add ballpark factors
            self.add_ballpark_factors()
            
            # Add time-based features
            self.add_time_features()
            
            # Add advanced player metrics
            self.add_advanced_metrics()
            
            self.logger.info("Feature engineering enhancements completed")
        except Exception as e:
            self.logger.error(f"Error enhancing features: {e}")

    def add_streak_features(self):
        """
        Add streak and momentum-based features to player and team data.
        """
        # Add batting streaks
        if self.batting_data is not None:
            # Group by player
            for player in self.batting_data['Batting'].unique():
                player_data = self.batting_data[self.batting_data['Batting'] == player].sort_values('Date')
                
                # Calculate hitting streaks
                player_data['hit_streak'] = (player_data['H'] > 0).astype(int)
                player_data['hit_streak'] = player_data['hit_streak'].groupby((player_data['hit_streak'] == 0).cumsum()).cumsum()
                
                # Calculate rolling averages for recent form (5-game, 10-game)
                player_data['rolling_5_avg'] = player_data['H'].rolling(5, min_periods=1).sum() / player_data['AB'].rolling(5, min_periods=1).sum()
                player_data['rolling_10_avg'] = player_data['H'].rolling(10, min_periods=1).sum() / player_data['AB'].rolling(10, min_periods=1).sum()
                
                # Update the main dataframe
                self.batting_data.loc[self.batting_data['Batting'] == player, 'hit_streak'] = player_data['hit_streak']
                self.batting_data.loc[self.batting_data['Batting'] == player, 'rolling_5_avg'] = player_data['rolling_5_avg']
                self.batting_data.loc[self.batting_data['Batting'] == player, 'rolling_10_avg'] = player_data['rolling_10_avg']
        
        # Add pitching streaks and form
        if self.pitching_data is not None:
            # Group by pitcher
            for pitcher in self.pitching_data['Pitching'].unique():
                pitcher_data = self.pitching_data[self.pitching_data['Pitching'] == pitcher].sort_values('Date')
                
                # Calculate quality start streaks (>=6 IP, <=3 ER)
                pitcher_data['quality_start'] = ((pitcher_data['IP'] >= 6) & (pitcher_data['ER'] <= 3)).astype(int)
                pitcher_data['quality_start_streak'] = pitcher_data['quality_start'].groupby((pitcher_data['quality_start'] == 0).cumsum()).cumsum()
                
                # Calculate rolling ERA and WHIP
                pitcher_data['rolling_3_ERA'] = (9 * pitcher_data['ER'].rolling(3, min_periods=1).sum() / 
                                                pitcher_data['IP'].rolling(3, min_periods=1).sum())
                
                # Update the main dataframe
                self.pitching_data.loc[self.pitching_data['Pitching'] == pitcher, 'quality_start_streak'] = pitcher_data['quality_start_streak']
                self.pitching_data.loc[self.pitching_data['Pitching'] == pitcher, 'rolling_3_ERA'] = pitcher_data['rolling_3_ERA']

    def add_opponent_adjustments(self):
        """
        Add opponent-specific adjustments based on team strengths and weaknesses.
        """
        # Create team offensive and defensive ratings
        team_offense = {}
        team_defense = {}
        
        # Calculate team offensive metrics (runs scored per game)
        for team in self.team_records.index:
            team_games = self.linescore_data[(self.linescore_data['HomeTeam'] == team) | (self.linescore_data['AwayTeam'] == team)]
            runs_scored = 0
            game_count = 0
            
            i = 0
            while i < len(team_games) - 1:
                try:
                    away_row = team_games.iloc[i]
                    home_row = team_games.iloc[i+1]
                    
                    if away_row['Date'] != home_row['Date'] or away_row['AwayTeam'] != home_row['AwayTeam'] or away_row['HomeTeam'] != home_row['HomeTeam']:
                        i += 1
                        continue
                    
                    away_runs = float(away_row.get('R', 0))
                    home_runs = float(home_row.get('R', 0))
                    
                    if team == away_row['AwayTeam']:
                        runs_scored += away_runs
                    else:
                        runs_scored += home_runs
                    
                    game_count += 1
                    i += 2
                except:
                    i += 1
            
            rpg = runs_scored / max(game_count, 1)
            team_offense[team] = rpg
        
        # Calculate team defensive metrics (runs allowed per game)
        for team in self.team_records.index:
            team_games = self.linescore_data[(self.linescore_data['HomeTeam'] == team) | (self.linescore_data['AwayTeam'] == team)]
            runs_allowed = 0
            game_count = 0
            
            i = 0
            while i < len(team_games) - 1:
                try:
                    away_row = team_games.iloc[i]
                    home_row = team_games.iloc[i+1]
                    
                    if away_row['Date'] != home_row['Date'] or away_row['AwayTeam'] != home_row['AwayTeam'] or away_row['HomeTeam'] != home_row['HomeTeam']:
                        i += 1
                        continue
                    
                    away_runs = float(away_row.get('R', 0))
                    home_runs = float(home_row.get('R', 0))
                    
                    if team == away_row['AwayTeam']:
                        runs_allowed += home_runs
                    else:
                        runs_allowed += away_runs
                    
                    game_count += 1
                    i += 2
                except:
                    i += 1
            
            rapg = runs_allowed / max(game_count, 1)
            team_defense[team] = rapg
        
        # Calculate league averages
        league_avg_offense = sum(team_offense.values()) / len(team_offense)
        league_avg_defense = sum(team_defense.values()) / len(team_defense)
        
        # Normalize to get relative ratings (above/below 1.0)
        for team in team_offense:
            team_offense[team] = team_offense[team] / league_avg_offense
            team_defense[team] = league_avg_defense / team_defense[team]  # Invert so higher is better
        
        # Add opponent adjustments to batting data
        if self.batting_data is not None:
            self.batting_data['opponent_defense'] = 1.0
            for idx, row in self.batting_data.iterrows():
                opponent = None
                if row['Home']:
                    # Find away team as opponent
                    game_date = row['Date']
                    game_rows = self.linescore_data[self.linescore_data['Date'] == game_date]
                    for i in range(0, len(game_rows) - 1, 2):
                        if game_rows.iloc[i+1]['HomeTeam'] == row['Team']:
                            opponent = game_rows.iloc[i]['AwayTeam']
                            break
                else:
                    # Find home team as opponent
                    game_date = row['Date']
                    game_rows = self.linescore_data[self.linescore_data['Date'] == game_date]
                    for i in range(0, len(game_rows) - 1, 2):
                        if game_rows.iloc[i]['AwayTeam'] == row['Team']:
                            opponent = game_rows.iloc[i+1]['HomeTeam']
                            break
                
                if opponent and opponent in team_defense:
                    self.batting_data.at[idx, 'opponent_defense'] = team_defense[opponent]
        
        # Add opponent adjustments to pitching data
        if self.pitching_data is not None:
            self.pitching_data['opponent_offense'] = 1.0
            for idx, row in self.pitching_data.iterrows():
                opponent = None
                if row['Home']:
                    # Find away team as opponent
                    game_date = row['Date']
                    game_rows = self.linescore_data[self.linescore_data['Date'] == game_date]
                    for i in range(0, len(game_rows) - 1, 2):
                        if game_rows.iloc[i+1]['HomeTeam'] == row['Team']:
                            opponent = game_rows.iloc[i]['AwayTeam']
                            break
                else:
                    # Find home team as opponent
                    game_date = row['Date']
                    game_rows = self.linescore_data[self.linescore_data['Date'] == game_date]
                    for i in range(0, len(game_rows) - 1, 2):
                        if game_rows.iloc[i]['AwayTeam'] == row['Team']:
                            opponent = game_rows.iloc[i+1]['HomeTeam']
                            break
                
                if opponent and opponent in team_offense:
                    self.pitching_data.at[idx, 'opponent_offense'] = team_offense[opponent]

    def add_ballpark_factors(self):
        """
        Add ballpark factors to adjust for stadium effects on performance.
        """
        # Calculate runs scored in each ballpark
        ballpark_runs = {}
        ballpark_games = {}
        
        # Process all games to calculate runs per ballpark
        i = 0
        while i < len(self.linescore_data) - 1:
            try:
                away_row = self.linescore_data.iloc[i]
                home_row = self.linescore_data.iloc[i+1]
                
                if away_row['Date'] != home_row['Date'] or away_row['AwayTeam'] != home_row['AwayTeam'] or away_row['HomeTeam'] != home_row['HomeTeam']:
                    i += 1
                    continue
                
                ballpark = home_row['HomeTeam']  # Use home team as proxy for ballpark
                away_runs = float(away_row.get('R', 0))
                home_runs = float(home_row.get('R', 0))
                total_runs = away_runs + home_runs
                
                if ballpark not in ballpark_runs:
                    ballpark_runs[ballpark] = 0
                    ballpark_games[ballpark] = 0
                
                ballpark_runs[ballpark] += total_runs
                ballpark_games[ballpark] += 1
                
                i += 2
            except:
                i += 1
        
        # Calculate average runs per game for each ballpark
        ballpark_factors = {}
        total_runs = sum(ballpark_runs.values())
        total_games = sum(ballpark_games.values())
        league_avg = total_runs / total_games if total_games > 0 else 0
        
        for ballpark in ballpark_runs:
            if ballpark_games[ballpark] > 0:
                park_avg = ballpark_runs[ballpark] / ballpark_games[ballpark]
                ballpark_factors[ballpark] = park_avg / league_avg
            else:
                ballpark_factors[ballpark] = 1.0
        
        # Add ballpark factors to the data
        if self.batting_data is not None:
            self.batting_data['ballpark_factor'] = 1.0
            for idx, row in self.batting_data.iterrows():
                if row['Home']:
                    ballpark = row['Team']
                else:
                    # Find the home team for this game
                    game_date = row['Date']
                    away_team = row['Team']
                    game_rows = self.linescore_data[self.linescore_data['Date'] == game_date]
                    ballpark = None
                    for i in range(0, len(game_rows) - 1, 2):
                        if game_rows.iloc[i]['AwayTeam'] == away_team:
                            ballpark = game_rows.iloc[i+1]['HomeTeam']
                            break
                
                if ballpark and ballpark in ballpark_factors:
                    self.batting_data.at[idx, 'ballpark_factor'] = ballpark_factors[ballpark]
        
        if self.pitching_data is not None:
            self.pitching_data['ballpark_factor'] = 1.0
            for idx, row in self.pitching_data.iterrows():
                if row['Home']:
                    ballpark = row['Team']
                else:
                    # Find the home team for this game
                    game_date = row['Date']
                    away_team = row['Team']
                    game_rows = self.linescore_data[self.linescore_data['Date'] == game_date]
                    ballpark = None
                    for i in range(0, len(game_rows) - 1, 2):
                        if game_rows.iloc[i]['AwayTeam'] == away_team:
                            ballpark = game_rows.iloc[i+1]['HomeTeam']
                            break
                
                if ballpark and ballpark in ballpark_factors:
                    self.pitching_data.at[idx, 'ballpark_factor'] = ballpark_factors[ballpark]

    def add_time_features(self):
        """
        Add time-based features to capture seasonal trends and fatigue.
        """
        # Add day of week
        if self.batting_data is not None:
            self.batting_data['day_of_week'] = self.batting_data['Date'].dt.dayofweek
            
        if self.pitching_data is not None:
            self.pitching_data['day_of_week'] = self.pitching_data['Date'].dt.dayofweek
        
        # Add days since last appearance for pitchers
        if self.pitching_data is not None:
            for pitcher in self.pitching_data['Pitching'].unique():
                pitcher_data = self.pitching_data[self.pitching_data['Pitching'] == pitcher].sort_values('Date')
                pitcher_data['days_rest'] = (pitcher_data['Date'] - pitcher_data['Date'].shift(1)).dt.days
                self.pitching_data.loc[self.pitching_data['Pitching'] == pitcher, 'days_rest'] = pitcher_data['days_rest']
            
            # Fill missing values (for first appearances)
            self.pitching_data['days_rest'] = self.pitching_data['days_rest'].fillna(5)  # Assume 5 days rest for first game
        
        # Add game number in season
        if self.batting_data is not None:
            min_date = self.batting_data['Date'].min()
            self.batting_data['game_in_season'] = (self.batting_data['Date'] - min_date).dt.days // 1 + 1
        
        if self.pitching_data is not None:
            min_date = self.pitching_data['Date'].min()
            self.pitching_data['game_in_season'] = (self.pitching_data['Date'] - min_date).dt.days // 1 + 1

    def add_advanced_metrics(self):
        """
        Add advanced metrics from baseball analytics.
        """
        # Add advanced batting metrics
        if self.batting_data is not None:
            # Calculate ISO (Isolated Power)
            if all(col in self.batting_data.columns for col in ['H', 'Double', 'Triple', 'HR', 'AB']):
                self.batting_data['ISO'] = (self.batting_data['Double'] + 2*self.batting_data['Triple'] + 3*self.batting_data['HR']) / self.batting_data['AB']
                self.batting_data['ISO'] = self.batting_data['ISO'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Calculate BABIP (Batting Average on Balls in Play)
            if all(col in self.batting_data.columns for col in ['H', 'HR', 'AB', 'SO']):
                self.batting_data['BABIP'] = (self.batting_data['H'] - self.batting_data['HR']) / (self.batting_data['AB'] - self.batting_data['SO'] - self.batting_data['HR'])
                self.batting_data['BABIP'] = self.batting_data['BABIP'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Add advanced pitching metrics
        if self.pitching_data is not None:
            # Calculate FIP (Fielding Independent Pitching) - simplified version
            if all(col in self.pitching_data.columns for col in ['HR', 'BB', 'SO', 'IP']):
                self.pitching_data['FIP'] = ((13*self.pitching_data['HR']) + (3*self.pitching_data['BB']) - (2*self.pitching_data['SO'])) / self.pitching_data['IP']
                self.pitching_data['FIP'] = self.pitching_data['FIP'].replace([np.inf, -np.inf], np.nan).fillna(0)
                self.pitching_data['FIP'] = self.pitching_data['FIP'] + 3.2  # Add league constant (approx)
            
            # Calculate BABIP against
            if all(col in self.pitching_data.columns for col in ['H', 'HR', 'BF', 'BB', 'SO']):
                self.pitching_data['BABIP_against'] = (self.pitching_data['H'] - self.pitching_data['HR']) / (self.pitching_data['BF'] - self.pitching_data['BB'] - self.pitching_data['SO'] - self.pitching_data['HR'])
                self.pitching_data['BABIP_against'] = self.pitching_data['BABIP_against'].replace([np.inf, -np.inf], np.nan).fillna(0)

    def analyze_prediction_calibration(self, actual_results_file: str) -> Dict[str, Dict[str, float]]:
        """
        Analyze and calibrate model predictions by comparing against actual results.
        
        This method evaluates how well-calibrated the probability estimates are
        and applies calibration corrections to improve reliability of confidence intervals.
        
        Args:
            actual_results_file (str): Path to CSV file with actual game/player results
            
        Returns:
            Dict: Calibration metrics and correction factors by prediction type
        """
        try:
            import numpy as np
            import pandas as pd
            from sklearn.calibration import calibration_curve
            import matplotlib.pyplot as plt
            
            self.logger.info(f"Analyzing prediction calibration using {actual_results_file}")
            
            # Load actual results data
            actual_results = pd.read_csv(actual_results_file)
            
            # Initialize calibration results
            calibration_results = {
                'batting': {},
                'pitching': {},
                'game': {}
            }
            
            # Create directory for calibration plots
            os.makedirs('calibration_analysis', exist_ok=True)
            
            # Analyze batting predictions
            batting_metrics = ['H', 'R', 'HR', 'RBI', 'AVG']
            for metric in batting_metrics:
                pred_col = f'pred_{metric}'
                actual_col = f'actual_{metric}'
                
                if pred_col in actual_results.columns and actual_col in actual_results.columns:
                    # Get predictions and actual values
                    preds = actual_results[pred_col].values
                    actuals = actual_results[actual_col].values
                    
                    # Calculate prediction errors
                    errors = preds - actuals
                    mae = np.mean(np.abs(errors))
                    rmse = np.sqrt(np.mean(np.square(errors)))
                    
                    # Create bins for calibration analysis
                    bins = np.linspace(np.min(preds), np.max(preds), 10)
                    bin_indices = np.digitize(preds, bins)
                    
                    bin_actuals = [np.mean(actuals[bin_indices == i]) for i in range(1, len(bins))]
                    bin_preds = [(bins[i-1] + bins[i])/2 for i in range(1, len(bins))]
                    
                    # Calculate calibration metrics
                    calibration_error = np.mean(np.abs(np.array(bin_preds) - np.array(bin_actuals)))
                    
                    # Store results
                    calibration_results['batting'][metric] = {
                        'mae': mae,
                        'rmse': rmse,
                        'calibration_error': calibration_error,
                        'correction_factor': np.mean(actuals) / np.mean(preds) if np.mean(preds) > 0 else 1.0
                    }
                    
                    # Plot calibration curve
                    plt.figure(figsize=(8, 6))
                    plt.plot(bin_preds, bin_actuals, 'o-', label='Observed')
                    plt.plot([np.min(preds), np.max(preds)], [np.min(preds), np.max(preds)], '--', label='Perfect calibration')
                    plt.xlabel(f'Predicted {metric}')
                    plt.ylabel(f'Actual {metric}')
                    plt.title(f'Calibration Curve for {metric}')
                    plt.legend()
                    plt.savefig(f'calibration_analysis/batting_{metric}_calibration.png')
                    plt.close()
            
            # Similar analysis for pitching metrics
            pitching_metrics = ['IP', 'H', 'ER', 'SO', 'ERA', 'WHIP']
            for metric in pitching_metrics:
                # Similar code as for batting metrics
                pred_col = f'pred_{metric}'
                actual_col = f'actual_{metric}'
                
                if pred_col in actual_results.columns and actual_col in actual_results.columns:
                    # Implementation follows same pattern as batting metrics
                    # ...
                    pass
            
            # Analyze game win predictions
            if 'pred_home_win' in actual_results.columns and 'actual_home_win' in actual_results.columns:
                home_win_probs = actual_results['pred_home_win'].values
                home_win_actuals = actual_results['actual_home_win'].values
                
                # Calculate calibration curve
                prob_true, prob_pred = calibration_curve(home_win_actuals, home_win_probs, n_bins=10)
                
                # Calculate Brier score
                brier_score = np.mean((home_win_probs - home_win_actuals) ** 2)
                
                # Store results
                calibration_results['game']['win_probability'] = {
                    'brier_score': brier_score,
                    'calibration_curve': {
                        'prob_true': prob_true.tolist(),
                        'prob_pred': prob_pred.tolist()
                    }
                }
                
                # Plot calibration curve
                plt.figure(figsize=(8, 6))
                plt.plot(prob_pred, prob_true, 's-', label='Game win predictions')
                plt.plot([0, 1], [0, 1], '--', label='Perfectly calibrated')
                plt.xlabel('Predicted probability')
                plt.ylabel('True probability')
                plt.title('Win Probability Calibration Curve')
                plt.legend()
                plt.savefig('calibration_analysis/game_win_calibration.png')
                plt.close()
            
            # Save calibration results to file
            with open('calibration_analysis/calibration_metrics.json', 'w') as f:
                json.dump(calibration_results, f, indent=2)
            
            return calibration_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing prediction calibration: {e}")
            return {'error': str(e)}

    def implement_calibrated_predictions(self, player_name: str, is_home_game: bool, 
                                    opponent: str, target_date: str) -> Dict[str, any]:
        """
        Generate predictions with improved calibration based on historical error analysis.
        
        This method applies correction factors derived from the analyze_prediction_calibration
        method to adjust predictions and confidence intervals for better reliability.
        
        Args:
            player_name (str): Name of the player
            is_home_game (bool): Whether it's a home game
            opponent (str): Opponent team name
            target_date (str): Game date
            
        Returns:
            Dict: Calibrated predictions with adjusted confidence intervals
        """
        try:
            # Get uncalibrated predictions
            raw_predictions = self.predict_player_performance(
                player_name, is_home_game, opponent, target_date
            )
            
            # Check if we have calibration factors available
            calibration_file = 'calibration_analysis/calibration_metrics.json'
            if not os.path.exists(calibration_file):
                return raw_predictions
                
            with open(calibration_file, 'r') as f:
                calibration_data = json.load(f)
            
            # Determine if player is a batter or pitcher
            is_batter = False
            player_data = self.batting_data[self.batting_data['Batting'].str.contains(player_name, case=False, na=False)]
            if not player_data.empty:
                is_batter = True
                
            # Apply calibration factors
            calibrated_predictions = {
                'prediction': {},
                'confidence': {},
                'calibration_info': {}
            }
            
            if is_batter:
                category = 'batting'
            else:
                category = 'pitching'
                
            # Apply calibration to each metric
            for metric, value in raw_predictions['prediction'].items():
                if metric in calibration_data.get(category, {}):
                    # Get calibration factor
                    correction_factor = calibration_data[category][metric].get('correction_factor', 1.0)
                    
                    # Apply correction
                    calibrated_value = value * correction_factor
                    
                    # Get confidence interval
                    if metric in raw_predictions['confidence']:
                        lower, upper = raw_predictions['confidence'][metric]
                        
                        # Apply correction to confidence interval
                        calibrated_lower = lower * correction_factor
                        calibrated_upper = upper * correction_factor
                        
                        # Adjust interval width based on calibration error
                        calibration_error = calibration_data[category][metric].get('calibration_error', 0)
                        interval_width = calibrated_upper - calibrated_lower
                        
                        # Widen interval if calibration error is large
                        if calibration_error > 0:
                            widening_factor = 1.0 + min(1.0, calibration_error / value if value > 0 else 0)
                            half_width = interval_width * widening_factor / 2
                            calibrated_lower = max(0, calibrated_value - half_width)
                            calibrated_upper = calibrated_value + half_width
                        
                        calibrated_predictions['confidence'][metric] = (calibrated_lower, calibrated_upper)
                    
                    # Store calibrated prediction
                    calibrated_predictions['prediction'][metric] = calibrated_value
                    
                    # Store calibration info
                    calibrated_predictions['calibration_info'][metric] = {
                        'correction_factor': correction_factor,
                        'original_value': value
                    }
                else:
                    # Use uncalibrated values if no calibration data available
                    calibrated_predictions['prediction'][metric] = value
                    if metric in raw_predictions['confidence']:
                        calibrated_predictions['confidence'][metric] = raw_predictions['confidence'][metric]
            
            return calibrated_predictions
            
        except Exception as e:
            self.logger.error(f"Error implementing calibrated predictions for {player_name}: {e}")
            return raw_predictions  # Return uncalibrated predictions if error occurs

    def add_weather_data(self, weather_file: str) -> None:
        """
        Incorporates weather data to enhance game predictions.
        
        Weather conditions can significantly impact game outcomes and player performance.
        This method loads weather data and creates features to capture these effects.
        
        Args:
            weather_file (str): Path to CSV file with weather data
        """
        try:
            import pandas as pd
            
            self.logger.info(f"Loading weather data from {weather_file}")
            
            # Load weather data
            weather_df = pd.read_csv(weather_file)
            
            # Ensure date column is in datetime format
            if 'date' in weather_df.columns:
                weather_df['date'] = pd.to_datetime(weather_df['date'])
            
            # Store weather data
            self.weather_data = weather_df
            
            # Create mapping of weather conditions to impact factors
            self.weather_effects = {
                'temperature': {
                    'low': {'hitting': 0.9, 'power': 0.85, 'pitching_control': 1.05, 'description': 'Cold temperatures reducing ball flight'},
                    'moderate': {'hitting': 1.0, 'power': 1.0, 'pitching_control': 1.0, 'description': 'Moderate temperatures with neutral effects'},
                    'high': {'hitting': 1.05, 'power': 1.1, 'pitching_control': 0.95, 'description': 'Hot temperatures increasing ball flight'}
                },
                'wind': {
                    'in': {'hitting': 0.9, 'power': 0.85, 'pitching_control': 1.05, 'description': 'Wind blowing in reducing ball flight'},
                    'out': {'hitting': 1.1, 'power': 1.15, 'pitching_control': 0.95, 'description': 'Wind blowing out increasing ball flight'},
                    'left': {'hitting_lefty': 1.05, 'hitting_righty': 0.95, 'description': 'Wind blowing left-to-right'},
                    'right': {'hitting_lefty': 0.95, 'hitting_righty': 1.05, 'description': 'Wind blowing right-to-left'},
                    'none': {'hitting': 1.0, 'power': 1.0, 'pitching_control': 1.0, 'description': 'No significant wind effect'}
                },
                'precipitation': {
                    'none': {'hitting': 1.0, 'power': 1.0, 'pitching_control': 1.0, 'description': 'No precipitation'},
                    'light': {'hitting': 0.95, 'power': 0.97, 'pitching_control': 0.95, 'description': 'Light rain affecting grip'},
                    'moderate': {'hitting': 0.9, 'power': 0.95, 'pitching_control': 0.9, 'description': 'Moderate rain significantly affecting conditions'},
                    'heavy': {'hitting': 0.8, 'power': 0.9, 'pitching_control': 0.8, 'description': 'Heavy rain severely limiting performance'}
                },
                'humidity': {
                    'low': {'hitting': 0.98, 'power': 0.95, 'description': 'Low humidity reducing ball flight'},
                    'moderate': {'hitting': 1.0, 'power': 1.0, 'description': 'Moderate humidity with neutral effects'},
                    'high': {'hitting': 1.02, 'power': 1.05, 'description': 'High humidity increasing ball flight'}
                }
            }
            
            self.logger.info(f"Weather data loaded with {len(weather_df)} records")
            
        except Exception as e:
            self.logger.error(f"Error loading weather data: {e}")

    def predict_game_with_weather(self, home_team: str, away_team: str, 
                                target_date: str) -> Dict[str, float]:
        """
        Enhanced game prediction that incorporates weather conditions.
        
        This method extends the base game prediction by adjusting win probabilities
        based on weather conditions and their expected impact on team performance.
        
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            target_date (str): The date of the game to predict
            
        Returns:
            Dict: Enhanced win probability predictions with weather effects
        """
        try:
            # Get base prediction without weather
            base_prediction = self.predict_game_winner(home_team, away_team, target_date)
            
            # Check if we have weather data
            if not hasattr(self, 'weather_data'):
                return base_prediction
                
            # Find weather for this game
            target_date_dt = pd.to_datetime(target_date)
            game_weather = self.weather_data[
                (self.weather_data['date'] == target_date_dt) & 
                (self.weather_data['home_team'] == home_team)
            ]
            
            if game_weather.empty:
                return base_prediction
                
            # Extract weather conditions
            weather_row = game_weather.iloc[0]
            
            temperature = weather_row.get('temperature', 70)
            wind_direction = weather_row.get('wind_direction', 'none')
            wind_speed = weather_row.get('wind_speed', 0)
            precipitation = weather_row.get('precipitation', 'none')
            humidity = weather_row.get('humidity', 50)
            
            # Categorize weather conditions
            if temperature < 50:
                temp_category = 'low'
            elif temperature > 85:
                temp_category = 'high'
            else:
                temp_category = 'moderate'
                
            if precipitation == 'none':
                precip_category = 'none'
            elif precipitation == 'light':
                precip_category = 'light'
            elif precipitation == 'moderate':
                precip_category = 'moderate'
            else:
                precip_category = 'heavy'
                
            if humidity < 30:
                humidity_category = 'low'
            elif humidity > 70:
                humidity_category = 'high'
            else:
                humidity_category = 'moderate'
                
            # Wind effect depends on ballpark orientation and wind direction/speed
            if wind_speed < 5:
                wind_category = 'none'
            else:
                wind_category = wind_direction
                
            # Get team stats for analysis
            home_stats = self.get_team_stats(home_team, target_date)
            away_stats = self.get_team_stats(away_team, target_date)
            
            # Analyze team composition to determine sensitivity to weather
            home_hr_rate = home_stats.get('home_hr_per_game', 0)
            away_hr_rate = away_stats.get('away_hr_per_game', 0)
            
            home_power_team = home_hr_rate > 1.2  # Arbitrary threshold
            away_power_team = away_hr_rate > 1.0  # Lower threshold for away teams
            
            # Calculate weather adjustments
            home_adjustment = 1.0
            away_adjustment = 1.0
            weather_description = []
            
            # Temperature effects
            if temp_category in self.weather_effects['temperature']:
                temp_effect = self.weather_effects['temperature'][temp_category]
                if home_power_team:
                    home_adjustment *= temp_effect['power']
                else:
                    home_adjustment *= temp_effect['hitting']
                    
                if away_power_team:
                    away_adjustment *= temp_effect['power']
                else:
                    away_adjustment *= temp_effect['hitting']
                    
                weather_description.append(temp_effect['description'])
                
            # Wind effects - more significant factor
            if wind_category in self.weather_effects['wind']:
                wind_effect = self.weather_effects['wind'][wind_category]
                
                if wind_category in ['in', 'out']:
                    if home_power_team:
                        home_adjustment *= wind_effect['power'] * 1.1  # Extra factor for power teams
                    else:
                        home_adjustment *= wind_effect['hitting']
                        
                    if away_power_team:
                        away_adjustment *= wind_effect['power'] * 1.1
                    else:
                        away_adjustment *= wind_effect['hitting']
                
                weather_description.append(f"{wind_effect['description']} at {wind_speed} mph")
                
            # Precipitation effects
            if precip_category in self.weather_effects['precipitation']:
                precip_effect = self.weather_effects['precipitation'][precip_category]
                home_adjustment *= precip_effect['hitting']
                away_adjustment *= precip_effect['hitting']
                
                if precip_category != 'none':
                    weather_description.append(precip_effect['description'])
                
            # Humidity effects
            if humidity_category in self.weather_effects['humidity']:
                humidity_effect = self.weather_effects['humidity'][humidity_category]
                home_adjustment *= humidity_effect['power'] if home_power_team else humidity_effect['hitting']
                away_adjustment *= humidity_effect['power'] if away_power_team else humidity_effect['hitting']
                
                if humidity_category != 'moderate':
                    weather_description.append(humidity_effect['description'])
            
            # Apply adjustments to win probabilities
            base_home_prob = base_prediction['home_win_probability']
            base_away_prob = base_prediction['away_win_probability']
            
            # Calculate relative advantage
            relative_advantage = home_adjustment / away_adjustment
            
            # Convert to odds
            home_odds = base_home_prob / (1 - base_home_prob)
            
            # Apply relative advantage to odds
            adjusted_home_odds = home_odds * relative_advantage
            
            # Convert back to probability
            adjusted_home_prob = adjusted_home_odds / (1 + adjusted_home_odds)
            adjusted_away_prob = 1 - adjusted_home_prob
            
            # Limit extreme adjustments
            max_adjustment = 0.15  # Maximum 15% shift due to weather
            if abs(adjusted_home_prob - base_home_prob) > max_adjustment:
                if adjusted_home_prob > base_home_prob:
                    adjusted_home_prob = base_home_prob + max_adjustment
                else:
                    adjusted_home_prob = base_home_prob - max_adjustment
                    
                adjusted_away_prob = 1 - adjusted_home_prob
            
            # Create enhanced prediction
            weather_prediction = {
                'home_win_probability': adjusted_home_prob,
                'away_win_probability': adjusted_away_prob,
                'confidence': base_prediction['confidence'],
                'base_prediction': {
                    'home_win_probability': base_home_prob,
                    'away_win_probability': base_away_prob
                },
                'weather_effects': {
                    'home_adjustment': home_adjustment,
                    'away_adjustment': away_adjustment,
                    'temperature': temperature,
                    'wind_direction': wind_direction,
                    'wind_speed': wind_speed,
                    'precipitation': precipitation,
                    'humidity': humidity,
                    'description': '; '.join(weather_description)
                }
            }
            
            return weather_prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting game with weather for {home_team} vs {away_team}: {e}")
            # Fall back to base prediction if there's an error
            return self.predict_game_winner(home_team, away_team, target_date)

    def analyze_prediction_errors(self, actual_results_file: str) -> Dict[str, Dict[str, float]]:
        """
        Analyze prediction errors to identify patterns and areas for improvement.
        
        This method compares predictions against actual results to understand where
        the model performs well and where it struggles, broken down by various factors.
        
        Args:
            actual_results_file (str): Path to CSV file with actual game/player results
            
        Returns:
            Dict: Error metrics by various dimensions and categories
        """
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            self.logger.info(f"Analyzing prediction errors using {actual_results_file}")
            
            # Load actual results
            results_df = pd.read_csv(actual_results_file)
            
            # Create directory for error analysis
            os.makedirs('error_analysis', exist_ok=True)
            
            # Initialize error metrics dictionary
            error_metrics = {
                'overall': {},
                'by_position': {},
                'by_team': {},
                'by_month': {},
                'by_ballpark': {},
                'home_vs_away': {},
                'by_prediction_confidence': {}
            }
            
            # Extract prediction metrics
            batting_metrics = ['H', 'R', 'HR', 'RBI', 'AVG']
            pitching_metrics = ['IP', 'H', 'ER', 'SO', 'ERA']
            
            # Overall error metrics
            overall_errors = {}
            
            # Process each metric
            all_metrics = batting_metrics + pitching_metrics
            for metric in all_metrics:
                pred_col = f'pred_{metric}'
                actual_col = f'actual_{metric}'
                
                if pred_col in results_df.columns and actual_col in results_df.columns:
                    # Calculate errors
                    results_df[f'error_{metric}'] = results_df[actual_col] - results_df[pred_col]
                    results_df[f'abs_error_{metric}'] = np.abs(results_df[f'error_{metric}'])
                    results_df[f'pct_error_{metric}'] = np.abs(results_df[f'error_{metric}'] / results_df[actual_col].replace(0, np.nan))
                    
                    # Calculate overall error metrics
                    mae = results_df[f'abs_error_{metric}'].mean()
                    rmse = np.sqrt((results_df[f'error_{metric}'] ** 2).mean())
                    mape = results_df[f'pct_error_{metric}'].mean() * 100  # Convert to percentage
                    
                    overall_errors[metric] = {
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape
                    }
                    
                    # Plot error distribution
                    plt.figure(figsize=(10, 6))
                    sns.histplot(results_df[f'error_{metric}'].dropna(), kde=True)
                    plt.axvline(x=0, color='r', linestyle='--')
                    plt.title(f'Error Distribution for {metric}')
                    plt.xlabel('Prediction Error')
                    plt.ylabel('Frequency')
                    plt.savefig(f'error_analysis/error_dist_{metric}.png')
                    plt.close()
            
            # Store overall metrics
            error_metrics['overall'] = overall_errors
            
            # Analyze errors by position
            positions = results_df['position'].dropna().unique()
            position_errors = {}
            
            for position in positions:
                position_df = results_df[results_df['position'] == position]
                position_metrics = {}
                
                for metric in all_metrics:
                    error_col = f'abs_error_{metric}'
                    if error_col in position_df.columns:
                        position_metrics[metric] = {
                            'mae': position_df[error_col].mean(),
                            'count': len(position_df[error_col].dropna())
                        }
                
                position_errors[position] = position_metrics
            
            error_metrics['by_position'] = position_errors
            
            # Analyze errors by team
            teams = results_df['team'].dropna().unique()
            team_errors = {}
            
            for team in teams:
                team_df = results_df[results_df['team'] == team]
                team_metrics = {}
                
                for metric in all_metrics:
                    error_col = f'abs_error_{metric}'
                    if error_col in team_df.columns:
                        team_metrics[metric] = {
                            'mae': team_df[error_col].mean(),
                            'count': len(team_df[error_col].dropna())
                        }
                
                team_errors[team] = team_metrics
            
            error_metrics['by_team'] = team_errors
            
            # Analyze home vs away
            if 'is_home' in results_df.columns:
                home_df = results_df[results_df['is_home'] == True]
                away_df = results_df[results_df['is_home'] == False]
                
                home_away_errors = {
                    'home': {},
                    'away': {}
                }
                
                for metric in all_metrics:
                    error_col = f'abs_error_{metric}'
                    if error_col in results_df.columns:
                        if not home_df.empty:
                            home_away_errors['home'][metric] = {
                                'mae': home_df[error_col].mean(),
                                'count': len(home_df[error_col].dropna())
                            }
                        
                        if not away_df.empty:
                            home_away_errors['away'][metric] = {
                                'mae': away_df[error_col].mean(),
                                'count': len(away_df[error_col].dropna())
                            }
                
                error_metrics['home_vs_away'] = home_away_errors
            
            # Analyze by month if date column is available
            if 'date' in results_df.columns:
                results_df['date'] = pd.to_datetime(results_df['date'])
                results_df['month'] = results_df['date'].dt.month
                
                month_errors = {}
                for month in results_df['month'].dropna().unique():
                    month_df = results_df[results_df['month'] == month]
                    month_metrics = {}
                    
                    for metric in all_metrics:
                        error_col = f'abs_error_{metric}'
                        if error_col in month_df.columns:
                            month_metrics[metric] = {
                                'mae': month_df[error_col].mean(),
                                'count': len(month_df[error_col].dropna())
                            }
                    
                    month_errors[int(month)] = month_metrics
                
                error_metrics['by_month'] = month_errors
            
            # Generate summary visualizations
            # 1. Error by position
            positions_to_plot = []
            metrics_to_plot = []
            mae_values = []
            
            for position, metrics in position_errors.items():
                for metric, values in metrics.items():
                    if values['count'] > 10:  # Only include with sufficient samples
                        positions_to_plot.append(position)
                        metrics_to_plot.append(metric)
                        mae_values.append(values['mae'])
            
            if positions_to_plot:
                position_error_df = pd.DataFrame({
                    'Position': positions_to_plot,
                    'Metric': metrics_to_plot,
                    'MAE': mae_values
                })
                
                plt.figure(figsize=(12, 8))
                sns.barplot(data=position_error_df, x='Position', y='MAE', hue='Metric')
                plt.title('Prediction Error by Position')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('error_analysis/error_by_position.png')
                plt.close()
            
            # 2. Error by team
            teams_to_plot = []
            metrics_to_plot = []
            mae_values = []
            
            for team, metrics in team_errors.items():
                for metric, values in metrics.items():
                    if values['count'] > 10:
                        teams_to_plot.append(team)
                        metrics_to_plot.append(metric)
                        mae_values.append(values['mae'])
            
            if teams_to_plot:
                team_error_df = pd.DataFrame({
                    'Team': teams_to_plot,
                    'Metric': metrics_to_plot,
                    'MAE': mae_values
                })
                
                plt.figure(figsize=(14, 8))
                sns.barplot(data=team_error_df, x='Team', y='MAE', hue='Metric')
                plt.title('Prediction Error by Team')
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig('error_analysis/error_by_team.png')
                plt.close()
            
            # Save error metrics to file
            with open('error_analysis/error_metrics.json', 'w') as f:
                json.dump(error_metrics, f, indent=2)
            
            self.logger.info("Error analysis complete. Results saved to error_analysis directory.")
            return error_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing prediction errors: {e}")
            return {'error': str(e)}

    def use_advanced_models_for_prediction(self, model_type='batting'):
        """
        Update the prediction methods to use advanced models instead of base models.
        This should be called after initializing advanced models.
        
        Args:
            model_type (str): Either 'batting', 'pitching', or 'team'
        """
        try:
            if model_type == 'batting' and hasattr(self, 'batting_ensemble'):
                # Store original models as backup
                self.original_batting_models = self.batting_models.copy()
                
                # Replace with advanced models
                for metric in self.batting_models.keys():
                    self.batting_models[metric] = self.batting_ensemble
                    
                self.logger.info("Advanced batting models activated for predictions")
                
            elif model_type == 'pitching' and hasattr(self, 'pitching_ensemble'):
                # Store original models as backup
                self.original_pitching_models = self.pitching_models.copy()
                
                # Replace with advanced models
                for metric in self.pitching_models.keys():
                    self.pitching_models[metric] = self.pitching_ensemble
                    
                self.logger.info("Advanced pitching models activated for predictions")
                
            elif model_type == 'team' and hasattr(self, 'team_ensemble'):
                # Store original model as backup
                self.original_team_models = self.team_models.copy()
                
                # Replace with advanced model
                if 'win_prediction' in self.team_models:
                    self.team_models['win_prediction'] = self.team_ensemble
                    
                self.logger.info("Advanced team models activated for predictions")
            
        except Exception as e:
            self.logger.error(f"Error activating advanced models for {model_type}: {e}")


def main():
    """
    Main function to demonstrate usage of the BaseballPredictor class with enhanced features.
    
    This function parses command-line arguments, initializes the predictor, loads data,
    trains models, and makes predictions with advanced options including weather effects,
    calibrated predictions, and comprehensive error analysis.
    """
    import argparse
    import os
    import pandas as pd
    import traceback
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Predict baseball statistics with enhanced features.')
    
    # Basic configuration arguments
    parser.add_argument('--data-path', type=str, default='.', help='Path to the data files.')
    parser.add_argument('--target-date', type=str, required=True, help='Target date for predictions (YYYY-MM-DD).')
    parser.add_argument('--lineup-file', type=str, default='2024_lineups.csv', 
                        help='CSV file containing team lineups.')
    
    # Model training and loading arguments
    parser.add_argument('--train', action='store_true', help='Train new models.')
    parser.add_argument('--save-models', action='store_true', help='Save trained models.')
    parser.add_argument('--load-models', action='store_true', help='Load existing models.')
    
    # Output options
    parser.add_argument('--output-file', type=str, default='', 
                        help='Output CSV file for predictions. Default: predictions_YYYY-MM-DD.csv')
    
    # New enhanced features
    parser.add_argument('--weather-file', type=str, default='', 
                        help='CSV file with weather data for enhanced predictions.')
    parser.add_argument('--calibration-file', type=str, default='', 
                        help='CSV file with actual results for calibration analysis.')
    parser.add_argument('--analyze-errors', type=str, default='', 
                        help='CSV file with actual results for error analysis.')
    parser.add_argument('--enhanced-features', action='store_true', 
                        help='Enable advanced feature engineering.')
    parser.add_argument('--calibrated-predictions', action='store_true', 
                        help='Use calibrated predictions with adjusted confidence intervals.')
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output_file:
        args.output_file = f"predictions_{args.target_date}.csv"
    
    # Initialize the predictor
    predictor = BaseballPredictor(data_path=args.data_path)
    
    # Load data
    predictor.load_data(args.target_date)

    # Enable enhanced features if requested
    if args.enhanced_features:
        predictor.enhance_features()
        print("Enhanced feature engineering enabled")
    
    # Train or load models
    if args.train:
        predictor.train_models()
        if args.save_models:
            predictor.save_models()
    elif args.load_models:
        predictor.load_models()
    else:
        predictor.train_models()  # Default to training if not specified
    

    
    # Load weather data if provided
    if args.weather_file and os.path.exists(args.weather_file):
        predictor.add_weather_data(args.weather_file)
        print(f"Weather data loaded from {args.weather_file}")
    
    # Perform calibration analysis if requested
    if args.calibration_file and os.path.exists(args.calibration_file):
        print(f"Performing prediction calibration analysis using {args.calibration_file}")
        calibration_results = predictor.analyze_prediction_calibration(args.calibration_file)
        print(f"Calibration analysis completed and saved to calibration_analysis directory")
    
    # Perform error analysis if requested
    if args.analyze_errors and os.path.exists(args.analyze_errors):
        print(f"Performing prediction error analysis using {args.analyze_errors}")
        error_metrics = predictor.analyze_prediction_errors(args.analyze_errors)
        print(f"Error analysis completed and saved to error_analysis directory")
    
    # Load lineup data
    lineup_file = os.path.join(args.data_path, args.lineup_file)
    if not os.path.exists(lineup_file):
        print(f"Lineup file not found: {lineup_file}")
        return
    
    try:
        # Load the lineup data
        lineups_df = pd.read_csv(lineup_file)

        # Convert Date to string format that matches target_date
        lineups_df['Date'] = lineups_df['Date'].astype(str).str[:-1]
        target_date_str = args.target_date.replace('-', '')

        # Filter lineups for the target date
        date_lineups = lineups_df[lineups_df['Date'] == target_date_str]

        if date_lineups.empty:
            print(f"No lineups found for date: {args.target_date}")
            return

        # Create DataFrame to store all predictions
        predictions_data = []
        summary_concat = pd.DataFrame()

        # Group lineups by game
        # Each game has 20 rows (10 for away team, 10 for home team)
        game_count = len(date_lineups) // 20
        print(f"Found {game_count} games for {args.target_date}")

        for game_idx in range(game_count):
            start_idx = game_idx * 20
            end_idx = start_idx + 20
            game_lineups = date_lineups.iloc[start_idx:end_idx]
            
            # Extract away and home lineups
            away_lineup = game_lineups[game_lineups['Home'] == False]
            home_lineup = game_lineups[game_lineups['Home'] == True]
            
            # Skip if we don't have both home and away lineups
            if away_lineup.empty or home_lineup.empty:
                print(f"Skipping incomplete game at index {game_idx}")
                continue
            
            # Get unique teams for this game
            away_team = away_lineup['team'].iloc[0]
            home_team = home_lineup['team'].iloc[0]
            
            print(f"\n\n---- GAME {game_idx + 1}: {away_team} at {home_team} ({args.target_date}) ----")
            
            # Predict game winner with weather effects if available
            if hasattr(predictor, 'weather_data'):
                game_prediction = predictor.predict_game_with_weather(
                    home_team,
                    away_team,
                    args.target_date
                )
                
                # Display weather effects if available
                if 'weather_effects' in game_prediction:
                    weather = game_prediction['weather_effects']
                    print(f"Weather Conditions: {weather.get('description', 'No data')}")
                    print(f"Temperature: {weather.get('temperature')}°F, Wind: {weather.get('wind_speed')} mph {weather.get('wind_direction')}")
                    print(f"Home Team Weather Adjustment: {weather.get('home_adjustment', 1.0):.2f}x")
                    print(f"Away Team Weather Adjustment: {weather.get('away_adjustment', 1.0):.2f}x")
            else:
                game_prediction = predictor.predict_game_winner(
                    home_team,
                    away_team,
                    args.target_date
                )
            
            print(f"Home Win Probability: {game_prediction['home_win_probability']:.2%}")
            print(f"Away Win Probability: {game_prediction['away_win_probability']:.2%}")
            print(f"Confidence: {game_prediction['confidence']}")

            # Add this after line 3186 in main()
            print(f"Predicted Winner: {game_prediction['predicted_winner']} ({game_prediction['win_probability']:.2%})")
            print(f"Expected Margin: {abs(game_prediction['win_margin']):.1f} runs")
            if 'explanation' in game_prediction:
                print(f"Analysis: {game_prediction['explanation']}")

            predictions_home = []
            predictions_away = []
            
            # Process away team lineup
            print(f"\n{away_team} Lineup Predictions:")
            away_players = away_lineup[away_lineup['player_name'].notna()]
            for _, player in away_players.iterrows():
                if pd.isna(player['player_name']) or player['player_name'] == '':
                    continue
                    
                # Predict player performance with calibration if requested
                if args.calibrated_predictions:
                    player_prediction = predictor.implement_calibrated_predictions(
                        player['player_name'],
                        False,  # Away team
                        home_team,  # Opponent
                        args.target_date
                    )
                    # Add calibration info to console output
                    if 'calibration_info' in player_prediction:
                        calibration_info = player_prediction['calibration_info']
                        # Display calibration factors in prediction output
                else:
                    player_prediction = predictor.predict_player_performance(
                        player['player_name'],
                        False,  # Away team
                        home_team,  # Opponent
                        args.target_date
                    )
                
                # Get player stats
                player_stats = predictor.get_player_stats(player['player_name'], args.target_date)
                
                # Print player info and add to predictions data
                print(f"\n{player['order']}. {player['player_name']} ({player['position']})")
                
                # Create player row for predictions DataFrame
                player_row = {
                    'date': args.target_date,
                    'player_name': player['player_name'],
                    'player_id': player.get('player_id', ''),
                    'team': away_team,
                    'opponent': home_team,
                    'position': player['position'],
                    'batting_order': player['order'],
                    'is_home': False,
                    'game_id': f"{away_team}@{home_team}_{args.target_date}"
                }
                
                # Add historical stats
                if player_stats['overall']:
                    for metric, value in player_stats['overall'].items():
                        if isinstance(value, float) and metric in ['AVG', 'HR', 'RBI', 'ERA', 'WHIP', 'games', 'H', 'R', 'IP', 'SO']:
                            player_row[f'hist_{metric}'] = value
                
                # Add predictions and confidence intervals
                if player_prediction['prediction']:
                    for metric, value in player_prediction['prediction'].items():
                        if isinstance(value, float):
                            player_row[f'pred_{metric}'] = value
                            if metric in player_prediction['confidence']:
                                lower, upper = player_prediction['confidence'][metric]
                                player_row[f'pred_{metric}_lower'] = lower
                                player_row[f'pred_{metric}_upper'] = upper
                
                # Add calibration info if available
                if args.calibrated_predictions and 'calibration_info' in player_prediction:
                    for metric, info in player_prediction['calibration_info'].items():
                        if 'correction_factor' in info:
                            player_row[f'calib_factor_{metric}'] = info['correction_factor']
                
                # Add game prediction
                player_row['team_win_prob'] = game_prediction['away_win_probability']
                
                # Add weather effects if available
                if hasattr(predictor, 'weather_data') and 'weather_effects' in game_prediction:
                    player_row['weather_adjustment'] = game_prediction['weather_effects'].get('away_adjustment', 1.0)
                
                # Add to predictions data
                predictions_away.append(player_row)
                predictions_data.append(player_row)
                
                # Print previous stats if available
                if player_stats['overall']:
                    print("  Recent Stats:")
                    for metric, value in player_stats['overall'].items():
                        if isinstance(value, float) and metric in ['AVG', 'HR', 'RBI', 'ERA', 'WHIP']:
                            print(f"    {metric}: {value:.3f}")
                
                # Print predictions if available
                if player_prediction['prediction']:
                    print("  Predicted Performance:")
                    for metric, value in player_prediction['prediction'].items():
                        if isinstance(value, float) and metric in ['H', 'R', 'RBI', 'HR', 'AVG', 'IP', 'ERA', 'WHIP', 'SO']:
                            lower, upper = player_prediction['confidence'][metric]
                            print(f"    {metric}: {value:.3f} (95% CI: {lower:.3f} - {upper:.3f})")
                            
                            # Print calibration info if available
                            if args.calibrated_predictions and 'calibration_info' in player_prediction and metric in player_prediction['calibration_info']:
                                correction = player_prediction['calibration_info'][metric].get('correction_factor', 1.0)
                                print(f"      [Calibration factor: {correction:.2f}x]")
            
            # Process home team lineup (similar to away team)
            print(f"\n{home_team} Lineup Predictions:")
            home_players = home_lineup[home_lineup['player_name'].notna()]
            for _, player in home_players.iterrows():
                if pd.isna(player['player_name']) or player['player_name'] == '':
                    continue
                    
                # Predict player performance with calibration if requested
                if args.calibrated_predictions:
                    player_prediction = predictor.implement_calibrated_predictions(
                        player['player_name'],
                        True,  # Home team
                        away_team,  # Opponent
                        args.target_date
                    )
                else:
                    player_prediction = predictor.predict_player_performance(
                        player['player_name'],
                        True,  # Home team
                        away_team,  # Opponent
                        args.target_date
                    )
                
                # Get player stats
                player_stats = predictor.get_player_stats(player['player_name'], args.target_date)
                
                # Print player info and add to predictions data
                print(f"\n{player['order']}. {player['player_name']} ({player['position']})")
                
                # Create player row for predictions DataFrame
                player_row = {
                    'date': args.target_date,
                    'player_name': player['player_name'],
                    'player_id': player.get('player_id', ''),
                    'team': home_team,
                    'opponent': away_team,
                    'position': player['position'],
                    'batting_order': player['order'],
                    'is_home': True,
                    'game_id': f"{away_team}@{home_team}_{args.target_date}"
                }
                
                # Add historical stats, predictions and game prediction to row
                # [same code as for away players]
                # Add historical stats
                if player_stats['overall']:
                    for metric, value in player_stats['overall'].items():
                        if isinstance(value, float) and metric in ['AVG', 'HR', 'RBI', 'ERA', 'WHIP', 'games', 'H', 'R', 'IP', 'SO']:
                            player_row[f'hist_{metric}'] = value
                
                # Add predictions and confidence intervals
                if player_prediction['prediction']:
                    for metric, value in player_prediction['prediction'].items():
                        if isinstance(value, float):
                            player_row[f'pred_{metric}'] = value
                            if metric in player_prediction['confidence']:
                                lower, upper = player_prediction['confidence'][metric]
                                player_row[f'pred_{metric}_lower'] = lower
                                player_row[f'pred_{metric}_upper'] = upper
                
                # Add calibration info if available
                if args.calibrated_predictions and 'calibration_info' in player_prediction:
                    for metric, info in player_prediction['calibration_info'].items():
                        if 'correction_factor' in info:
                            player_row[f'calib_factor_{metric}'] = info['correction_factor']
                
                # Add game prediction
                player_row['team_win_prob'] = game_prediction['home_win_probability']
                
                # Add weather effects if available
                if hasattr(predictor, 'weather_data') and 'weather_effects' in game_prediction:
                    player_row['weather_adjustment'] = game_prediction['weather_effects'].get('home_adjustment', 1.0)
                
                # Add to predictions data
                predictions_home.append(player_row)
                predictions_data.append(player_row)
                
                # Print previous stats if available
                if player_stats['overall']:
                    print("  Recent Stats:")
                    for metric, value in player_stats['overall'].items():
                        if isinstance(value, float) and metric in ['AVG', 'HR', 'RBI', 'ERA', 'WHIP']:
                            print(f"    {metric}: {value:.3f}")
                
                # Print predictions if available
                if player_prediction['prediction']:
                    print("  Predicted Performance:")
                    for metric, value in player_prediction['prediction'].items():
                        if isinstance(value, float) and metric in ['H', 'R', 'RBI', 'HR', 'AVG', 'IP', 'ERA', 'WHIP', 'SO']:
                            lower, upper = player_prediction['confidence'][metric]
                            print(f"    {metric}: {value:.3f} (95% CI: {lower:.3f} - {upper:.3f})")
                            
                            # Print calibration info if available
                            if args.calibrated_predictions and 'calibration_info' in player_prediction and metric in player_prediction['calibration_info']:
                                correction = player_prediction['calibration_info'][metric].get('correction_factor', 1.0)
                                print(f"      [Calibration factor: {correction:.2f}x]")

            # Run detailed game analysis
            detailed_analysis = predictor.analyze_game_outcome(
                game_prediction,
                predictions_home,
                predictions_away,
                home_team,
                away_team
            )

            # Add weather effects to the summary if available
            if hasattr(predictor, 'weather_data') and 'weather_effects' in game_prediction:
                weather_effects = game_prediction['weather_effects']
                detailed_analysis['weather_temp'] = weather_effects.get('temperature')
                detailed_analysis['weather_wind'] = f"{weather_effects.get('wind_speed')} mph {weather_effects.get('wind_direction')}"
                detailed_analysis['weather_precip'] = weather_effects.get('precipitation')
                detailed_analysis['weather_humidity'] = weather_effects.get('humidity')
                detailed_analysis['weather_description'] = weather_effects.get('description')
                detailed_analysis['home_weather_adj'] = weather_effects.get('home_adjustment')
                detailed_analysis['away_weather_adj'] = weather_effects.get('away_adjustment')

            summary_result = {
                'home_team': detailed_analysis['home_team'],
                'away_team': detailed_analysis['away_team'],
                'home_runs_scored': detailed_analysis['home_runs_scored'],
                'home_runs_allowed': detailed_analysis['home_runs_allowed'],
                'away_runs_scored': detailed_analysis['away_runs_scored'],
                'away_runs_allowed': detailed_analysis['away_runs_allowed'],
                'player_model_winner': detailed_analysis['player_model_winner'],
                'player_model_home_win_prob': detailed_analysis['player_model_home_win_prob'],
                'team_model_winner': detailed_analysis['team_model_winner'],
                'team_model_home_win_prob': detailed_analysis['team_model_home_win_prob'],
                'combined_home_win_prob': detailed_analysis['combined_home_win_prob'],
                'predicted_winner': detailed_analysis['predicted_winner'],
                'win_probability': detailed_analysis['win_probability'],
                'predicted_score': detailed_analysis['predicted_score'],
                'agreement': detailed_analysis['agreement'],
            }
            
            # Add weather data to summary if available
            if hasattr(predictor, 'weather_data') and 'weather_effects' in game_prediction:
                weather_effects = game_prediction['weather_effects']
                summary_result['weather_description'] = weather_effects.get('description')
                summary_result['home_weather_adj'] = weather_effects.get('home_adjustment')
                summary_result['away_weather_adj'] = weather_effects.get('away_adjustment')
            
            # Save the summary as CSV
            summary_df = pd.DataFrame([summary_result])
            summary_df.to_csv(f"analysis_summary_{away_team}_at_{home_team}_{args.target_date}.csv", index=False)
            summary_concat = pd.concat([summary_concat, summary_df], ignore_index=True)

            # Save the individual dataframes as separate sheets in an Excel file
            with pd.ExcelWriter(f"detailed_analysis_{away_team}_at_{home_team}_{args.target_date}.xlsx") as writer:
                # Summary sheet
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Player stats sheets
                home_batting_df = pd.DataFrame(detailed_analysis['home_batting_stats'])
                away_batting_df = pd.DataFrame(detailed_analysis['away_batting_stats'])
                home_pitching_df = pd.DataFrame(detailed_analysis['home_pitching_stats'])
                away_pitching_df = pd.DataFrame(detailed_analysis['away_pitching_stats'])
                
                if not home_batting_df.empty:
                    home_batting_df.to_excel(writer, sheet_name='Home_Batting', index=False)
                if not away_batting_df.empty:
                    away_batting_df.to_excel(writer, sheet_name='Away_Batting', index=False)
                if not home_pitching_df.empty:
                    home_pitching_df.to_excel(writer, sheet_name='Home_Pitching', index=False)
                if not away_pitching_df.empty:
                    away_pitching_df.to_excel(writer, sheet_name='Away_Pitching', index=False)

        # Create DataFrame from all predictions
        predictions_df = pd.DataFrame(predictions_data)

        # Export to CSV
        summary_concat.to_csv(f"summary_{args.target_date}.csv", index=False)
        predictions_df.to_csv(args.output_file, index=False)
        print(f"\nPredictions exported to {args.output_file}")
    
    except Exception as e:
        print(f"Error processing lineup data: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()