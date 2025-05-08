#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseball Statistics Prediction Algorithm

This module provides functionality to predict baseball player and team statistics
using Logistic Regression with Gradient Boosting based on historical performance data.
"""

import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
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
            self._validate_data()
            
            # Preprocess the data
            self._preprocess_data()
            
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
    
    def _validate_data(self) -> None:
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
    
    def _preprocess_data(self) -> None:
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
        self._create_team_performance_metrics()
        
        self.logger.info("Data preprocessing completed")
    
    def _create_team_performance_metrics(self) -> None:
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
                    self.logger.warning(f"Mismatched game data at index {i}, skipping")
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
    
    def train_models(self) -> None:
        """
        Train the prediction models using the preprocessed data.
        
        This method trains separate models for batting statistics, pitching statistics,
        and team performance predictions.
        """
        try:
            self.logger.info("Training prediction models")
            
            # Train batting models
            self._train_batting_models()
            
            # Train pitching models
            self._train_pitching_models()
            
            # Train team models
            self._train_team_models()
            
            self.logger.info("All models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            raise
    
    def _train_batting_models(self) -> None:
        """
        Train models for predicting batting statistics.
        
        This includes models for hits, home runs, RBIs, and other batting metrics.
        """
        self.logger.info("Training batting models")
        
        # Define the batting metrics to predict
        batting_metrics = ['H', 'R', 'RBI', 'HR', 'AVG', 'OBP']  # Added OBP here
        
        for metric in batting_metrics:
            try:
                # Skip if metric doesn't exist in the data
                if metric not in self.batting_data.columns:
                    self.logger.warning(f"Metric {metric} not found in batting data, skipping")
                    continue
                
                # Prepare features and target
                X = self.batting_data[['AB', 'BB', 'SO', 'PA', 'Pit', 'Str', 'HR', 'Double', 'Triple', 'SB', 'GDP', 'Home']].copy()

                # Add derived metrics
                X['BB_per_PA'] = X['BB'] / X['PA'].replace(0, 0.1)  # Walk rate
                X['SO_per_PA'] = X['SO'] / X['PA'].replace(0, 0.1)  # Strikeout rate
                X['Contact_Rate'] = (X['AB'] - X['SO']) / X['AB'].replace(0, 0.1)  # Contact rate
                X['XBH_per_AB'] = (X['HR'] + X['Double'] + X['Triple']) / X['AB'].replace(0, 0.1)  # Extra base hit rate
                X['Str_per_Pit'] = X['Str'] / X['Pit'].replace(0, 1)  # Strike percentage
                X['Power_Factor'] = (X['HR'] * 4 + X['Triple'] * 3 + X['Double'] * 2) / X['AB'].replace(0, 0.1)  # Power indicator

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
                
                # Create and train the model
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate the model
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                self.logger.info(f"Batting {metric} model RMSE: {rmse:.4f}")
                
                # Store the model
                self.batting_models[metric] = model
                
            except Exception as e:
                self.logger.error(f"Error training batting model for {metric}: {e}")
                continue
        
    def _train_pitching_models(self) -> None:
        """
        Train models for predicting pitching statistics.
        
        This includes models for ERA, WHIP, strikeouts, and other pitching metrics.
        """
        self.logger.info("Training pitching models")
        
        # Define the pitching metrics to predict
        pitching_metrics = ['IP', 'H', 'R', 'ER', 'ERA', 'WHIP', 'SO']
        
        for metric in pitching_metrics:
            try:
                # Skip if metric doesn't exist in the data
                if metric not in self.pitching_data.columns:
                    self.logger.warning(f"Metric {metric} not found in pitching data, skipping")
                    continue
                
                # Prepare features and target
                X = self.pitching_data[['IP', 'SO', 'BB', 'HR', 'GB', 'FB', 'Str', 'Pit', 'BF', 'Home']].copy()

                # Calculate derived metrics
                X['K_per_9'] = 9 * X['SO'] / X['IP'].replace(0, 0.1)  # Strikeouts per 9 innings
                X['BB_per_9'] = 9 * X['BB'] / X['IP'].replace(0, 0.1)  # Walks per 9 innings
                X['HR_per_9'] = 9 * X['HR'] / X['IP'].replace(0, 0.1)  # Home runs per 9 innings
                X['Str_pct'] = X['Str'] / X['Pit'].replace(0, 1)  # Strike percentage
                X['GB_FB_ratio'] = X['GB'] / X['FB'].replace(0, 1)  # Ground ball to fly ball ratio

                # Handle any NaN or infinite values that might have been created
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

                y = self.pitching_data[metric].copy()

                # Add team encoding
                X = pd.get_dummies(X, columns=['Home'], drop_first=True)

                # Handle missing values
                X = X.fillna(0)
                y = y.fillna(0)
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Create and train the model
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate the model
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                self.logger.info(f"Pitching {metric} model RMSE: {rmse:.4f}")
                
                # Store the model
                self.pitching_models[metric] = model
                
            except Exception as e:
                self.logger.error(f"Error training pitching model for {metric}: {e}")
                continue
    
    def _train_team_models(self) -> None:
        """
        Train models for predicting team performance.
        
        This includes models for win probability in different game contexts.
        """
        self.logger.info("Training team win prediction model")
        
        try:
            # Create features for team win prediction
            game_results = []
            
            # Process the linescore data two rows at a time
            i = 0
            while i < len(self.linescore_data) - 1:
                try:
                    # Get the away team row and home team row
                    away_row = self.linescore_data.iloc[i]
                    home_row = self.linescore_data.iloc[i+1]
                    
                    # Verify that these rows are for the same game
                    if away_row['Date'] != home_row['Date'] or away_row['AwayTeam'] != home_row['AwayTeam'] or away_row['HomeTeam'] != home_row['HomeTeam']:
                        # These rows don't represent the same game, skip to next row
                        self.logger.warning(f"Mismatched game data at index {i}, skipping")
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
                        
                        # Get team records up to this point
                        home_record = self.team_records.loc[home_team].copy()
                        away_record = self.team_records.loc[away_team].copy()
                        
                        # Create features
                        features = {
                            'HomeTeamWinPct': home_record['WinPct'],
                            'HomeTeamHomeWinPct': home_record['HomeWinPct'],
                            'AwayTeamWinPct': away_record['WinPct'],
                            'AwayTeamAwayWinPct': away_record['AwayWinPct'],
                            'HomeTeamWin': 1 if home_runs > away_runs else 0
                        }
                        
                        game_results.append(features)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Error processing game for team model: {e}")
                    
                    # Move to the next game (skip two rows)
                    i += 2
                except Exception as e:
                    self.logger.warning(f"Error processing game at index {i}: {e}")
                    i += 1  # Move to next row if there's an error
            
            # Create a DataFrame from the game results
            if not game_results:
                self.logger.warning("No valid game results found for team model training")
                return
                
            games_df = pd.DataFrame(game_results)
            
            # Continue with the rest of the method as before
            # Prepare features and target
            X = games_df.drop('HomeTeamWin', axis=1)
            y = games_df['HomeTeamWin']
            
            # Handle missing values
            X = X.fillna(0)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train the model
            model = LogisticRegression(class_weight='balanced', random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"Team win prediction model accuracy: {accuracy:.4f}")
            
            # Store the model
            self.team_models['win_prediction'] = model
            
        except Exception as e:
            self.logger.error(f"Error training team models: {e}")
            raise
    
    def get_player_stats(self, player_name: str, target_date: str, 
                        last_n_games: int = 10) -> Dict[str, Dict[str, float]]:
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
                (self.pitching_data['Pitching'] == player_name) & 
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
                
                # Calculate home stats
                if not home_data.empty:
                    result['home'] = {
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
                
                # Calculate away stats
                if not away_data.empty:
                    result['away'] = {
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
                        self.logger.warning(f"Mismatched game data at index {i}, skipping")
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
    
    def predict_player_performance(self, player_name: str, is_home_game: bool, 
                                opponent: str, target_date: str) -> Dict[str, Dict[str, float]]:
        """
        Predict player performance for an upcoming game.
        
        Args:
            player_name (str): Name of the player to predict performance for.
            is_home_game (bool): Whether the player's team is the home team.
            opponent (str): Name of the opposing team.
            target_date (str): The date of the game to predict.
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary containing predicted performance
                metrics and confidence intervals.
        """
        try:
            # First, check if the player is a batter or pitcher
            is_batter = False
            is_pitcher = False
            
            player_team = None
            
            # Try to determine if player is a batter
            batter_info = self.batting_data[self.batting_data['Batting'].str.contains(player_name, case=True, na=False)]
            if not batter_info.empty:
                is_batter = True
                player_team = batter_info['Team'].iloc[0]
            
            # Try to determine if player is a pitcher
            pitcher_info = self.pitching_data[self.pitching_data['Pitching'] == player_name]
            if not pitcher_info.empty:
                is_pitcher = True
                player_team = pitcher_info['Team'].iloc[0]
            
            if not is_batter and not is_pitcher:
                self.logger.warning(f"Player {player_name} not found in batting or pitching data")
                return {'prediction': {}, 'confidence': {}}
            
            # Get historical player statistics
            player_stats = self.get_player_stats(player_name, target_date)
            
            # Get team statistics
            team_stats = self.get_team_stats(player_team, target_date)
            opponent_stats = self.get_team_stats(opponent, target_date)
            
            prediction_result = {
                'prediction': {},
                'confidence': {}
            }
            
            # Make predictions based on player type
            # Inside predict_player_performance method, in the is_batter section:
            if is_batter:
                # Use batting models to make predictions
                for metric, model in self.batting_models.items():
                    # Create feature vector for prediction
                    features_dict = {
                        'AB': player_stats['overall'].get('AB', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'BB': player_stats['overall'].get('BB', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'SO': player_stats['overall'].get('SO', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'PA': player_stats['overall'].get('PA', player_stats['overall'].get('AB', 0) + player_stats['overall'].get('BB', 0)) / max(player_stats['overall'].get('games', 1), 1),
                        'Pit': player_stats['overall'].get('Pit', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'Str': player_stats['overall'].get('Str', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'HR': player_stats['overall'].get('HR', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'Double': player_stats['overall'].get('Double', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'Triple': player_stats['overall'].get('Triple', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'SB': player_stats['overall'].get('SB', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'GDP': player_stats['overall'].get('GDP', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'Home_True': 1 if is_home_game else 0
                    }

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

                    # Handle missing features based on model requirements
                    if hasattr(model, 'feature_names_in_'):
                        required_features = model.feature_names_in_
                        features = pd.DataFrame([features_dict])
                        
                        # Add any missing features with default value 0
                        for feature in required_features:
                            if feature not in features.columns:
                                features[feature] = 0
                                
                        # Ensure correct column order
                        features = features[required_features]
                    else:
                        features = pd.DataFrame([features_dict])
                        
                        # Create DataFrame with exactly the required features in the right order
                        features = pd.DataFrame([features_dict])[required_features]

                    # Make prediction
                    prediction = model.predict(features)[0]
                    
                    # Calculate confidence interval (using model's standard deviation)
                    std_dev = np.std(model.predict(features))
                    lower_bound = max(0, prediction - 1.96 * std_dev)
                    upper_bound = prediction + 1.96 * std_dev
                    
                    # Store prediction and confidence interval
                    prediction_result['prediction'][metric] = prediction
                    prediction_result['confidence'][metric] = (lower_bound, upper_bound)
                
            elif is_pitcher:
                # Use pitching models to make predictions
                for metric, model in self.pitching_models.items():
                    # Skip if the metric is the same as the feature
                    if metric == 'IP':
                        prediction_result['prediction'][metric] = player_stats['overall'].get('IP', 0) / player_stats['overall'].get('games', 1)
                        prediction_result['confidence'][metric] = (
                            prediction_result['prediction'][metric] * 0.7, 
                            prediction_result['prediction'][metric] * 1.3
                        )
                        continue
                    
                    # Create feature vector for prediction
                    features_dict = {
                        'IP': player_stats['overall'].get('IP', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'SO': player_stats['overall'].get('SO', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'BB': player_stats['overall'].get('BB', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'HR': player_stats['overall'].get('HR', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'GB': player_stats['overall'].get('GB', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'FB': player_stats['overall'].get('FB', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'Str': player_stats['overall'].get('Str', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'Pit': player_stats['overall'].get('Pit', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'BF': player_stats['overall'].get('BF', 0) / max(player_stats['overall'].get('games', 1), 1),
                        'Home_True': 1 if is_home_game else 0
                    }

                    # Calculate derived metrics
                    ip = max(player_stats['overall'].get('IP', 0), 0.1)
                    pit = max(player_stats['overall'].get('Pit', 0), 1)
                    fb = max(player_stats['overall'].get('FB', 0), 1)

                    features_dict['K_per_9'] = 9 * player_stats['overall'].get('SO', 0) / ip
                    features_dict['BB_per_9'] = 9 * player_stats['overall'].get('BB', 0) / ip
                    features_dict['HR_per_9'] = 9 * player_stats['overall'].get('HR', 0) / ip
                    features_dict['Str_pct'] = player_stats['overall'].get('Str', 0) / pit
                    features_dict['GB_FB_ratio'] = player_stats['overall'].get('GB', 0) / fb

                    # Handle missing features based on model requirements
                    if hasattr(model, 'feature_names_in_'):
                        required_features = model.feature_names_in_
                        features = pd.DataFrame([features_dict])
                        
                        # Add any missing features with default value 0
                        for feature in required_features:
                            if feature not in features.columns:
                                features[feature] = 0
                                
                        # Ensure correct column order
                        features = features[required_features]
                    else:
                        features = pd.DataFrame([features_dict])
                    
                    # Make prediction
                    prediction = model.predict(features)[0]
                    
                    # Calculate confidence interval (using model's standard deviation)
                    std_dev = np.std(model.predict(features))
                    lower_bound = max(0, prediction - 1.96 * std_dev)
                    upper_bound = prediction + 1.96 * std_dev
                    
                    # Store prediction and confidence interval
                    prediction_result['prediction'][metric] = prediction
                    prediction_result['confidence'][metric] = (lower_bound, upper_bound)
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Error predicting player performance for {player_name}: {e}")
            return {'prediction': {}, 'confidence': {}}
    
    def predict_game_winner(self, home_team: str, away_team: str, 
                        target_date: str) -> Dict[str, float]:
        """
        Predict the winner of an upcoming game.
        
        Args:
            home_team (str): Name of the home team.
            away_team (str): Name of the away team.
            target_date (str): The date of the game to predict.
            
        Returns:
            Dict[str, float]: Dictionary containing win probabilities for both teams.
        """
        try:
            # Get team statistics
            home_team_stats = self.get_team_stats(home_team, target_date)
            away_team_stats = self.get_team_stats(away_team, target_date)
            
            if not home_team_stats or not away_team_stats:
                self.logger.warning(f"Missing team stats for {home_team} or {away_team}")
                return {
                    'home_win_probability': 0.5,
                    'away_win_probability': 0.5,
                    'confidence': 'low'
                }
            
            # Create feature vector for prediction
            features = pd.DataFrame({
                'HomeTeamWinPct': [home_team_stats.get('win_pct', 0)],
                'HomeTeamHomeWinPct': [home_team_stats.get('home_win_pct', 0)],
                'AwayTeamWinPct': [away_team_stats.get('win_pct', 0)],
                'AwayTeamAwayWinPct': [away_team_stats.get('away_win_pct', 0)]
            })
            
            # Make prediction if we have a team win prediction model
            if 'win_prediction' in self.team_models:
                model = self.team_models['win_prediction']
                
                # Get win probability for home team
                home_win_prob = model.predict_proba(features)[0][1]
                
                # Determine confidence level based on probability range
                if 0.4 <= home_win_prob <= 0.6:
                    confidence = 'low'
                elif 0.3 <= home_win_prob < 0.4 or 0.6 < home_win_prob <= 0.7:
                    confidence = 'medium'
                else:
                    confidence = 'high'
                
                return {
                    'home_win_probability': home_win_prob,
                    'away_win_probability': 1 - home_win_prob,
                    'confidence': confidence
                }
            else:
                # No model available, use simple heuristic
                home_factor = home_team_stats.get('home_win_pct', 0) * 2
                away_factor = away_team_stats.get('away_win_pct', 0) * 1.5
                
                total_factor = home_factor + away_factor
                if total_factor == 0:
                    home_win_prob = 0.5  # 50-50 if no data
                else:
                    home_win_prob = home_factor / total_factor
                
                return {
                    'home_win_probability': home_win_prob,
                    'away_win_probability': 1 - home_win_prob,
                    'confidence': 'low'  # Low confidence without a trained model
                }
            
        except Exception as e:
            self.logger.error(f"Error predicting game winner for {home_team} vs {away_team}: {e}")
            return {
                'home_win_probability': 0.5,
                'away_win_probability': 0.5,
                'confidence': 'low'
            }
    
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
    def analyze_game_outcome(self, game_prediction: Dict[str, any], 
                        home_players: List[Dict], away_players: List[Dict],
                        home_team: str, away_team: str) -> Dict[str, any]:
        """
        Analyze game outcome based on detailed player projections.
        
        This function creates dataframes of projected batting and pitching stats,
        calculates expected run differentials, and compares with the team model
        prediction to determine the most likely winner.
        
        Args:
            game_prediction (Dict): Team model's game prediction dictionary
            home_players (List[Dict]): List of home team player prediction dictionaries
            away_players (List[Dict]): List of away team player prediction dictionaries
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            
        Returns:
            Dict: Enhanced game prediction with detailed analysis
        """
        try:
            # Separate batters and pitchers
            home_batters = [p for p in home_players if p['position'] != 'P']
            home_pitchers = [p for p in home_players if p['position'] == 'P']
            away_batters = [p for p in away_players if p['position'] != 'P']
            away_pitchers = [p for p in away_players if p['position'] == 'P']
            
            # Create batting dataframes
            home_batting_data = []
            for player in home_batters:
                
                    row = {
                        'name': player['player_name'],
                        'position': player['position'],
                        'order': player['batting_order'],
                        'R': player.get('pred_R', 0),
                        'H': player.get('pred_H', 0),
                        'RBI': player.get('pred_RBI', 0),
                        'BB': player.get('pred_OBP', 0)
                    }
                    home_batting_data.append(row)
            
            away_batting_data = []
            for player in away_batters:
                    row = {
                        'name': player['player_name'],
                        'position': player['position'],
                        'order': player['batting_order'],
                        'R': player.get('pred_R', 0),
                        'H': player.get('pred_H', 0),
                        'RBI': player.get('pred_RBI', 0),
                        'BB': player.get('pred_OBP', 0)
                    }
                    away_batting_data.append(row)
            
            # Create pitching dataframes
            home_pitching_data = []
            for player in home_pitchers:
                
                    row = {
                        'name': player['player_name'],
                        'IP': player.get('pred_IP', 0),
                        'H': player.get('pred_H', 0),
                        'R': player.get('pred_R', 0),
                        'ER': player.get('pred_ERA', 0),
                        #'BB': player['predictions'].get('BB', 0),
                        'SO': player.get('pred_SO', 0)
                    }
                    home_pitching_data.append(row)
            
            away_pitching_data = []
            for player in away_pitchers:
                
                    row = {
                        'name': player['player_name'],
                        'IP': player.get('pred_IP', 0),
                        'H': player.get('pred_H', 0),
                        'R': player.get('pred_R', 0),
                        'ER': player.get('pred_ERA', 0),
                        #'BB': player['predictions'].get('BB', 0),
                        'SO': player.get('pred_SO', 0)
                    }
                    away_pitching_data.append(row)
            
            # Convert to dataframes
            home_batting_df = pd.DataFrame(home_batting_data)
            away_batting_df = pd.DataFrame(away_batting_data)
            home_pitching_df = pd.DataFrame(home_pitching_data)
            away_pitching_df = pd.DataFrame(away_pitching_data)
            
            # Calculate team-level run production and prevention
            home_runs_scored = home_batting_df['R'].sum() if 'R' in home_batting_df.columns else 0
            home_runs_allowed = home_pitching_df['R'].sum() if 'R' in home_pitching_df.columns else 0
            away_runs_scored = away_batting_df['R'].sum() if 'R' in away_batting_df.columns else 0
            away_runs_allowed = away_pitching_df['R'].sum() if 'R' in away_pitching_df.columns else 0
            
            # Compare with team model prediction
            team_model_home_win_prob = game_prediction['home_win_probability']
            team_model_winner = home_team if team_model_home_win_prob > 0.5 else away_team
            
            # Calculate expected score
            expected_home_score = max(0, round((home_runs_scored + away_runs_allowed) / 2))
            expected_away_score = max(0, round((away_runs_scored + home_runs_allowed) / 2))
            
            if expected_home_score > expected_away_score:
                player_model_winner = home_team
                margin = expected_home_score - expected_away_score
                player_model_home_win_prob = 0.5 + min(0.4, margin * 0.05)  # Cap at 90%
            elif expected_home_score < expected_away_score:
                player_model_winner = away_team
                margin = expected_away_score - expected_home_score
                player_model_home_win_prob = 0.5 - min(0.4, margin * 0.05)
            else:
                player_model_winner = "Tie - Slight edge to home team"
                player_model_home_win_prob = 0.52

            # Calculate weighted final prediction (60% team model, 40% player model)
            combined_home_win_prob = (team_model_home_win_prob * 0.4) + (player_model_home_win_prob * 0.6)
            final_winner = home_team if combined_home_win_prob > 0.5 else away_team
            final_win_prob = combined_home_win_prob if final_winner == home_team else 1 - combined_home_win_prob

            # Ensure winner has higher score
            if final_winner == home_team and expected_home_score <= expected_away_score:
                expected_home_score = expected_away_score + 1
            elif final_winner == away_team and expected_away_score <= expected_home_score:
                expected_away_score = expected_home_score + 1
            
            # Create detailed analysis result
            analysis_result = {
                'home_team': home_team,
                'away_team': away_team,
                'home_batting_stats': home_batting_df.to_dict('records') if not home_batting_df.empty else [],
                'away_batting_stats': away_batting_df.to_dict('records') if not away_batting_df.empty else [],
                'home_pitching_stats': home_pitching_df.to_dict('records') if not home_pitching_df.empty else [],
                'away_pitching_stats': away_pitching_df.to_dict('records') if not away_pitching_df.empty else [],
                'home_runs_scored': home_runs_scored,
                'home_runs_allowed': home_runs_allowed,
                'away_runs_scored': away_runs_scored,
                'away_runs_allowed': away_runs_allowed,
                'player_model_winner': player_model_winner,
                'player_model_home_win_prob': player_model_home_win_prob,
                'team_model_winner': team_model_winner,
                'team_model_home_win_prob': team_model_home_win_prob,
                'combined_home_win_prob': combined_home_win_prob,
                'predicted_winner': final_winner,
                'win_probability': final_win_prob,
                'predicted_score': f"{away_team} {expected_away_score}, {home_team} {expected_home_score}",
                'agreement': player_model_winner == team_model_winner,
                'confidence': 'high' if abs(combined_home_win_prob - 0.5) > 0.2 else 
                            ('medium' if abs(combined_home_win_prob - 0.5) > 0.1 else 'low')
            }
            
            # Add original game prediction data
            analysis_result.update({
                'original_prediction': game_prediction
            })
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing game outcome: {e}")
            return {
                'error': str(e),
                'predicted_winner': game_prediction.get('predicted_winner', 'Unknown'),
                'original_prediction': game_prediction
            }

    

def main():
    """
    Main function to demonstrate usage of the BaseballPredictor class with lineup data.
    """
    import argparse
    import traceback
    import pandas as pd
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Predict baseball statistics.')
    parser.add_argument('--data-path', type=str, default='.', help='Path to the data files.')
    parser.add_argument('--target-date', type=str, required=True, help='Target date for predictions (YYYY-MM-DD).')
    parser.add_argument('--lineup-file', type=str, default='2024_lineups.csv', 
                        help='CSV file containing team lineups.')
    parser.add_argument('--train', action='store_true', help='Train new models.')
    parser.add_argument('--save-models', action='store_true', help='Save trained models.')
    parser.add_argument('--load-models', action='store_true', help='Load existing models.')
    parser.add_argument('--output-file', type=str, default='', 
                        help='Output CSV file for predictions. Default: predictions_YYYY-MM-DD.csv')
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output_file:
        args.output_file = f"predictions_{args.target_date}.csv"
    
    # Initialize the predictor
    predictor = BaseballPredictor(data_path=args.data_path)
    
    # Load data
    predictor.load_data(args.target_date)
    
    # Train or load models
    if args.train:
        predictor.train_models()
        if args.save_models:
            predictor.save_models()
    elif args.load_models:
        predictor.load_models()
    else:
        predictor.train_models()  # Default to training if not specified
    
    # Load lineup data
    lineup_file = os.path.join(args.data_path, args.lineup_file)
    if not os.path.exists(lineup_file):
        print(f"Lineup file not found: {lineup_file}")
        return
    
    try:
        # Replace the lineup handling part in main()
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
            
            # Predict game winner
            game_prediction = predictor.predict_game_winner(
                home_team,
                away_team,
                args.target_date
            )
            
            print(f"Home Win Probability: {game_prediction['home_win_probability']:.2%}")
            print(f"Away Win Probability: {game_prediction['away_win_probability']:.2%}")
            print(f"Confidence: {game_prediction['confidence']}")

            predictions_home = []
            predictions_away = []
            # Process away team lineup
            print(f"\n{away_team} Lineup Predictions:")
            away_players = away_lineup[away_lineup['player_name'].notna()]
            for _, player in away_players.iterrows():
                if pd.isna(player['player_name']) or player['player_name'] == '':
                    continue
                    
                # Predict player performance
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
                
                # Add game prediction
                player_row['team_win_prob'] = game_prediction['away_win_probability']
                
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
            
            # Process home team lineup
            print(f"\n{home_team} Lineup Predictions:")
            home_players = home_lineup[home_lineup['player_name'].notna()]
            for _, player in home_players.iterrows():
                if pd.isna(player['player_name']) or player['player_name'] == '':
                    continue
                    
                # Predict player performance
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
                
                # Add game prediction
                player_row['team_win_prob'] = game_prediction['home_win_probability']
                
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

            # Run detailed game analysis
            detailed_analysis = predictor.analyze_game_outcome(
                game_prediction,
                predictions_home,
                predictions_away,
                home_team,
                away_team
            )

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