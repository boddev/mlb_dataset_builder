#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Aggregator for Baseball Statistics

This module provides functionality to aggregate baseball statistics
from individual game files into consolidated datasets for analysis.
"""

import os
import pandas as pd
import glob
import logging
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_aggregation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BaseballDataAggregator:
    """
    A class for aggregating baseball statistics from individual game files.
    
    This class processes individual game statistic files for batting, pitching,
    and linescore data and consolidates them into unified datasets.
    """
    
    def __init__(self, data_dir: str = 'files'):
        """
        Initialize the BaseballDataAggregator with data directory.
        
        Args:
            data_dir (str): Path to the directory containing the individual game files.
        """
        self.data_dir = data_dir
        self.logger = logger
        
    def aggregate_data(self, output_dir: str = '.', year: str = '2024') -> None:
        """
        Aggregate the baseball data from individual game files.
        
        Args:
            output_dir (str): Directory to save the aggregated data files.
            year (str): Year to include in output filenames.
            
        Returns:
            None
        """
        try:
            self.logger.info(f"Starting data aggregation from {self.data_dir}")
            
            # Initialize combined dataframes
            combined_batting = pd.DataFrame()
            combined_pitching = pd.DataFrame()
            combined_linescore = pd.DataFrame()
            
            # Function to remove unnamed columns
            def remove_unnamed_columns(df):
                return df.loc[:, ~df.columns.str.contains('Unnamed', case=False)]
            
            # Get all batting files
            batting_files = glob.glob(os.path.join(self.data_dir, '*_batting.csv'))
            self.logger.info(f"Found {len(batting_files)} batting files")
            
            # Process batting files
            for file in batting_files:
                try:
                    df = pd.read_csv(file)
                    df = remove_unnamed_columns(df)
                    combined_batting = pd.concat([combined_batting, df], ignore_index=True)
                except Exception as e:
                    self.logger.error(f"Error processing {file}: {e}")
            
            # Get all pitching files
            pitching_files = glob.glob(os.path.join(self.data_dir, '*_pitching.csv'))
            self.logger.info(f"Found {len(pitching_files)} pitching files")
            
            # Process pitching files
            for file in pitching_files:
                try:
                    df = pd.read_csv(file)
                    df = remove_unnamed_columns(df)
                    combined_pitching = pd.concat([combined_pitching, df], ignore_index=True)
                except Exception as e:
                    self.logger.error(f"Error processing {file}: {e}")
            
            # Get all linescore files
            linescore_files = glob.glob(os.path.join(self.data_dir, '*_linescore.csv'))
            self.logger.info(f"Found {len(linescore_files)} linescore files")
            
            # Process linescore files
            for file in linescore_files:
                try:
                    df = pd.read_csv(file)
                    df = remove_unnamed_columns(df)
                    combined_linescore = pd.concat([combined_linescore, df], ignore_index=True)
                except Exception as e:
                    self.logger.error(f"Error processing {file}: {e}")
            
            # Clean up data
            if not combined_batting.empty:
                # Convert date string to datetime if it's not already
                if 'Date' in combined_batting.columns and pd.api.types.is_string_dtype(combined_batting['Date']):
                    combined_batting['Date'] = pd.to_datetime(combined_batting['Date'])
                combined_batting = combined_batting.sort_values('Date')
            
            if not combined_pitching.empty:
                # Convert date string to datetime if it's not already
                if 'Date' in combined_pitching.columns and pd.api.types.is_string_dtype(combined_pitching['Date']):
                    combined_pitching['Date'] = pd.to_datetime(combined_pitching['Date'])
                combined_pitching = combined_pitching.sort_values('Date')
            
            if not combined_linescore.empty:
                # Convert date string to datetime if it's not already
                if 'Date' in combined_linescore.columns and pd.api.types.is_string_dtype(combined_linescore['Date']):
                    combined_linescore['Date'] = pd.to_datetime(combined_linescore['Date'])
                combined_linescore = combined_linescore.sort_values('Date')
            
            # Save combined data
            batting_output = os.path.join(output_dir, f'{year}_batting.csv')
            pitching_output = os.path.join(output_dir, f'{year}_pitching.csv')
            linescore_output = os.path.join(output_dir, f'{year}_linescore.csv')
            
            combined_batting.to_csv(batting_output, index=False)
            combined_pitching.to_csv(pitching_output, index=False)
            combined_linescore.to_csv(linescore_output, index=False)
            
            # Also save as Excel file
            excel_output = os.path.join(output_dir, f'{year}_stats.xlsx')
            with pd.ExcelWriter(excel_output, mode='w') as writer:
                combined_batting.to_excel(writer, sheet_name='batting', index=False)
                combined_pitching.to_excel(writer, sheet_name='pitching', index=False)
                combined_linescore.to_excel(writer, sheet_name='linescore', index=False)
            
            self.logger.info(f"Data aggregation complete. Files saved to {output_dir}")
            self.logger.info(f"Batting rows: {len(combined_batting)}, Pitching rows: {len(combined_pitching)}, Linescore rows: {len(combined_linescore)}")
            
        except Exception as e:
            self.logger.error(f"Error during data aggregation: {e}")
            raise

def main():
    """
    Main function to demonstrate usage of the BaseballDataAggregator class.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate baseball statistics.')
    parser.add_argument('--data-dir', type=str, default='files', help='Directory with individual game files.')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save aggregated files.')
    parser.add_argument('--year', type=str, default='2024', help='Year to include in output filenames.')
    
    args = parser.parse_args()
    
    # Initialize the aggregator
    aggregator = BaseballDataAggregator(data_dir=args.data_dir)
    
    # Aggregate the data
    aggregator.aggregate_data(output_dir=args.output_dir, year=args.year)
    
    print(f"Data aggregation complete. Files saved to {args.output_dir}")

if __name__ == "__main__":
    main()

