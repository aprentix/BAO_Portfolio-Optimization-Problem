import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from typing import Dict

class ExperimentLoader:
    
    def __init__(self, experiment_folder: str, returns_path: str):
        self.experiment_folder = experiment_folder
        self.returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        self.experiment_data = self._load_and_enrich_data()
        
    def _load_data(self) -> pd.DataFrame:
        """Load raw experiment results from CSV files"""
        files = [f for f in listdir(self.experiment_folder) 
                if isfile(join(self.experiment_folder, f)) and f.endswith('.csv')]
        
        dfs = []
        for f in files:
            df = pd.read_csv(join(self.experiment_folder, f))
            df['experiment_file'] = f  # Preserve filename context
            dfs.append(df)
            
        return pd.concat(dfs, ignore_index=True)
    
    def _process_weights(self, row: Dict) -> Dict:
        """Convert weights string to array and calculate statistics"""
        weights = np.array(eval(row['weights']))
        return {
            'max_weight': weights.max(),
            'min_weight': weights.min(),
            'num_assets': len(weights),
            'diversity': weights.std(),
            'weights': weights  # Keep raw weights for later analysis
        }
    
    def _calculate_performance(self, row: Dict) -> Dict:
        """Calculate portfolio performance metrics using returns data"""
        weights = row['weights']
        portfolio_returns = self.returns[row['assets']].values @ weights
        return {
            'annualized_return': np.prod(1 + portfolio_returns) ** (252/len(portfolio_returns)) - 1,
            'annualized_volatility': portfolio_returns.std() * np.sqrt(252),
            'max_drawdown': (1 - (portfolio_returns + 1).cumprod()).max()
        }
    
    def _load_and_enrich_data(self) -> pd.DataFrame:
        """Full data loading and enrichment pipeline"""
        # Load base experiment data
        raw_data = self._load_data()
        
        # Process weights column
        weights_data = raw_data.apply(self._process_weights, axis=1, result_type='expand')
        enriched_data = pd.concat([raw_data, weights_data], axis=1)
        
        # Calculate performance metrics
        performance_data = enriched_data.apply(self._calculate_performance, axis=1, result_type='expand')
        final_data = pd.concat([enriched_data, performance_data], axis=1)
        
        # Add experiment parameters
        final_data['strategy'] = final_data['experiment_file'].str.extract(r'strategy_(\w+)')
        final_data['num_assets'] = final_data['experiment_file'].str.extract(r'assets_(\d+)').astype(int)
        
        return final_data

    def get_analysis_dataframe(self) -> pd.DataFrame:
        """Get cleaned dataframe for analysis"""
        return self.experiment_data[[
            'experiment_id', 'strategy', 'num_assets', 
            'sharpe_ratio', 'annualized_return', 'annualized_volatility',
            'max_drawdown', 'max_weight', 'diversity'
        ]].copy()