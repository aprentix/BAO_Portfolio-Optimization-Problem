import pandas as pd
import numpy as np
import shutil
from os import makedirs
from os.path import isdir
from typing import Dict, Tuple
from pathlib import Path
from inspyred import ec
from random import Random

from src.ga.ga_portfolio_optimization import GAPortfolioOptimization

class GAExperimentExecuter:
    def __init__(self, data_path="data/returns.csv"):
        self.data_path = Path(data_path)
        self.dataset = self._load_dataset()
        
    def _load_dataset(self):
        """Load and preprocess returns dataset"""
        df = pd.read_csv(self.data_path, index_col=0)
        self.asset_names = df.columns.tolist()
        return df

    def run_single_experiment(self, experiment_config: Dict, seed=None):
        """
        Runs a single portfolio optimization experiment
        :param experiment_config: Dictionary with experiment parameters:
            - n_assets: number of assets to include
            - strategy: 'random', 'high_corr' or 'low_corr'
            - risk_free_rate: risk-free rate for Sharpe ratio
            - max_weight: maximum allowed weight per asset
        """
        # Get experiment parameters
        n_assets = experiment_config.get('n_assets', 50)
        strategy = experiment_config.get('strategy', 'random')
        risk_free_rate = experiment_config.get('risk_free_rate', 0.042)
        max_weight = experiment_config.get('max_weight', 0.1)

        # Create asset subset
        returns_subset = self._create_subset(n_assets, strategy, seed)
        mean_returns = returns_subset.mean(axis=0).values
        cov_matrix = np.cov(returns_subset, rowvar=False)

        # Initialize and run GA
        ga = GAPortfolioOptimization(
            num_assets=n_assets,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            **self._get_ga_params(n_assets)
        )
        
        best_weights, best_sharpe = ga.run(seed=seed)
        return ga, best_weights, best_sharpe

    def run_repeated_experiment(self, experiment_config: Dict, n_repeat=31):
        """Run multiple trials of the same experiment configuration"""
        results = []
        for i in range(n_repeat):
            seed = 1 + i  # Different seed for each trial
            ga, weights, sharpe = self.run_single_experiment(
                experiment_config, seed=seed
            )
            results.append({
                'sharpe_ratio': sharpe,
                'max_weight': np.max(weights),
                'diversity': np.std(weights),
                'n_evaluations': ga.num_evaluations
            })
        return pd.DataFrame(results)

    def run_all_experiments(self, experiment_folder: str, 
                          experiment_groups: Dict,
                          overwrite=False,
                          n_repeat=31):
        """Run all experiment configurations"""
        if isdir(experiment_folder):
            if overwrite:
                shutil.rmtree(experiment_folder)
            else:
                raise ValueError(f"Folder {experiment_folder} exists. Use overwrite=True")

        makedirs(experiment_folder)

        # Generate experiment configurations
        experiments = self._generate_experiments(experiment_groups)

        for exp_id, config in enumerate(experiments):
            print(f"Running experiment {exp_id}: {config}")
            results = self.run_repeated_experiment(config, n_repeat)
            results['experiment_id'] = exp_id
            results.to_csv(f"{experiment_folder}/exp_{exp_id}.csv", index=False)

        print(f"All experiments saved to {experiment_folder}")

    def _create_subset(self, n_assets: int, strategy: str, seed=None):
        """Create asset subset based on selection strategy"""
        if n_assets > len(self.asset_names):
            raise ValueError("Requested assets exceed available data")

        if strategy == 'random':
            return self.dataset.sample(n=n_assets, axis=1, random_state=seed)
        
        corr_matrix = self.dataset.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        avg_corr = corr_matrix.mean()

        if strategy == 'high_corr':
            selected = avg_corr.nlargest(n_assets).index
        elif strategy == 'low_corr':
            selected = avg_corr.nsmallest(n_assets).index
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return self.dataset[selected]

    def _get_ga_params(self, n_assets: int) -> Dict:
        """Get GA parameters based on problem size"""
        base_params = {
            'selector': ec.selectors.tournament_selection,
            'tournament_size': 2,
            'mutation_rate': 0.1,
            'gaussian_stdev': 0.05,
            'max_generations': 100,
            'num_elites': 1
        }

        if n_assets < 50:
            base_params['pop_size'] = 50
        elif n_assets < 100:
            base_params['pop_size'] = 100
        else:
            base_params['pop_size'] = 200

        return base_params

    def _generate_experiments(self, experiment_groups: Dict):
        """Generate experiment configurations from groups"""
        experiments = []
        for group in experiment_groups.get('size_variation', []):
            for n in group['n_assets']:
                for strategy in group['strategies']:
                    experiments.append({
                        'n_assets': n,
                        'strategy': strategy,
                        'risk_free_rate': 0.042,
                        'max_weight': 0.1
                    })
        return experiments