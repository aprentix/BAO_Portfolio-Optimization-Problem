import pandas as pd
from pathlib import Path

class ExperimentLoader:
    """
    A utility class to load experiment configurations from CSV files.
    """

    @staticmethod
    def load_experiments(file_path: str) -> list[dict]:
        """
        Load experiment configurations from a CSV file.

        Args:
            file_path (str): Path to the CSV file containing experiment configurations.

        Returns:
            list[dict]: A list of experiment configurations as dictionaries.
        """
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f"Configuration file {file_path} does not exist.")

        # Load configurations into a list of dictionaries
        df = pd.read_csv(file)
        return df.to_dict(orient="records")