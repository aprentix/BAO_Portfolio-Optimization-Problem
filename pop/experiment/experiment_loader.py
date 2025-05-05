import pandas as pd
from pathlib import Path

class ExperimentLoader:
    """
    A utility class to load experiment configurations from CSV files.
    """

    @staticmethod
    def load_experiments(folder_path: str) -> list[dict]:
        """
        Load all experiment configurations from CSV files in a folder.

        Args:
            folder_path (str): Path to the folder containing experiment CSV files.

        Returns:
            list[dict]: A list of experiment configurations as dictionaries.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder {folder_path} does not exist.")

        experiments = []
        for csv_file in folder.glob("*.csv"):
            df = pd.read_csv(csv_file)
            experiments.extend(df.to_dict(orient="records"))
        return experiments