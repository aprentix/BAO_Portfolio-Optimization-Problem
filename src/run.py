from dataset import DatasetManager

def main():
    dataset_manager = DatasetManager()
    dataset_manager.read_anual_resume(0.042, "2015-01-01", "2020-01-01")


if __name__ == "__main__":
    main()
