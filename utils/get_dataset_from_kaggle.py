import os

def download_sentiment140():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    kaggle_config_dir = os.path.join(current_directory, '.kaggle')
    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config_dir

    dataset_name = "kazanova/sentiment140"
    download_path = os.path.join(current_directory, "sentiment140")

    try:
        print("Downloading Sentiment140 dataset...")
        os.system(f"kaggle datasets download -d {dataset_name} -p {download_path} --unzip")
        print(f"Dataset downloaded and extracted to {download_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_sentiment140()
