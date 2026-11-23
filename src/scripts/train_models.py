import os

from utils.datamap import download_data_mapping


MODEL_REGISTRY_DIR = "model_registry"
DATA_MAP_PATH = f"{MODEL_REGISTRY_DIR}/snli_roberta_0_6_data_map_coordinates.jsonl"


def main():
    if not os.path.exists(DATA_MAP_PATH):
        download_data_mapping(output_path=DATA_MAP_PATH)


if __name__ == "__main__":
    main()
