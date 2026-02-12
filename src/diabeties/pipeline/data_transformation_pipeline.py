import os
from src.diabeties import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from src.diabeties.entity.config_entity import DataTransformationConfig
from src.diabeties.config.configuration import ConfigurationManager  # Added to access schema dynamically

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        config_manager = ConfigurationManager()
        schema = config_manager.schema

        target_column = schema.TARGET_COLUMN.name
        categorical_columns = [
            col for col, dtype in schema.COLUMNS.items() if dtype == "object"
        ]

        y = data[target_column]
        X = data.drop(columns=[target_column])

        if categorical_columns:
            X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
            logger.info(f"Encoded categorical columns: {categorical_columns}")
            logger.info(f"New feature columns after encoding: {list(X_encoded.columns)}")
        else:
            X_encoded = X
            logger.info("No categorical columns found; proceeding without encoding.")

        train_x, test_x, train_y, test_y = train_test_split(
            X_encoded, y, test_size=0.20, random_state=42, stratify=y
        )

        train = pd.concat([train_x, train_y.reset_index(drop=True)], axis=1)
        test = pd.concat([test_x, test_y.reset_index(drop=True)], axis=1)

        train.rename(columns={train_y.name: target_column}, inplace=True)
        test.rename(columns={test_y.name: target_column}, inplace=True)

        train_path = os.path.join(self.config.root_dir, "train.csv")
        test_path = os.path.join(self.config.root_dir, "test.csv")

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        logger.info("Data transformation (encoding + splitting) completed successfully")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")
        logger.info(f"Train saved to: {train_path}")
        logger.info(f"Test saved to: {test_path}")