import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object


logger = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> object:

        try:
            numerical_column = ['age', 'fever']
            ordinal_column = ['cough']
            onehot_columns = ['gender', 'city']

            # For fill Numerical data
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='mean'))
                ]
            )

            # For Ordinal categorical data
            ordinal_pipeline = Pipeline(
                steps=[(
                        "ordinal_encoder",
                        OrdinalEncoder(categories=[['Mild', 'Strong']])
                    )])

            # gender, city --> One Hot Encoding
            onehot_pipeline = Pipeline(steps=[("onehot_encoder", OneHotEncoder(handle_unknown='ignore'))])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_column),
                    ('ordinal_pipeline', ordinal_pipeline, ordinal_column),
                    ('onehot_pipeline', onehot_pipeline, onehot_columns)
                ])
            return preprocessor
            logger.info("Preprocessor object created successfully")


        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str
    ) -> tuple:

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Train and test data loaded for transformation")

            target_column = "has_covid"

            input_feature_train_df = train_df.drop(
                columns=[target_column]
            )

            target_feature_train_df = train_df[target_column].map(
                {"Yes": 1, "No": 0}
            )

            input_feature_test_df = test_df.drop(
                columns=[target_column]
            )

            target_feature_test_df = test_df[target_column].map(
                {"Yes": 1, "No": 0}
            )

            preprocessor_obj = self.get_data_transformer_object()

            logger.info("Applying preprocessing on train and test data")

            input_feature_train_arr = preprocessor_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessor_obj.transform(
                input_feature_test_df
            )

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logger.info(
                "Preprocessor object saved as preprocessor.pkl"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()

    obj.initiate_data_transformation(
        'artifacts/train.csv',
        'artifacts/test.csv'
    )