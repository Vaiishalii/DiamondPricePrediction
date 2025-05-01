import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.impute import SimpleImputer #handling missing values
from sklearn.preprocessing import StandardScaler #handling feature scaling
from sklearn.preprocessing import OrdinalEncoder #ordinal encoding
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# model training
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


##Data Transformation Config
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            #define which columns should be ordinal encoded and which should be scaled
            categorical_cols = ['cut','color','clarity']
            numerical_cols = ['carat','depth','table','x','y','z']

            #define custom ranking for each ordinal variable
            cut_categories = ['Fair','Good','Very Good','Ideal','Premium']
            color_categories = ['J','I','H','G','F','E','D']
            clarity_categories = ['I1','I2','I3','SI1','SI2','VS1','VS2','VVS1','VVS2','IF']

            logging.info('Pipeline initiated')

            #numerical columns pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ] 
            )

            #categorical columns pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ] 
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline',num_pipeline,numerical_cols),
                    ('cat_pipeline',cat_pipeline,categorical_cols)
                ]
            )

            return preprocessor

            logging.info('Data Transformation completed')
        
        except Exception as e:
            logging.info('Error occurred during data transformation')
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Data loaded successfully')
            logging.info('Train data shape: {}'.format(train_df.shape))
            logging.info('Test data shape: {}'.format(test_df.shape))
            logging.info('obtaining preprocessor object')

            preprocessor_obj = self.get_data_transformation_object()
            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            ##feature into independent and dependent features

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            ##apply the transformation on the training data
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing datasets')

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
                )
            
            logging.info('Preprocessor pickle is created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                )
        
        except Exception as e:
            logging.info("Exception occured in the initiating datatransformation")

            raise CustomException(e,sys)