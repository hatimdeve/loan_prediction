import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
               "LogisticRegression": LogisticRegression(),
               "DecisionTreeClassifier": DecisionTreeClassifier(),
               "RandomForestClassifier": RandomForestClassifier(),
               "XGBClassifier": XGBClassifier(),
               "GaussianNB": GaussianNB(),
               "GradientBoostingClassifier": GradientBoostingClassifier(),
               "LGBMClassifier": LGBMClassifier(),
               "KNeighborsClassifier": KNeighborsClassifier(),
               "SVC": SVC(),
            }
            param_grid = {
                "LogisticRegression": {
                            "penalty": ["l2"],
                            "C": [0.1, 1.0, 10.0]
            },
                "DecisionTreeClassifier": {
                            "max_depth": [None, 5, 10],
                            "min_samples_split": [2, 5, 10]
            },
                "RandomForestClassifier": {
                            "n_estimators": [100, 200, 300],
                            "max_depth": [None, 5, 10]
            },
                "XGBClassifier": {
                            "learning_rate": [0.1, 0.01, 0.001],
                            "max_depth": [3, 5, 10]
            },
                "GaussianNB": {},
                "GradientBoostingClassifier": {
                            "learning_rate": [0.1, 0.01, 0.001],
                            "n_estimators": [100, 200, 300],
                            "max_depth": [3, 5, 10]
            },
                "LGBMClassifier": {
                            "learning_rate": [0.1, 0.01, 0.001],
                            "n_estimators": [100, 200, 300],
                            "max_depth": [3, 5, 10]
            },
                "KNeighborsClassifier": {
                            "n_neighbors": [3, 5, 10]
            },
                "SVC": {
                            "C": [0.1, 1.0, 10.0],
                            "kernel": ["linear", "rbf"]
            }
        }
            

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=param_grid)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
            



            
        except Exception as e:
            raise CustomException(e,sys)