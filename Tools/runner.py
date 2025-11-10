# from crewai.tools import tool
# import pandas as pd
# import json
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from pathlib import Path

# @tool("ml_model_trainer")
# def train_ml_model( filename: str,
#     model_name: str,
#     target_variable: str,
#     split: float,
#     flag: int = 0,
#     param: dict = None
# ) -> str:
#     """
#     Trains a machine learning model (SVM, Logistic Regression, or Decision Tree) with optional hyperparameter tuning.
    
#     Args:
        
#         filename: CSV file name in ./Datasets/ directory
#         model_name: "decision_tree", "svm", or "logistic_regression"
#         target_variable: name of the target column
#         split: test size ratio (e.g., 0.2 for 20% test data)
#         flag: 0 for no tuning, 1 for hyperparameter tuning with GridSearchCV (default: 0)
#         param: dict of model parameters, empty {} for defaults (default: None)
    
#     Returns:
#         String containing accuracy, classification report, confusion matrix, and parameters used
#     """
#     try:
#         # # Load parameters from JSON
#         # json_dir = Path('./JSONs/')
#         # with open(json_dir / json_file, 'r') as file:
#         #     parameters = json.load(file)

#         with open('/Users/gunik/scientific_exocortex/JSONs/model_parameters.json','r') as file:
#             parameters = json.load(file)
            

#         model_name = parameters['model_name']
#         df = pd.read_csv("./Datasets/" + str(parameters['filename']))
#         flag = parameters.get('flag', 0)

#         # Check for target_variable
#         target_variable = parameters.get("target_variable", None)
#         if target_variable is None:
#             return "ERROR: Target variable not specified in the parameters."

#         X = df.drop(columns=[target_variable])
#         y = df[target_variable]

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=parameters['split'], random_state=42
#         )

#         # Model configuration
#         model_config = {
#             "decision_tree": {
#                 "param_dict": "default_decision_tree_parameters", 
#                 "lib_name": DecisionTreeClassifier
#             },
#             "svm": {
#                 "param_dict": "default_svm_parameters", 
#                 "lib_name": SVC
#             },
#             "logistic_regression": {
#                 "param_dict": "default_lr_parameters", 
#                 "lib_name": LogisticRegression
#             }
#         }

#         if model_name not in model_config:
#             return f"ERROR: Unknown model name '{model_name}'. Choose from: {list(model_config.keys())}"

#         params = model_config[model_name]["param_dict"]
#         param = model_param[params]
#         ModelClass = model_config[model_name]["lib_name"]
        
#         # NO hyperparameter tuning
#         if flag == 0:
#             # Merge default and user-provided parameters
#             merged_parameters = {**param, **parameters.get("param", {})}

#             # Initialize and train model
#             model = ModelClass(**merged_parameters)
#             model.fit(X_train, y_train)
            
#             # Make predictions
#             y_pred = model.predict(X_test)
            
#             acc = accuracy_score(y_test, y_pred)
#             cr = classification_report(y_test, y_pred)
#             cm = confusion_matrix(y_test, y_pred)
            
#             result = f"""
# Model Training Results (No Hyperparameter Tuning):
# Model: {model_name}
# Accuracy: {acc:.4f}
# Parameters Used: {parameters.get("param", {})}

# Classification Report:
# {cr}

# Confusion Matrix:
# {cm}
# """
#             return result
        
#         # Hyperparameter tuning with GridSearchCV
#         else:
#             param_grid = parameters.get("param", {})
#             grid_search = GridSearchCV(
#                 ModelClass(), 
#                 param_grid, 
#                 cv=2, 
#                 scoring='accuracy'
#             )

#             # Fit grid search
#             grid_search.fit(X_train, y_train)

#             # Get best model
#             best_params = grid_search.best_params_
#             final_model = grid_search.best_estimator_
            
#             # Make predictions
#             y_pred = final_model.predict(X_test)

#             acc = accuracy_score(y_test, y_pred)
#             cr = classification_report(y_test, y_pred)
#             cm = confusion_matrix(y_test, y_pred)

#             result = f"""
# Model Training Results (With Hyperparameter Tuning):
# Model: {model_name}
# Accuracy: {acc:.4f}
# Best Parameters: {best_params}

# Classification Report:
# {cr}

# Confusion Matrix:
# {cm}
# """
#             return result
            
#     except FileNotFoundError as e:
#         return f"ERROR: File not found - {str(e)}"
#     except KeyError as e:
#         return f"ERROR: Missing required parameter - {str(e)}"
#     except Exception as e:
#         return f"ERROR: {str(e)}"


from crewai.tools import tool
import pandas as pd
import json
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path

# Fixed location for model default parameters
# MODEL_DEFAULTS_FILE = Path(__file__).resolve().parent / "model_defaults.json"

@tool("ml_model_trainer")
def train_ml_model(
    filename: str,
    model_name: str,
    target_variable: str,
    split: float,
    flag: int = 0,
    param: dict = None
) -> dict:
    """
    Trains a machine learning model (SVM, Logistic Regression, or Decision Tree) with optional hyperparameter tuning.
    
    Args:
        filename: CSV file name in ./Datasets/ directory
        model_name: "decision_tree", "svm", or "logistic_regression"
        target_variable: name of the target column
        split: test size ratio (e.g., 0.2 for 20% test data)
        flag: 0 for no tuning, 1 for hyperparameter tuning with GridSearchCV (default: 0)
        param: dict of model parameters, empty {} for defaults (default: None)
    
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - accuracy: float (if success)
            - classification_report: string (if success)
            - confusion_matrix: list of lists (if success)
            - parameters_used: dict (if success)
            - model_name: string (if success)
            - dataset: string (if success)
            - error: string (if error)
    """
    try:
        # Handle None param
        if param is None:
            param = {}
            
        # Load model defaults from fixed location
        with open('/Users/gunik/scientific_exocortex/JSONs/model_parameters.json', 'r') as file:
            model_param = json.load(file)

        df = pd.read_csv("./Datasets/" + str(filename))

        X = df.drop(columns=[target_variable])
        y = df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split, random_state=42
        )

        # Model configuration
        model_config = {
            "decision_tree": {
                "param_dict": "default_decision_tree_parameters", 
                "lib_name": DecisionTreeClassifier
            },
            "svm": {
                "param_dict": "default_svm_parameters", 
                "lib_name": SVC
            },
            "logistic_regression": {
                "param_dict": "default_lr_parameters", 
                "lib_name": LogisticRegression
            }
        }

        if model_name not in model_config:
            return {
                "status": "error",
                "error": f"Unknown model name '{model_name}'. Choose from: {list(model_config.keys())}"
            }

        param_key = model_config[model_name]["param_dict"]
        default_params = model_param[param_key]
        ModelClass = model_config[model_name]["lib_name"]
        
        # NO hyperparameter tuning
        if flag == 0:
            # Merge default and user-provided parameters
            merged_parameters = {**default_params, **param}

            # Initialize and train model
            model = ModelClass(**merged_parameters)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            return {
                "status": "success",
                "model_name": model_name,
                "dataset": filename,
                "target_variable": target_variable,
                "train_test_split": f"{(1-split)*100:.0f}% / {split*100:.0f}%",
                "hyperparameter_tuning": False,
                "accuracy": float(acc),
                "classification_report": cr,
                "confusion_matrix": cm.tolist(),
                "parameters_used": param or default_params
            }
        
        # Hyperparameter tuning with GridSearchCV
        else:
            param_grid = param
            if not param_grid:
                return {
                    "status": "error",
                    "error": "flag=1 requires param_grid in 'param' field for hyperparameter tuning"
                }
                
            grid_search = GridSearchCV(
                ModelClass(), 
                param_grid, 
                cv=2, 
                scoring='accuracy'
            )

            # Fit grid search
            grid_search.fit(X_train, y_train)

            # Get best model
            best_params = grid_search.best_params_
            final_model = grid_search.best_estimator_
            
            # Make predictions
            y_pred = final_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            return {
                "status": "success",
                "model_name": model_name,
                "dataset": filename,
                "target_variable": target_variable,
                "train_test_split": f"{(1-split)*100:.0f}% / {split*100:.0f}%",
                "hyperparameter_tuning": True,
                "accuracy": float(acc),
                "classification_report": cr,
                "confusion_matrix": cm.tolist(),
                "best_parameters": best_params,
                "parameter_grid_searched": param_grid
            }
            
    except FileNotFoundError as e:
        return {
            "status": "error",
            "error": f"File not found - {str(e)}. Make sure model_defaults.json exists and dataset exists in ./Datasets/"
        }
    except KeyError as e:
        return {
            "status": "error",
            "error": f"Missing required parameter or column - {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }