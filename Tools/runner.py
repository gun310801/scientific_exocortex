from crewai.tools import tool
import pandas as pd
import json
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path

@tool("ml_model_trainer")
def train_ml_model(json_file: str) -> str:
    """
    Trains a machine learning model (SVM, Logistic Regression, or Decision Tree) with optional hyperparameter tuning.
    
    Args:
        
        model_file: Name of the JSON file containing model default parameters (in ./JSONs/ directory)
    
    Returns:
        String containing accuracy, classification report, confusion matrix, and parameters used
    """
    try:
        # Load parameters from JSON
        json_dir = Path('./JSONs/')
        with open(json_dir / json_file, 'r') as file:
            parameters = json.load(file)
            
        with open(json_dir / model_file, 'r') as file:
            model_param = json.load(file)

        model_name = parameters['model_name']
        df = pd.read_csv("./Datasets/" + str(parameters['filename']))
        flag = parameters.get('flag', 0)

        # Check for target_variable
        target_variable = parameters.get("target_variable", None)
        if target_variable is None:
            return "ERROR: Target variable not specified in the parameters."

        X = df.drop(columns=[target_variable])
        y = df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=parameters['split'], random_state=42
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
            return f"ERROR: Unknown model name '{model_name}'. Choose from: {list(model_config.keys())}"

        params = model_config[model_name]["param_dict"]
        param = model_param[params]
        ModelClass = model_config[model_name]["lib_name"]
        
        # NO hyperparameter tuning
        if flag == 0:
            # Merge default and user-provided parameters
            merged_parameters = {**param, **parameters.get("param", {})}

            # Initialize and train model
            model = ModelClass(**merged_parameters)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            result = f"""
Model Training Results (No Hyperparameter Tuning):
Model: {model_name}
Accuracy: {acc:.4f}
Parameters Used: {parameters.get("param", {})}

Classification Report:
{cr}

Confusion Matrix:
{cm}
"""
            return result
        
        # Hyperparameter tuning with GridSearchCV
        else:
            param_grid = parameters.get("param", {})
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

            result = f"""
Model Training Results (With Hyperparameter Tuning):
Model: {model_name}
Accuracy: {acc:.4f}
Best Parameters: {best_params}

Classification Report:
{cr}

Confusion Matrix:
{cm}
"""
            return result
            
    except FileNotFoundError as e:
        return f"ERROR: File not found - {str(e)}"
    except KeyError as e:
        return f"ERROR: Missing required parameter - {str(e)}"
    except Exception as e:
        return f"ERROR: {str(e)}"