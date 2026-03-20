
import xgboost as xgb
import pandas as pd
import config
import pathlib

def check_importance_classifier(model_path, name):
    print(f"\n--- {name} (Classifier) ---")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    importance = model.get_booster().get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, score) in enumerate(sorted_importance[:20]):
        print(f"#{i+1:2d} {feat:<25} : {score:.4f}")

def check_importance_regressor(model_path, name):
    print(f"\n--- {name} (Regressor) ---")
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    importance = model.get_booster().get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, score) in enumerate(sorted_importance[:20]):
        print(f"#{i+1:2d} {feat:<25} : {score:.4f}")

if __name__ == "__main__":
    check_importance_classifier(config.BRAIN1_MODEL_LONG_PATH, "Brain 1 LONG")
    check_importance_classifier(config.BRAIN1_MODEL_SHORT_PATH, "Brain 1 SHORT")
    check_importance_regressor(config.BRAIN2_MODEL_PATH, "Brain 2 Conviction")
