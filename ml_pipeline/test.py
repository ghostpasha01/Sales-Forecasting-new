from sklearn.ensemble import GradientBoostingRegressor
from pygam import LinearGAM, PoissonGAM,GammaGAM
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def predictions(X_test, y_test, model_gbm, gam):
    '''
    The function takes in the final models and predicts the results

    '''
    y_pred_gbm=model_gbm.predict(X_test)
    print("R2 of GBM is")
    print(r2_score(y_test,y_pred_gbm))

    print(" ")

    y_pred_gam=gam.predict(X_test.values)
    print("R2 of GAM is")
    print(r2_score(y_test,y_pred_gam))

    return y_pred_gbm, y_pred_gam

