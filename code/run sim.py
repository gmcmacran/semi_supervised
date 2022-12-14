# %%
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from scipy.stats import uniform, randint 
from sklearn.model_selection import RandomizedSearchCV

##############
# Set working directory
##############
# %%
os.chdir('S:/Python/projects/semi_supervised')

##############
# Define helpers
##############
# %%
def create_data(type, nrow, ncol, seed):
    if type == "classification":
        X, y = make_classification(n_samples = nrow, n_features = ncol, n_informative = ncol, n_redundant = 0, random_state=seed)
    else:
        X, y = make_regression(n_samples = nrow, n_features = ncol, n_informative = ncol, random_state=seed)

    ID = np.arange(0, X.shape[0], 1)

    colnames = ['ID', 'Y'] + ["X" + str(i) for i in np.arange(0, X.shape[1])]
    DF = pd.DataFrame(data = np.concatenate((ID.reshape(ID.shape[0], 1), y.reshape(y.shape[0], 1), X), axis = 1), columns = colnames)

    return DF

def extract_XY(DF, prop, type):
  DF_known = DF.sample(frac = prop)
  DF_unknown = DF.loc[~DF.ID.isin(DF_known.ID)]
  
  X_known = DF_known[["X" + str(i) for i in np.arange(0, DF_known.shape[1] - 2)]].to_numpy()
  Y_known = DF_known[['Y']].to_numpy()
  Y_known = Y_known.reshape((Y_known.shape[0],))


  X_unknown = DF_unknown[["X" + str(i) for i in np.arange(0, DF_known.shape[1] - 2)]].to_numpy()
  Y_unknown = DF_unknown[['Y']].to_numpy()
  Y_unknown = Y_unknown.reshape((Y_unknown.shape[0],))

  if type == "classification":
    Y_known = Y_known.astype('int32')
    Y_unknown = Y_unknown.astype('int32')
  
  return X_known, Y_known, X_unknown, Y_unknown

def standardize(X_known, X_unknown):
    X_both = np.concatenate((X_known, X_unknown), axis = 0)
    scaler = StandardScaler()
    scaler.fit(X_both)
    X_known = scaler.transform(X_known)
    X_unknown = scaler.transform(X_unknown)

    return X_known, X_unknown, scaler

def create_projection_obj(X_known, X_unknown, kernel):
    X_both = np.concatenate((X_known, X_unknown), axis = 0)

    PCA = KernelPCA(n_components = 3, kernel = kernel, remove_zero_eig=True)
    PCA.fit(X = X_both)

    return PCA

def project(X_known, X_unknown, PCA):
    X_known_project = PCA.transform(X_known)
    X_unknown_project = PCA.transform(X_unknown)

    return X_known_project, X_unknown_project

def impute_y(X_known, Y_known, X_unknown, type):
    if type == "classification":
        knn = KNeighborsClassifier(n_neighbors = 5)
    else:
        knn = KNeighborsRegressor(n_neighbors = 5)
    
    knn.fit(X = X_known, y = Y_known)
    Y_impute = knn.predict(X_unknown)

    return Y_impute

def calc_match_metric(Y_unknown, Y_impute, type):
    if type == "classification":
        metric = accuracy_score(Y_unknown, Y_impute)
    else:
        metric = mean_absolute_error(Y_unknown, Y_impute)
    return metric

def train_model(X_train, Y_train, type):
    # run search and pick best fit
    params = {'learning_rate': uniform(.005, .015), 'max_iter': randint(100, 200), 'max_depth': randint(1, 14)}

    if type == "classification":
        model = HistGradientBoostingClassifier(early_stopping=True)
        CV = RandomizedSearchCV(estimator = model, param_distributions = params, scoring = 'roc_auc')
    else:
        model = HistGradientBoostingRegressor(early_stopping=True)
        CV = RandomizedSearchCV(estimator = model, param_distributions = params, scoring = 'neg_median_absolute_error')
    
    CV.fit(X_train, Y_train)
    model = CV.best_estimator_

    return model

def calc_model_metric(model_known, scaler_known, model_both, scaler_unknown, proj, type, nrow, ncol):
 
    # make unseen data
    if type == "classification":
        X_unseen, Y_unseen = make_classification(n_samples = nrow, n_features = ncol, n_informative = ncol, n_redundant = 0)
    else:
        X_unseen, Y_unseen = make_regression(n_samples = nrow, n_features = ncol, n_informative = ncol)

    if type == "classification":
        # known approach
        X_unseen_stanard = scaler_known.transform(X_unseen)
        preds_known = model_known.predict_proba(X_unseen_stanard)[:, 1]

        # semi supervised
        X_unseen = scaler_unknown.transform(X_unseen)
        X_unseen_proj = proj.transform(X_unseen)
        preds_both = model_both.predict_proba(X_unseen_proj)[:, 1]
    
        AUC_known = roc_auc_score(Y_unseen, preds_known)
        AUC_impute = roc_auc_score(Y_unseen, preds_both)

        metric = AUC_impute - AUC_known
    else:
       # known approach
        X_unseen_stanard = scaler_known.transform(X_unseen)
        preds_known = model_known.predict(X_unseen_stanard)

        # semi supervised
        X_unseen = scaler_unknown.transform(X_unseen)
        X_unseen_proj = proj.transform(X_unseen)
        preds_both = model_both.predict(X_unseen_proj)
    
        MAE_known = mean_absolute_error(Y_unseen, preds_known)
        MAE_impute = mean_absolute_error(Y_unseen, preds_both)

        metric = MAE_impute - MAE_known

    return metric

##############
# Run simulation
##############

# %%
np.random.seed(42)
pieces = []
seed = 0
for type in ["classification", "regression"]:
    for prop in np.arange(.05, 1, .05):
        print(prop)
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']:
            for b in np.arange(0, 5):

                seed += 1
                DF = create_data(type, 100000, 10, seed)

                X_known, Y_known, X_unknown, Y_unknown = extract_XY(DF, prop, type)

                # just known data
                scaler_known = StandardScaler()
                scaler_known.fit(X_known)
                X_known = scaler_known.transform(X_known)
 
                model_known = train_model(X_known, Y_known,  type)

                # semi supervised (imputing labels)
                DF = create_data(type, 100000, 10, seed)
                X_known, Y_known, X_unknown, Y_unknown = extract_XY(DF, prop, type)

                X_known, X_unknown, scaler_unknown = standardize(X_known, X_unknown)
                proj = create_projection_obj(X_known, X_unknown, kernel)
                X_known, X_unknown = project(X_known, X_unknown, proj)

                Y_impute = impute_y(X_known, Y_known, X_unknown, type)

                X_both = np.concatenate((X_known, X_unknown), axis = 0)
                Y_both = np.concatenate((Y_known, Y_impute), axis = 0)
                model_both = train_model(X_both, Y_both,  type)

                matchMetric = calc_match_metric(Y_unknown, Y_impute, type)

                modelMetric = calc_model_metric(model_known, scaler_known, model_both, scaler_unknown, proj, type, 20000, 10)

                piece = {'type':[type], 'prop':[prop], 'kernel':[kernel], 'b':[b], 'matchMetric':[matchMetric], 'modelMetric':[modelMetric]}
                piece = pd.DataFrame(piece)
                pieces.append(piece)

result = pd.concat(pieces)
result.head()

# %%
result.to_csv(path_or_buf = 'data/result.csv', index=False)

# %%
