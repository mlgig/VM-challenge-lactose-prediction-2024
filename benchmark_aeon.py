import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from aeon.transformations.collection.convolution_based import Rocket, MiniRocket, MiniRocketMultivariate, MultiRocket, MultiRocketMultivariate
from aeon.regression.deep_learning import FCNRegressor, CNNRegressor, InceptionTimeRegressor, ResNetRegressor, TapNetRegressor
from aeon.regression.convolution_based import RocketRegressor
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
from aeon.regression.feature_based import Catch22Regressor, FreshPRINCERegressor, SummaryRegressor, TSFreshRegressor
from aeon.regression.interval_based import RandomIntervalSpectralEnsembleRegressor, RandomIntervalRegressor, TimeSeriesForestRegressor,CanonicalIntervalForestRegressor,DrCIFRegressor,IntervalForestRegressor
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error, r2_score


def load_raw_data():
    raw_data = pd.read_excel('../data/train_dataset.xlsx', sheet_name=None)

    y = pd.read_excel('../data/reference_values.xlsx', 0, engine='openpyxl')['lactose content'].values

    return raw_data, y


def convert_data(raw_data, output_type='multi', uni_dimension = '3d'):
    uni_output = np.zeros((len(raw_data),1, 3424))
    if output_type == 'multi':
        train_X_multi = np.zeros((len(raw_data),300, 3424))
        for i,v in enumerate(raw_data.values()):        
            train_X_multi[i,:v.values.shape[0],:] = v.values
        return train_X_multi
    if output_type == 'uni_mean':
        #train_X_uni_mean = np.zeros((len(raw_data),1, 3424))
        for i,v in enumerate(raw_data.values()):        
            uni_output[i,0,:] = np.mean(v.values, axis = 0)        
        #return train_X_uni_mean
    if output_type == 'uni_med':
        #train_X_uni_med = np.zeros((len(raw_data),1, 3424))
        for i,v in enumerate(raw_data.values()):        
            uni_output[i,0,:] = np.median(v.values, axis = 0)
        #return train_X_uni_med
    if output_type.startswith('uni_p'):
        q = int(output_type.split('_')[-1])
        #train_X_uni_percentile = np.zeros((len(raw_data),1, 3424))
        for i,v in enumerate(raw_data.values()):        
            uni_output[i,0,:] = np.percentile(v.values, q = q, axis = 0)
        #return train_X_uni_percentile
    if output_type == 'uni_flatten':
        uni_output = np.zeros((len(raw_data),1, 300*3424))
        for i,v in enumerate(raw_data.values()):        
            unpadded = v.values.flatten()
            uni_output[i,0,:len(unpadded)] = unpadded
        #return train_X_uni_flatten
    
    if uni_dimension == '3d':
        return uni_output
    elif uni_dimension == '2d':
        return uni_output[:,0,:]
    

def single_split_exp(raw_data, y, models, data_type='multi', uni_dimension = '3d'):
    
    rmse = []
    r2 = []    
    X = convert_data(raw_data, output_type=data_type, uni_dimension=uni_dimension)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    for k,m in models.items():
        print(f"Train and test {k} model on {data_type} data type:")
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        rmse.append(math.sqrt(mean_squared_error(y_test, y_pred)))
        r2.append(r2_score(y_test, y_pred))
        print(f"RMSE: {rmse[-1]}, R2: {r2[-1]}")

    df = pd.DataFrame({'dtype':data_type,'model': models.keys(), 'rmse': rmse, 'r2': r2})
    df.to_csv('results/single_split_results.csv', index=False, mode='a', header=False)
   
def prevalidate_predict(X, y, rgr, cv=4, random_state = 42):

    
    skf = StratifiedKFold(n_splits=cv, random_state = random_state, shuffle=True)    

    
    y_preval = np.zeros(len(y))
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rgr.fit(X_train, y_train)
        y_pred = rgr.predict(X_test)        
        y_preval[test_index] = y_pred
        
    return y_preval

def prevalidate_experiment(raw_data, y , models, data_type='multi', uni_dimension = '3d'):
    
    X = convert_data(raw_data, output_type=data_type, uni_dimension=uni_dimension)
    all_preval = np.zeros((len(models), len(y)))
    rmse = []
    r2 = []
    for i,(k,m) in  enumerate(models.items()):
        print(f"Prevalidating {k} model on {data_type} data type:")
        
        y_preval = prevalidate_predict(X, y, m)
        all_preval[i,:] = y_preval

        rmse.append(math.sqrt(mean_squared_error(y, y_preval)))
        r2.append(r2_score(y, y_preval))
        print(f"RMSE: {rmse[-1]}, R2: {r2[-1]}")
        
    pd.DataFrame({'dtype':data_type,'model': models.keys(), 'rmse': rmse, 'r2': r2}).to_csv('results/preval_results.csv', index=False, mode='a', header=False)
    preval_df = pd.concat([pd.DataFrame({'dtype':data_type,'method':models.keys()}) ,pd.DataFrame(all_preval)],axis=1)    
    preval_df.to_csv('results/preval_predictions.csv', index=False, mode='a', header=False)


    
    
def benchmark_aeon(input_type = 'uni_p_75' ):
    
    feature_based_models = {        
        'freshprince':FreshPRINCERegressor(random_state=0), # very expensive
        'catch22':Catch22Regressor(random_state=0),
        'summary':SummaryRegressor(random_state=0),
        'tsfresh':TSFreshRegressor(random_state=0),        
              }
    deep_models = {
        'fcn':FCNRegressor(random_state=0),
        'cnn':CNNRegressor(random_state=0),
        'inception':InceptionTimeRegressor(random_state=0),
        'resnet':ResNetRegressor(random_state=0),
        'tapnet':TapNetRegressor(random_state=0),
    }
    interval_based_models = {
        'cif': CanonicalIntervalForestRegressor(random_state=0),
        'drcif':DrCIFRegressor(random_state=0),
        'iforest':IntervalForestRegressor(random_state=0),
        'rinterval':RandomIntervalRegressor(random_state=0),
        'rise':RandomIntervalSpectralEnsembleRegressor(random_state=0),
        'tsforest':TimeSeriesForestRegressor(random_state=0),
    }
    other_models = {
        'knn':KNeighborsTimeSeriesRegressor(),
        'rocket':RocketRegressor(random_state=0),
        'minirocket':RocketRegressor(rocket_transform='minirocket',random_state=0),
        'multirocket':RocketRegressor(rocket_transform='multirocket',random_state=0),       
        
    }

    tabular_models = {
        'linreg':LinearRegression(),        
        'ridgecv':RidgeCV(alphas=np.logspace(-3, 3, 10)),      
        'pls5':PLSRegression(n_components = 5),
        'pls10':PLSRegression(n_components = 10),
    }
    
    
    raw_data, y = load_raw_data()
    
    prevalidate_experiment(raw_data, y, interval_based_models, data_type = input_type)
    # prevalidate_experiment(raw_data, y, tabular_models, data_type = input_type, uni_dimension = '2d') # sklearn models only accept 2d data
    # single_split_exp(raw_data, y, other_models, data_type = input_type)    
   

    
        
if __name__ == "__main__":
    benchmark_aeon()