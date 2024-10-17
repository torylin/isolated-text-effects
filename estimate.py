import pandas as pd
import os
from utils import get_unnoised_labels
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, ElasticNetCV, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
# from pactools.grid_search import GridSearchCVProgressBar
import argparse
import pdb
from tqdm import tqdm
import random
import numpy as np
import joblib
from utils import get_embeds_clf
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=os.path.join(os.getenv('WORK_DIR'), 'data/causal_text/amazon_synthetic/'))
    parser.add_argument('--csv-path', type=str, default='pr_sampled_label_synthetic_numerical_coef_direct1.00.0.csv')
    parser.add_argument('--true-effects-path', type=str, default='label_synthetic_direct_feature_coefs.csv')
    # parser.add_argument('--treatment', type=str, nargs='+')
    parser.add_argument('--covariates', type=str, default='from_true_effects')
    parser.add_argument('--outcome', type=str, default='label_synthetic')
    parser.add_argument('--seed', type=int, default=240522)
    parser.add_argument('--bootstrap-iters', type=int, default=100)
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--split-type', type=str, default='bootstrap')
    parser.add_argument('--save-results', action='store_true')
    parser.add_argument('--output-csv', type=str)
    parser.add_argument('--estimation-strat', type=int, default=1)
    parser.add_argument('--normalize-weights', type=str)
    # parser.add_argument('--dr-type', type=int, default=2)
    parser.add_argument('--clf', type=str, default='lr')
    parser.add_argument('--g', type=str, default='lr')
    parser.add_argument('--naive', type=str, default='unadjusted')
    parser.add_argument('--true-outcome-model', type=str)
    parser.add_argument('--nu-type', type=str, default='dr')
    parser.add_argument('--num-features', type=int)
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--Cy', type=float, default=1.0)
    parser.add_argument('--Cd', type=float, default=1.0)
    # parser.add_argument('--Cy-upper', type=float, default=0.3)
    # parser.add_argument('--Cd-upper', type=float, default=0.3)
    parser.add_argument('--Cy-upper', type=float, default=0.5)
    parser.add_argument('--Cd-upper', type=float, default=0.5)
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--RV', action='store_true')
    parser.add_argument('--RV0', action='store_true')
    parser.add_argument('--RV-csv', type=str)
    parser.add_argument('--RV-type', type=str)
    parser.add_argument('--RV-interval', type=float)
    parser.add_argument('--no-ood', action='store_true')
    parser.add_argument('--truncate', type=float, default=None)
    parser.add_argument('--plot-weights', action='store_true')
    parser.add_argument('--se-type', type=str)
    parser.add_argument('--text-col', type=str)
    parser.add_argument('--lm-library', type=str, default='sentence-transformers')
    parser.add_argument('--lm-name', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--compute-ovb-grid', action='store_true')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--pred-dir', type=str)
    parser.add_argument('--save-preds', action='store_true')
    parser.add_argument('--drop-feat', type=str)
    args = parser.parse_args()

    return args

def process_data(df, a_name, all_features, outcome):
    # if args.num_features is not None:
    # true_effects = true_effects.iloc[0:min(args.num_features, true_effects.shape[0])]
    y = df[outcome]
    a = np.empty(df.shape[0])
    a[:] = np.nan
    if a_name in df.columns:
        a = df[a_name]
    if args.num_features is not None:
        feature_subset = all_features[all_features != a_name]
        feature_subset = feature_subset[0:min(args.num_features, feature_subset.shape[0])].values
        if a_name in df.columns:
            feature_subset = np.concatenate([feature_subset, [a_name]])
        X = df[feature_subset]
    else:
        X = df[all_features]
    A_c = X.drop(a_name, axis=1)
    if args.covariates == 'text':
        A_c = get_embeds_clf(args.lm_library, args.lm_name, device, A_c[args.text_col].values, args.drop_feat, progress=True)
        X1 = np.concatenate([np.array([1]*A_c.shape[0]).reshape(-1, 1), A_c], axis=1)
        X0 = np.concatenate([np.array([0]*A_c.shape[0]).reshape(-1, 1), A_c], axis=1)
        X = np.concatenate([a.values.reshape(-1, 1), A_c], axis=1)
    else:
        X1 = X.copy()
        X0 = X.copy()
        X1[a_name] = 1
        X0[a_name] = 0

    return (X, X1, X0, A_c, a, y)

def quantile_without_inf(column, q):
    # Filter out inf values
    filtered_column = column[np.isfinite(column)]
    # Compute the quantile on the filtered column
    return np.quantile(filtered_column, q)

def estimate(true_effects, df_nuisance, df_ood, df, iter_idx):
    df_ood_orig = df_ood.copy()
    treatments = true_effects.feature.values
    num_treatments = len(treatments)

    if args.covariates == 'from_true_effects':
        all_features = true_effects.feature
    elif args.covariates == 'from_header':
        drop_cols = [args.outcome]
        # if args.treatment is not None:
        #     if args.treatment is list:
        #         drop_cols += args.treatment
        #     else:
        #         drop_cols += [args.treatment]
        all_features = df.drop(drop_cols, axis=1).columns
    elif args.covariates == 'text':
        all_features = np.append(treatments, [args.text_col])
        # raise Exception('Unstructured text not yet implemented')
    
    ipw_estimates = []
    dr_estimates = []
    outcome_estimates = []
    naive_estimates = []
    sigmasq_estimates = []
    nusq_estimates = []

    # if args.treatment is None:
    #     treatments = all_features
    # else:
    #     treatments = args.treatments
    # num_treatments = len(treatments)
    

    # ipw_1_weights = np.zeros((num_treatments, df.shape[0]))
    # ipw_0_weights = np.zeros((num_treatments, df.shape[0]))
    # ipw_1_weights_ood = np.zeros((num_treatments, df_ood.shape[0]))
    # ipw_0_weights_ood = np.zeros((num_treatments, df_ood.shape[0]))
    # preds_y_1 = np.zeros((num_treatments, df.shape[0]))
    # preds_y_0 = np.zeros((num_treatments, df.shape[0]))
    # preds_y_ood_1 = np.zeros((num_treatments, df_ood.shape[0]))
    # preds_y_ood_0 = np.zeros((num_treatments, df_ood.shape[0]))
    # preds_y_combined = np.zeros((num_treatments, df.shape[0]+df_ood.shape[0]))
    # preds_y = np.zeros((num_treatments, df.shape[0]))
    # preds_y_ood = np.zeros((num_treatments, df_ood.shape[0]))
    # trues_y_1 = np.zeros((num_treatments, df.shape[0]))
    # trues_y_0 = np.zeros((num_treatments, df.shape[0]))
    # trues_y_ood_1 = np.zeros((num_treatments, df_ood.shape[0]))
    # trues_y_ood_0 = np.zeros((num_treatments, df_ood.shape[0]))
    # as_ood = np.zeros((num_treatments, df_ood.shape[0]))
    # indicators_p = np.zeros((num_treatments, df.shape[0] + df_ood.shape[0]))
    # indicators_a = np.zeros((num_treatments, df.shape[0] + df_ood.shape[0]))

    ipw_1_weights = []
    ipw_0_weights = []
    ipw_1_weights_ood = []
    ipw_0_weights_ood = []
    preds_y_1 = []
    preds_y_0 = []
    preds_y_ood_1 = []
    preds_y_ood_0 = []
    preds_y_combined = []
    preds_y = []
    preds_y_ood = []
    trues_y_1 = []
    trues_y_0 = []
    trues_y_ood_1 = []
    trues_y_ood_0 = []
    as_ood = []
    indicators_p = []
    indicators_a = []

    # psi_theta_estimates = np.zeros((num_treatments, df.shape[0]+df_ood.shape[0]))
    # psi_sigmasq_estimates = np.zeros((num_treatments, df.shape[0]+df_ood.shape[0]))
    # psi_nusq_estimates = np.zeros((num_treatments, df.shape[0]+df_ood.shape[0]))
    psi_theta_estimates = []
    psi_sigmasq_estimates = []
    psi_nusq_estimates = []

    num_features_string = ''
    if args.num_features is not None:
        num_features_string = '_{}feat'.format(args.num_features)


    for i in range(num_treatments):
        a_name = treatments[i]
        a_true = true_effects.coef[i]

        if (args.estimation_strat == 1) or (args.estimation_strat == 3):
            df_ood = df_ood_orig[df_ood_orig[a_name] == 1]

        X_nuisance, _, _, A_c_nuisance, a_nuisance, y_nuisance = process_data(df_nuisance, a_name, all_features, args.outcome)
        scaler_a = StandardScaler()
        scaler_x = StandardScaler()
        if args.scale:
            A_c_nuisance = scaler_a.fit_transform(A_c_nuisance)
            X_nuisance = scaler_x.fit_transform(X_nuisance)

        y_vals = list(set(df[args.outcome].values))
        y_vals.sort()

        clf_model_path = os.path.join(args.model_dir, '{}_clf{}_{}{}{}_type{}_{}{}{}{}.pkl'.format(
                a_name, args.clf, rep_name, num_features_string, drop_string, args.estimation_strat, ood_string, norm_string, args.split_type, iter_idx))

        g_model_path = os.path.join(args.model_dir, '{}_g{}_{}{}{}_type{}_{}{}{}{}.pkl'.format(
                    args.outcome, args.g, rep_name, num_features_string, drop_string, args.estimation_strat, ood_string, norm_string, args.split_type, iter_idx))
        
        if os.path.exists(clf_model_path):
            clf = joblib.load(clf_model_path)
        else:
            if args.clf == 'lr':
                param_grid = {'C': [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2],
                    'l1_ratio': [0, .1, .5, .7, .9, .95, .99, 1]}
                clf = LogisticRegression(penalty='elasticnet', max_iter=10000, solver='saga')
                grid_search_clf = GridSearchCV(clf, param_grid, cv=5,
                                    scoring='accuracy', n_jobs=-1, verbose=2)
            elif args.clf == 'svm':
                clf = SVC(probability=True)
            elif args.clf == 'gbm':
                clf = GradientBoostingClassifier(subsample=0.7, random_state=args.seed)
            elif args.clf == 'mlp':
                param_grid = {
                    'hidden_layer_sizes': [(128,), (128,128,), (128,256,128,)],
                    'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4]
                }
                clf = MLPClassifier(early_stopping=True, random_state=args.seed,
                                    solver='adam', validation_fraction=0.3)
                grid_search_clf = GridSearchCV(clf, param_grid, cv=5,
                                        scoring='accuracy', n_jobs=-1, verbose=2)

        if os.path.exists(g_model_path):
            g = joblib.load(g_model_path)
        else:
            if len(y_vals) == 2:
                if args.g == 'lr':
                    # param_grid = [{'penalty': ['l1', 'l2']},
                    #     {'penalty': ['elasticnet'],
                    #     'l1_ratios': [np.array([.1, .5, .7, .9, .95, .99, 1])]}]
                    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2],
                        'l1_ratio': [0, .1, .5, .7, .9, .95, .99, 1]}
                    g = LogisticRegression(max_iter=1000, solver='saga')
                    grid_search_g = GridSearchCV(g, param_grid, cv=5,
                                    scoring='accuracy', n_jobs=-1, verbose=2)
                elif args.g == 'svm':
                    g = SVC(probability=True)
                elif args.g == 'gbm':
                    g = GradientBoostingClassifier(subsample=0.7, random_state=args.seed)
                elif args.g == 'mlp':
                    param_grid = {
                        'hidden_layer_sizes': [(128,), (128,128,), (128,256,128,)],
                        'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4]
                    }
                    g = MLPClassifier(early_stopping=True, random_state=args.seed,
                                    solver='adam', validation_fraction=0.3)
                    grid_search_g = GridSearchCV(g, param_grid, cv=5,
                                        scoring='accuracy', n_jobs=-1, verbose=2)
                y_vec = np.empty((df.shape[0], 2))
                y_vec[:,0] = y_vals[0]
                y_vec[:,1] = y_vals[1]
                y_vec_ood = np.empty((df_ood.shape[0], 2))
                y_vec_ood[:,0] = y_vals[0]
                y_vec_ood[:,1] = y_vals[1]

            else:
                if args.g == 'lr':
                    g = ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
                elif args.g == 'svm':
                    g = SVR()
                elif args.g == 'gbm':
                    g = GradientBoostingRegressor(subsample=0.7, random_state=args.seed)
                elif args.g == 'mlp':
                    param_grid = {
                        'hidden_layer_sizes': [(128,), (128,128,), (128,256,128,)],
                        'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4]
                    }
                    g = MLPRegressor(early_stopping=True, random_state=args.seed,
                                    solver='adam', validation_fraction=0.3)
                    grid_search_g = GridSearchCV(g, param_grid, cv=5,
                                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)


        if not os.path.exists(clf_model_path):
            if (args.clf == 'mlp') or (args.clf == 'lr'):
                # Fit the grid search on the nuisance data
                
                grid_search_clf.fit(A_c_nuisance, a_nuisance)

                # Best model and alpha
                clf = grid_search_clf.best_estimator_
                # best_alpha = grid_search_clf.best_params_['alpha']
                # print(best_alpha)
                print(grid_search_clf.cv_results_['mean_test_score'])
            else:
                clf.fit(A_c_nuisance, a_nuisance)
            
            if args.save_models:
                joblib.dump(clf, clf_model_path)
        
        if not os.path.exists(g_model_path):
            # if i == 0:
            if (args.g == 'mlp') or ((args.g == 'lr') and (len(y_vals) == 2)):
                grid_search_g.fit(X_nuisance, y_nuisance)
                g = grid_search_g.best_estimator_
                # best_alpha = grid_search_g.best_params_['alpha']
                # print(best_alpha)
                print(grid_search_g.cv_results_['mean_test_score'])
            else:
                g.fit(X_nuisance, y_nuisance)
            
            if args.save_models:
                joblib.dump(g, g_model_path)
        
        X, X1, X0, A_c, a, y = process_data(df, a_name, all_features, args.outcome)
        if a_name in df_ood.columns:
            X_ood, X1_ood, X0_ood, A_c_ood, a_ood, y_ood = process_data(df_ood, a_name, all_features, args.outcome)
            if args.scale:
                A_c_ood_scaled = scaler_a.transform(A_c_ood)
        else:
            X_ood, X1_ood, X0_ood, A_c_ood, _, y_ood = process_data(df_ood, a_name, all_features, args.outcome)
            if args.scale:
                A_c_ood_scaled = scaler_a.transform(A_c_ood)
                a_ood = clf.predict(A_c_ood_scaled)
            else:
                a_ood = clf.predict(A_c_ood)
            if args.covariates == 'text':
                X_ood[:,0] = a_ood
            else:
                X_ood[a_name] = a_ood

        if args.scale:
            X1_scaled = scaler_x.transform(X1)
            X0_scaled = scaler_x.transform(X0)
            X_scaled = scaler_x.transform(X)
            A_c_scaled = scaler_a.transform(A_c)
            X1_ood_scaled = scaler_x.transform(X1_ood)
            X0_ood_scaled = scaler_x.transform(X0_ood)
            X_ood_scaled = scaler_x.transform(X_ood)
        indicator_a = np.concatenate([a, a_ood])

        n1 = (a==1).sum()
        n0 = (a==0).sum()
        n_ood = df_ood.shape[0]
        n1_ood = (a_ood == 1).sum()
        n0_ood = (a_ood == 0).sum()
        corp_prob = (X1.shape[0]/(X1.shape[0]+X1_ood.shape[0]), 
                     X1_ood.shape[0]/(X1.shape[0]+X1_ood.shape[0]))
        treat_prob = (n1/(n1+n0), n0/(n1+n0))
        indicator_p = np.array([1]*X1.shape[0] + [0]*X1_ood.shape[0])
        y_combined = np.concatenate([df[args.outcome].values, np.zeros(X1_ood.shape[0])])

        # indicators_p[i] = indicator_p
        # indicators_a[i] = indicator_a
        indicators_p.append(indicator_p)
        indicators_a.append(indicator_a)
        # Fitting treatment classifier
        
        # if j == 0:
            # print('{} majority vote test accuracy: {:.3f}'.format(a_name, max(np.mean(a), 1-np.mean(a))))
            # print('{} classifier test accuracy: {:.3f}'.format(a_name, clf.score(A_c, a)))

        if args.scale:
            prob_a_cond = clf.predict_proba(A_c_scaled)
        else:
            prob_a_cond = clf.predict_proba(A_c)
        prob_a_marg = (a==1).mean()
        if prob_a_marg == 1:
            prob_a_marg -= 1e-8
            # print('Treatment {} probability = 1'.format(a_name))
        elif prob_a_marg == 0:
            prob_a_marg += 1e-8
            # print('Treatment {} probability = 0'.format(a_name))

        if args.scale:
            prob_a_cond_ood = clf.predict_proba(A_c_ood_scaled)
        else:
            prob_a_cond_ood = clf.predict_proba(A_c_ood)
        # if a_name in df_ood.columns:
        prob_a_marg_ood = (a_ood==1).mean()
        if prob_a_marg_ood == 1:
            prob_a_marg_ood -= 1e-8
            # print('Treatment {} OOD probability = 1'.format(a_name))
        elif prob_a_marg_ood == 0:
            prob_a_marg_ood += 1e-8
            # print('Treatment {} OOD probability = 0'.format(a_name))
        # else:
        #     prob_a_marg_ood = prob_a_marg

        # Fitting outcome model
        
        # if i == 0:
        # if j == 0:
            # print('outcome test R2: {:.3f}'.format(g.score(X, y)))
        # if i == 0:
        if len(set(df[args.outcome].values)) == 2:
            if args.scale:
                prob_y_ood_1 = g.predict_proba(X1_ood_scaled)
                prob_y_ood_0 = g.predict_proba(X0_ood_scaled)
                prob_y_ood = g.predict_proba(X_ood_scaled)
                # if i == 0:
                prob_y_1 = g.predict_proba(X1_scaled)
                prob_y_0 = g.predict_proba(X0_scaled)
                prob_y = g.predict_proba(X_scaled)
            else:
                prob_y_ood_1 = g.predict_proba(X1_ood)
                prob_y_ood_0 = g.predict_proba(X0_ood)
                prob_y_ood = g.predict_proba(X_ood)
                # if i == 0:
                prob_y_1 = g.predict_proba(X1)
                prob_y_0 = g.predict_proba(X0)
                prob_y = g.predict_proba(X)
            pred_y_ood_1 = prob_y_ood_1[:,1]
            pred_y_ood_0 = prob_y_ood_0[:,1]
            pred_y_ood = prob_y_ood[:,1]
            # if i == 0:
            pred_y_1 = prob_y_1[:,1]
            pred_y_0 = prob_y_0[:,1]
            pred_y = prob_y[:,1]
            # pred_y_ood_1 = prob_y_ood_1.max(axis=1)*y_vec_ood[np.arange(len(y_vec_ood)), prob_y_ood_1.argmax(axis=1)]
            # pred_y_ood_0 = prob_y_ood_0.max(axis=1)*y_vec_ood[np.arange(len(y_vec_ood)), prob_y_ood_0.argmax(axis=1)]
            # pred_y_ood = prob_y_ood.max(axis=1)*y_vec_ood[np.arange(len(y_vec_ood)), prob_y_ood.argmax(axis=1)]
            # pred_y_1 = prob_y_1.max(axis=1)*y_vec[np.arange(len(y_vec)), prob_y_1.argmax(axis=1)]
            # pred_y_0 = prob_y_0.max(axis=1)*y_vec[np.arange(len(y_vec)), prob_y_0.argmax(axis=1)]
            # pred_y = prob_y.max(axis=1)*y_vec[np.arange(len(y_vec)), prob_y.argmax(axis=1)]
        else:
            if args.scale:
                pred_y_ood_1 = g.predict(X1_ood_scaled)
                pred_y_ood_0 = g.predict(X0_ood_scaled)
                # if i == 0:
                pred_y_1 = g.predict(X1_scaled)
                pred_y_0 = g.predict(X0_scaled)
                pred_y = g.predict(X_scaled)
                pred_y_ood = g.predict(X_ood_scaled)
            else:
                pred_y_ood_1 = g.predict(X1_ood)
                pred_y_ood_0 = g.predict(X0_ood)
                # if i == 0:
                pred_y_1 = g.predict(X1)
                pred_y_0 = g.predict(X0)
                pred_y = g.predict(X)
                pred_y_ood = g.predict(X_ood)

        pred_y_1_combined = np.concatenate([pred_y_1, pred_y_ood_1])
        pred_y_0_combined = np.concatenate([pred_y_0, pred_y_ood_0])
        pred_y_combined = np.concatenate([pred_y, pred_y_ood])

        # Naive estimate
        if args.naive == 'unadjusted':
            mu1_naive = ((df[a_name]==1)*df[args.outcome]).sum()/n1
            mu0_naive = ((df[a_name]==0)*df[args.outcome]).sum()/n0
        elif args.naive == 'true_outcome':

            true_g = joblib.load(args.true_outcome_model)
            if len(set(df[args.outcome].values)) == 2:

                true_prob_y_ood_1 = true_g.predict_proba(X1_ood)
                true_prob_y_ood_0 = true_g.predict_proba(X0_ood)
                true_y_ood_1 = true_prob_y_ood_1[:,1]
                true_y_ood_0 = true_prob_y_ood_0[:,1]
                # if i == 0:
                true_prob_y_1 = true_g.predict_proba(X1)
                true_prob_y_0 = true_g.predict_proba(X0)
                true_y_1 = true_prob_y_1[:,1]
                true_y_0 = true_prob_y_0[:,1]
                # true_y_ood_1 = true_prob_y_ood_1.max(axis=1)*y_vec_ood[np.arange(len(y_vec_ood)), true_prob_y_ood_1.argmax(axis=1)]
                # true_y_ood_0 = true_prob_y_ood_0.max(axis=1)*y_vec_ood[np.arange(len(y_vec_ood)), true_prob_y_ood_0.argmax(axis=1)]
                # true_y_1 = true_prob_y_1.max(axis=1)*y_vec[np.arange(len(y_vec)), true_prob_y_1.argmax(axis=1)]
                # true_y_0 = true_prob_y_0.max(axis=1)*y_vec[np.arange(len(y_vec)), true_prob_y_0.argmax(axis=1)]
            else:
                
                # if i == 0:
                true_y_1 = true_g.predict(X1)
                true_y_0 = true_g.predict(X0)
                true_y_ood_1 = true_g.predict(X1_ood)
                true_y_ood_0 = true_g.predict(X0_ood)

            mu1_naive = np.nansum(list(true_y_1) + list(true_y_ood_1))/len(list(true_y_1) + list(true_y_ood_1))
            mu0_naive = np.nansum(list(true_y_0) + list(true_y_ood_0))/len(list(true_y_0) + list(true_y_ood_0))

        ate_naive = mu1_naive - mu0_naive
        naive_estimates.append(ate_naive)

        if args.estimation_strat == 1:
        # (1) P*(a^c(X)) = P(a^c(X)|a(X)=1)
            # ipw_1_weight = np.ones(df.shape[0])
            # ipw_0_weight = prob_a_cond[:,1]*(1-prob_a_marg)/(prob_a_cond[:,0]*prob_a_marg)
            # ipw_1_weight_ood = np.ones(df_ood.shape[0])
            # ipw_0_weight_ood = prob_a_cond_ood[:,1]*(1-prob_a_marg_ood)/(prob_a_cond_ood[:,0]*prob_a_marg_ood)
            ipw_1_weight = np.ones(df.shape[0])/prob_a_marg
            ipw_0_weight = prob_a_cond[:,1]/(prob_a_cond[:,0]*prob_a_marg)
            ipw_1_weight_ood = np.ones(df_ood.shape[0])/prob_a_marg_ood
            ipw_0_weight_ood = prob_a_cond_ood[:,1]/(prob_a_cond_ood[:,0]*prob_a_marg_ood)

        elif args.estimation_strat == 2:
        # (2) P*(a^c(X)) = P(a^c(X))
            # ipw_1_weight = prob_a_marg/prob_a_cond[:,1]
            # ipw_0_weight = (1-prob_a_marg)/prob_a_cond[:,0]
            # ipw_1_weight_ood = prob_a_marg_ood/prob_a_cond_ood[:,1]
            # ipw_0_weight_ood = (1-prob_a_marg_ood)/prob_a_cond_ood[:,0]
            ipw_1_weight = 1/prob_a_cond[:,1]
            ipw_0_weight = 1/prob_a_cond[:,0]
            ipw_1_weight_ood = 1/prob_a_cond_ood[:,1]
            ipw_0_weight_ood = 1/prob_a_cond_ood[:,0]

        elif args.estimation_strat == 3:
        # (3) General formulation
            df_Pstar_nuisance, df_nuisance_temp = train_test_split(df_nuisance, test_size=0.5, random_state=args.seed)
            df_Pstar_nuisance = df_Pstar_nuisance[df_Pstar_nuisance[a_name] == 1]
            df_Pstar_nuisance['C'] = 1
            df_nuisance_temp['C'] = 0
            df_nuisance_combined = pd.concat([df_Pstar_nuisance, df_nuisance_temp], axis=0)

            _, _, _, A_c_nuisance_combined, _, _ = process_data(df_nuisance_combined, a_name, all_features, args.outcome)
            if args.scale:
                A_c_nuisance_combined = scaler_a.transform(A_c_nuisance_combined)

            # if args.num_features is not None:
            #     feature_subset = all_features[all_features != a_name]
            #     feature_subset = feature_subset[0:min(args.num_features, feature_subset.shape[0])].values
            #     feature_subset = np.concatenate([feature_subset, [a_name]])
            #     X_nuisance_combined = df_nuisance_combined[feature_subset]
            # else:
            #     X_nuisance_combined = df_nuisance_combined[all_features]
            # A_c_nuisance_combined = X_nuisance_combined.drop(a_name, axis=1)
            
            clf_C_model_path = os.path.join(args.model_dir, '{}_clfC_{}{}{}_type{}_{}{}{}{}.pkl'.format(
                    a_name, rep_name, num_features_string, drop_string, args.estimation_strat, ood_string, norm_string, args.split_type, iter_idx))
            if os.path.exists(clf_C_model_path):
                clf_C = joblib.load(clf_C_model_path)
            else: 
                param_grid = {'C': [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2],
                    'l1_ratio': [0, .1, .5, .7, .9, .95, .99, 1]}
                clf_C = LogisticRegression(penalty='elasticnet', max_iter=10000, solver='saga')
                grid_search_clf_C = GridSearchCV(clf_C, param_grid, cv=5,
                            scoring='accuracy', n_jobs=-1, verbose=2)
                grid_search_clf_C.fit(A_c_nuisance_combined, df_nuisance_combined['C'].values)
                clf_C = grid_search_clf_C.best_estimator_

                if args.save_models:
                    joblib.dump(clf_C, clf_C_model_path)

            if args.scale:
                prob_C_cond = clf_C.predict_proba(A_c_scaled)
            else:
                prob_C_cond = clf_C.predict_proba(A_c)
            # prob_C_marg = (df[a_name]==1).mean()
            prob_C_marg = (df_nuisance_combined['C']==1).mean()
            if prob_C_marg == 1:
                prob_C_marg -= 1e-8
                # print('Treatment {} probability = 1'.format(a_name))
            elif prob_C_marg == 0:
                prob_C_marg += 1e-8
                # print('Treatment {} probability = 0'.format(a_name))

            if args.scale:
                prob_C_cond_ood = clf_C.predict_proba(A_c_ood_scaled)
            else:
                prob_C_cond_ood = clf_C.predict_proba(A_c_ood)
            # prob_C_marg_ood = (df_ood[a_name]==1).mean()
            # if prob_C_marg_ood == 1:
            #     prob_C_marg_ood -= 1e-8
            #     print('Treatment {} OOD probability = 1'.format(a_name))
            # elif prob_C_marg_ood == 0:
            #     prob_C_marg_ood += 1e-8
            #     print('Treatment {} OOD probability = 0'.format(a_name))

            # ipw_1_weight = (1-prob_C_marg)*prob_C_cond[:,1]*prob_a_marg/(prob_C_marg*prob_C_cond[:,0]*prob_a_cond[:,1])
            # ipw_0_weight = (1-prob_C_marg)*prob_C_cond[:,1]*(1-prob_a_marg)/(prob_C_marg*prob_C_cond[:,0]*prob_a_cond[:,0])
            # ipw_1_weight_ood = (1-prob_C_marg_ood)*prob_C_cond_ood[:,1]*prob_a_marg_ood/(prob_C_marg_ood*prob_C_cond_ood[:,0]*prob_a_cond_ood[:,1])
            # ipw_0_weight_ood = (1-prob_C_marg_ood)*prob_C_cond_ood[:,1]*(1-prob_a_marg_ood)/(prob_C_marg_ood*prob_C_cond_ood[:,0]*prob_a_cond_ood[:,0])
            ipw_1_weight = (1-prob_C_marg)*prob_C_cond[:,1]/(prob_C_marg*prob_C_cond[:,0]*prob_a_cond[:,1])
            ipw_0_weight = (1-prob_C_marg)*prob_C_cond[:,1]/(prob_C_marg*prob_C_cond[:,0]*prob_a_cond[:,0])
            ipw_1_weight_ood = (1-prob_C_marg)*prob_C_cond_ood[:,1]/(prob_C_marg*prob_C_cond_ood[:,0]*prob_a_cond_ood[:,1])
            ipw_0_weight_ood = (1-prob_C_marg)*prob_C_cond_ood[:,1]/(prob_C_marg*prob_C_cond_ood[:,0]*prob_a_cond_ood[:,0])

        # Fixing inf values
        # try:

        ipw_1_weight[np.isinf(ipw_1_weight)] = max(ipw_1_weight[np.isfinite(ipw_1_weight)])
        ipw_1_weight_ood[np.isinf(ipw_1_weight_ood)] = max(ipw_1_weight_ood[np.isfinite(ipw_1_weight_ood)])
        ipw_0_weight[np.isinf(ipw_0_weight)] = max(ipw_0_weight[np.isfinite(ipw_0_weight)])
        ipw_0_weight_ood[np.isinf(ipw_0_weight_ood)] = max(ipw_0_weight_ood[np.isfinite(ipw_0_weight_ood)])

        if args.truncate is not None:
            lower_bound_1 = np.nanquantile(ipw_1_weight, args.truncate)
            upper_bound_1 = np.nanquantile(ipw_1_weight, 1-args.truncate)
            ipw_1_weight[ipw_1_weight < lower_bound_1] = lower_bound_1
            ipw_1_weight[ipw_1_weight > upper_bound_1] = upper_bound_1
            
            lower_bound_0 = np.nanquantile(ipw_0_weight, args.truncate)
            upper_bound_0 = np.nanquantile(ipw_0_weight, 1-args.truncate)
            ipw_0_weight[ipw_0_weight < lower_bound_0] = lower_bound_0
            ipw_0_weight[ipw_0_weight > upper_bound_0] = upper_bound_0

            lower_bound_1_ood = np.nanquantile(ipw_1_weight_ood, args.truncate)
            upper_bound_1_ood = np.nanquantile(ipw_1_weight_ood, 1-args.truncate)
            ipw_1_weight_ood[ipw_1_weight_ood < lower_bound_1_ood] = lower_bound_1_ood
            ipw_1_weight_ood[ipw_1_weight_ood > upper_bound_1_ood] = upper_bound_1_ood

            lower_bound_0_ood = np.nanquantile(ipw_0_weight_ood, args.truncate)
            upper_bound_0_ood = np.nanquantile(ipw_0_weight_ood, 1-args.truncate)
            ipw_0_weight_ood[ipw_0_weight_ood < lower_bound_0_ood] = lower_bound_0_ood
            ipw_0_weight_ood[ipw_0_weight_ood > upper_bound_0_ood] = upper_bound_0_ood
        # ipw_1_weight_combined = np.concatenate([ipw_1_weight, np.zeros(X1_ood.shape[0])])
        # ipw_0_weight_combined = np.concatenate([ipw_0_weight, np.zeros(X1_ood.shape[0])])
        ipw_1_weight_combined = np.concatenate([ipw_1_weight, ipw_1_weight_ood])
        ipw_0_weight_combined = np.concatenate([ipw_0_weight, ipw_0_weight_ood])
        # dr_norm_1 = np.nansum(ipw_1_weight_combined)/len(ipw_1_weight_combined)
        # dr_norm_0 = np.nansum(ipw_0_weight_combined)/len(ipw_0_weight_combined)
        # except:
        #     pdb.set_trace()

        # Normalize weights
        if args.normalize_weights == 'norm_total':
            # ipw_1_weight /= np.nansum(ipw_1_weight)/df.shape[0]
            # ipw_1_weight_ood_ipwonly = ipw_1_weight_ood/(np.nansum(ipw_1_weight_ood)/df_ood.shape[0])
            # ipw_0_weight /= np.nansum(ipw_0_weight)/df.shape[0]
            # ipw_0_weight_ood_ipwonly = ipw_0_weight_ood/(np.nansum(ipw_0_weight_ood)/df_ood.shape[0])
            # a_combined = np.concatenate([a, a_ood])
            # ipw_1_weight_combined /= np.nansum(ipw_1_weight_combined*indicator_a)/(df.shape[0]+df_ood.shape[0])
            # ipw_0_weight_combined /= np.nansum(ipw_0_weight_combined*(1-indicator_a))/(df.shape[0]+df_ood.shape[0])
            ipw_1_weight_combined /= np.nansum(ipw_1_weight_combined)/(df.shape[0]+df_ood.shape[0])
            ipw_0_weight_combined /= np.nansum(ipw_0_weight_combined)/(df.shape[0]+df_ood.shape[0])
        elif args.normalize_weights == 'norm_a':
            ipw_1_weight_combined /= np.nansum(ipw_1_weight_combined*indicator_a)/(n1+n1_ood)
            ipw_0_weight_combined /= np.nansum(ipw_0_weight_combined*(1-indicator_a))/(n0+n0_ood)

        # IPW estimate
        # mu1_ipw = np.nanmean(y_combined*ipw_1_weight_combined*indicator_p*indicator_a/(corp_prob[0]*treat_prob[0]))
        # mu0_ipw = np.nanmean(y_combined*ipw_0_weight_combined*indicator_p*(1-indicator_a)/(corp_prob[0]*treat_prob[1]))
        mu1_ipw = np.nanmean(y_combined*ipw_1_weight_combined*indicator_p*indicator_a/(corp_prob[0]))
        mu0_ipw = np.nanmean(y_combined*ipw_0_weight_combined*indicator_p*(1-indicator_a)/(corp_prob[0]))
        # mu1_ipw = np.nansum((df[a_name]==1)*ipw_1_weight*df[args.outcome])/n1
        # mu0_ipw = np.nansum((df[a_name]==0)*ipw_0_weight*df[args.outcome])/n0

        ate_ipw = mu1_ipw - mu0_ipw
        ipw_estimates.append(ate_ipw)

        # if args.dr_type == 1:
        #     # DR1 estimate
        #     mu1_dr = np.nansum((df[a_name]==1)*ipw_1_weight*(df[args.outcome] - pred_y_1))/(df[a_name]==1).sum() + \
        #         np.nansum(ipw_1_weight_ood*pred_y_ood_1)/df_ood.shape[0]
        #     mu0_dr = np.nansum((df[a_name]==0)*ipw_0_weight*(df[args.outcome] - pred_y_0))/(df[a_name]==0).sum() + \
        #         np.nansum(ipw_0_weight_ood*pred_y_ood_0)/df_ood.shape[0]
            
        # elif args.dr_type == 2:
            # DR2 estimate
        # mu1_dr = np.nanmean((y_combined - pred_y_1_combined)*ipw_1_weight_combined*indicator_p*indicator_a/(corp_prob[0]*treat_prob[0]) +
        #                     pred_y_1_combined*(1-indicator_p)/corp_prob[1])
        # mu0_dr = np.nanmean((y_combined - pred_y_0_combined)*ipw_0_weight_combined*indicator_p*(1-indicator_a)/(corp_prob[0]*treat_prob[1]) +
        #                     pred_y_0_combined*(1-indicator_p)/corp_prob[1])
        # if 'amazon' in args.data_dir:
        #     try:
        #         mu1_dr = np.nanmean((y_combined - pred_y_1_combined)*ipw_1_weight_combined*indicator_p*indicator_a/(corp_prob[0]) +
        #                     pred_y_1_combined*(1-indicator_p)/corp_prob[1])
        #     except:
        #         pdb.set_trace()
        mu1_dr = np.nanmean((y_combined - pred_y_1_combined)*ipw_1_weight_combined*indicator_p*indicator_a/(corp_prob[0]) +
                            pred_y_1_combined*(1-indicator_p)/corp_prob[1])
        mu0_dr = np.nanmean((y_combined - pred_y_0_combined)*ipw_0_weight_combined*indicator_p*(1-indicator_a)/(corp_prob[0]) +
                            pred_y_0_combined*(1-indicator_p)/corp_prob[1])
        # mu1_dr = np.nansum((df[a_name]==1)*ipw_1_weight*(df[args.outcome] - pred_y_1))/n1 + \
        #     np.nansum(pred_y_ood_1)/n_ood
        # mu0_dr = np.nansum((df[a_name]==0)*ipw_0_weight*(df[args.outcome] - pred_y_0))/n0 + \
            # np.nansum(pred_y_ood_0)/n_ood

        ate_dr = mu1_dr - mu0_dr
        dr_estimates.append(ate_dr)

        mu1_outcome = np.nanmean(pred_y_1_combined)
        mu0_outcome = np.nanmean(pred_y_0_combined)
        # mu1_outcome = np.nansum(list(pred_y_1) + list(pred_y_ood_1))/len(list(pred_y_1) + list(pred_y_ood_1))
        # mu0_outcome = np.nansum(list(pred_y_0) + list(pred_y_ood_0))/len(list(pred_y_0) + list(pred_y_ood_0))

        ate_outcome = mu1_outcome - mu0_outcome
        outcome_estimates.append(ate_outcome)

        sigmasq = np.nanmean(((y_combined - pred_y_combined)**2)*indicator_p/corp_prob[0]) # TODO: Double check what's happening here, why is it going up w/ more features?
        alpha_s = (ipw_1_weight_combined*indicator_a - ipw_0_weight_combined*(1-indicator_a))
        if args.nu_type == 'plugin':
            nusq = np.nanmean((alpha_s**2)*indicator_p/corp_prob[0])
        elif args.nu_type == 'dr':
            # nusq = np.nanmean(2*(ipw_1_weight_combined*indicator_a - ipw_0_weight_combined*(1-indicator_a))*(1-indicator_p)/corp_prob[1]) - \
            #                   np.nanmean((alpha_s**2)*indicator_p/corp_prob[0])
            nusq = np.nanmean(2*(ipw_1_weight_combined + ipw_0_weight_combined)*(1-indicator_p)/corp_prob[1]) - \
                    np.nanmean((alpha_s**2)*indicator_p/corp_prob[0])

        sigmasq_estimates.append(sigmasq)
        nusq_estimates.append(nusq)

        psi_theta = (y_combined - pred_y_1_combined)*ipw_1_weight_combined*indicator_p*indicator_a/(corp_prob[0]) + \
            pred_y_1_combined*(1-indicator_p)/corp_prob[1] - \
                (y_combined - pred_y_0_combined)*ipw_0_weight_combined*indicator_p*(1-indicator_a)/(corp_prob[0]) + \
                pred_y_0_combined*(1-indicator_p)/corp_prob[1] - ate_dr
        
        psi_sigmasq = (((y_combined - pred_y_combined)**2)/corp_prob[0] - sigmasq)*indicator_p
        if args.nu_type == 'plugin':
            psi_nusq = (alpha_s**2/corp_prob[0] - nusq)*indicator_p # TODO: Confirm that it's necessary to have the indicator and that indicator should be outside the centering
        elif args.nu_type == 'dr':
            # psi_nusq = 2*(ipw_1_weight_combined - ipw_0_weight_combined)*(1-indicator_p)/corp_prob[1] - (alpha_s**2)*indicator_p/corp_prob[0] - nusq
            psi_nusq = 2*(ipw_1_weight_combined + ipw_0_weight_combined)*(1-indicator_p)/corp_prob[1] - (alpha_s**2)*indicator_p/corp_prob[0] - nusq

        # ipw_1_weights[i] = ipw_1_weight
        # ipw_0_weights[i] = ipw_0_weight
        # ipw_1_weights_ood[i] = ipw_1_weight_ood
        # ipw_0_weights_ood[i] = ipw_0_weight_ood
        # as_ood[i] = a_ood
        ipw_1_weights.append(ipw_1_weight)
        ipw_0_weights.append(ipw_0_weight)
        ipw_1_weights_ood.append(ipw_1_weight_ood)
        ipw_0_weights_ood.append(ipw_0_weight_ood)
        as_ood.append(a_ood)
        # ipw_1_weights_combined[i] = ipw_1_weight_combined
        # ipw_0_weights_combined[i] = ipw_0_weight_combined
        # preds_y_1[i] = pred_y_1
        # preds_y_0[i] = pred_y_0
        # preds_y_ood_1[i] = pred_y_ood_1
        # preds_y_ood_0[i] = pred_y_ood_0
        # preds_y_combined[i] = pred_y_combined
        # preds_y[i] = pred_y
        # preds_y_ood[i] = pred_y_ood
        preds_y_1.append(pred_y_1)
        preds_y_0.append(pred_y_0)
        preds_y_ood_1.append(pred_y_ood_1)
        preds_y_ood_0.append(pred_y_ood_0)
        preds_y_combined.append(pred_y_combined)
        preds_y.append(pred_y)
        preds_y_ood.append(pred_y_ood)
        # preds_y_1_combined[i] = pred_y_1_combined
        # preds_y_0_combined[i] = pred_y_0_combined

        # psi_theta_estimates[i] = psi_theta
        # psi_sigmasq_estimates[i] = psi_sigmasq
        # psi_nusq_estimates[i] = psi_nusq
        psi_theta_estimates.append(psi_theta)
        psi_sigmasq_estimates.append(psi_sigmasq)
        psi_nusq_estimates.append(psi_nusq)


        if args.naive == 'true_outcome':
            # trues_y_1[i] = true_y_1
            # trues_y_0[i] = true_y_0
            # trues_y_ood_1[i] = true_y_ood_1
            # trues_y_ood_0[i] = true_y_ood_0
            trues_y_1.append(true_y_1)
            trues_y_0.append(true_y_0)
            trues_y_ood_1.append(true_y_ood_1)
            trues_y_ood_0.append(true_y_ood_0)

    if args.split_type == 'bootstrap':
        return (ipw_estimates, dr_estimates, naive_estimates, outcome_estimates, 
                sigmasq_estimates, nusq_estimates, 
                psi_theta_estimates, psi_sigmasq_estimates, psi_nusq_estimates)

    if args.naive == 'unadjusted':
        return (ipw_1_weights, ipw_0_weights, ipw_1_weights_ood, ipw_0_weights_ood, as_ood,
                preds_y_1, preds_y_0, preds_y_ood_1, preds_y_ood_0, preds_y_combined, preds_y, preds_y_ood,
                indicators_p, indicators_a)
    return (ipw_1_weights, ipw_0_weights, ipw_1_weights_ood, ipw_0_weights_ood, as_ood,
            preds_y_1, preds_y_0, preds_y_ood_1, preds_y_ood_0, preds_y_combined, preds_y, preds_y_ood,
            indicators_p, indicators_a,
            trues_y_1, trues_y_0, trues_y_ood_1, trues_y_ood_0)

def compute_RV(theta_s, theta_t, theta_t_lower, theta_t_upper, psi_theta, sigmasq, nusq, Ssq, psi_nusq, psi_sigmasq):
    
    gap = 100
    Cy = 0
    Cd = 0
    count = 0

    while gap > 0:
        theta_minus = theta_s - np.abs(args.rho)*np.sqrt(np.abs(Ssq))*Cy*Cd
        theta_plus = theta_s + np.abs(args.rho)*np.sqrt(np.abs(Ssq))*Cy*Cd

        psi_minus = psi_theta - np.abs(args.rho)*Cy*Cd/(2*np.sqrt(np.abs(sigmasq*nusq)))*(sigmasq*psi_nusq+nusq*psi_sigmasq)
        psi_plus = psi_theta + np.abs(args.rho)*Cy*Cd/(2*np.sqrt(np.abs(sigmasq*nusq)))*(sigmasq*psi_nusq+nusq*psi_sigmasq)

        psi_minussq = np.nanmean(psi_minus**2)
        psi_plussq = np.nanmean(psi_plus**2)

        ovb_ci_lower = theta_minus - 1.96*np.sqrt(psi_minussq/len(psi_theta))
        ovb_ci_upper = theta_plus + 1.96*np.sqrt(psi_plussq/len(psi_theta))

        if args.RV:
            if args.RV_type == 'bound':
                gap1 = theta_minus - theta_t_lower
                gap2 = theta_t_upper - theta_plus
                # gap = min(gap1, gap2)
                # if gap < 0:
                #     raise Exception('Gap < 0; check code')
                # if theta_s < theta_t:
                #     gap = theta_minus - theta_t_lower
                # elif theta_s > theta_t:
                #     gap = theta_t_upper - theta_plus
            elif args.RV_type == 'bound_ci':

                gap1 = ovb_ci_lower - theta_t_lower
                gap2 = theta_t_upper - ovb_ci_upper
        
            gap = min(gap1, gap2)
        
        elif args.RV0:
            if theta_s >= 0:
                gap = theta_minus
            else:
                gap = -theta_plus

            # if gap < 0:
            #     raise Exception('Gap < 0; check code')
            # if theta_s < theta_t:
            #     gap = ovb_ci_lower - theta_t_lower
            # elif theta_s > theta_t:
            #     gap = theta_t_upper - ovb_ci_upper
        # Should be no overshoot for multi_ci scenario

        Cy += args.RV_interval
        Cd += args.RV_interval
        count += 1

    Cy -= args.RV_interval
    Cd -= args.RV_interval

    return (Cy, Cd)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

args = get_args()

true_effects = pd.read_csv(os.path.join(args.data_dir, args.true_effects_path))
# if args.num_features is not None:
#     true_effects = true_effects.iloc[0:min(args.num_features, true_effects.shape[0])]
df_original = pd.read_csv(os.path.join(args.data_dir, args.csv_path))
random.seed(args.seed)

norm_string = ''
clf_string = ''
out_string = ''
num_features_string = ''
ood_string = ''
trunc_string = ''
if args.normalize_weights is not None:
    norm_string = 'norm_'
if args.clf != 'lr':
    clf_string = 'treat{}_'.format(args.clf)
if args.g != 'lr':
    out_string = 'out{}_'.format(args.g)
if args.no_ood:
    ood_string = 'no_ood_'
if args.truncate is not None:
    trunc_string = 'trunc{}_'.format(args.truncate)
    
drop_string = ''
if args.drop_feat is not None:
    drop_string = '_drop{}'.format(args.drop_feat)
# if args.RV_type is not None:
    # rvtype_string = 'rvtype{}_'.format(args.RV_type)
# if args.num_features is not None:
#     num_features_string = 'features{}_'.format(args.num_features)
rep_name = 'discretecovs'

if args.covariates == 'text':
    rep_name = '{}-{}-{}'.format(args.text_col, args.lm_library, args.lm_name)

if args.split_type == 'bootstrap':
    num_string = args.bootstrap_iters
elif args.split_type == 'cv':
    num_string = args.cv_folds

if args.split_type == 'cv':
    treatments = true_effects.feature.values
    num_treatments = len(treatments)
    if args.covariates == 'from_true_effects':
        all_features = true_effects.feature
    elif args.covariates == 'from_header':
        drop_cols = [args.outcome]
        # if args.treatment is not None:
        #     if args.treatment is list:
        #         drop_cols += args.treatment
        #     else:
        #         drop_cols += [args.treatment]
        all_features = df_original.drop(drop_cols, axis=1).columns
    elif args.covariates == 'text':
        all_features = np.append(treatments, [args.text_col])
        # raise Exception('Unstructured text not yet implemented')


    # if args.treatment is None:
    #     treatments = all_features
    # else:
    #     treatments = args.treatments
    # num_treatments = len(treatments)

    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    fold_idx = list(kf.split(df_original))
    folds = np.array(list(range(len(fold_idx))))
    df_full_dict = {}
    pred_y_ood_1_dict = {}
    pred_y_ood_0_dict = {}
    # pred_y_combined_dict = {}
    pred_y_ood_dict = {}
    ipw_1_weight_ood_dict = {}
    ipw_0_weight_ood_dict = {}
    as_ood_dict = {}
    if args.naive == 'true_outcome':
        true_y_ood_1_dict = {}
        true_y_ood_0_dict = {}

    for a_name in treatments:
        df_full_dict[a_name] = df_original.copy()
        df_full_dict[a_name]['ipw_1_weight'] = np.nan
        df_full_dict[a_name]['ipw_0_weight'] = np.nan
        pred_y_ood_1_dict[a_name] = []
        pred_y_ood_0_dict[a_name] = []
        # pred_y_combined_dict[a_name] = []
        pred_y_ood_dict[a_name] = []
        ipw_1_weight_ood_dict[a_name] = []
        ipw_0_weight_ood_dict[a_name] = []
        as_ood_dict[a_name] = []
        if args.naive == 'true_outcome':
            true_y_ood_1_dict[a_name] = []
            true_y_ood_0_dict[a_name] = []

    for i in tqdm(folds):
        estimation_fold = i
        if args.no_ood:
            ood_fold = i
            nuisance_folds = folds[folds != i]
        else:
            remaining_folds = folds[folds != estimation_fold]
            ood_fold = remaining_folds[0]
            nuisance_folds = remaining_folds[1:]

        nuisance_idx = np.concatenate([fold_idx[i][1] for i in nuisance_folds])
        estimation_idx = fold_idx[estimation_fold][1]
        ood_idx = fold_idx[ood_fold][1]
        # ood_idx = np.concatenate([fold_idx[i][1] for i in estimation_folds])

        df_nuisance = df_original.iloc[nuisance_idx]
        df_ood = df_original.iloc[ood_idx]
        df = df_original.iloc[estimation_idx]

        if args.naive == 'unadjusted':
            ipw_1_weights, ipw_0_weights, ipw_1_weights_ood, ipw_0_weights_ood, as_ood, preds_y_1, preds_y_0, preds_y_ood_1, preds_y_ood_0, preds_y_combined, preds_y, preds_y_ood, indicators_p, indicators_a = estimate(
                true_effects, df_nuisance, df_ood, df, i)
        elif args.naive == 'true_outcome':
            ipw_1_weights, ipw_0_weights, ipw_1_weights_ood, ipw_0_weights_ood, as_ood, preds_y_1, preds_y_0, preds_y_ood_1, preds_y_ood_0, preds_y_combined, preds_y, preds_y_ood, indicators_p, indicators_a, trues_y_1, trues_y_0, trues_y_ood_1, trues_y_ood_0 = estimate(
                true_effects, df_nuisance, df_ood, df, i)

        for j in range(num_treatments):
            a_name = treatments[j]
            df_full_dict[a_name].loc[estimation_idx, 'ipw_1_weight'] = ipw_1_weights[j]
            df_full_dict[a_name].loc[estimation_idx, 'ipw_0_weight'] = ipw_0_weights[j]
            df_full_dict[a_name].loc[estimation_idx, 'pred_y_1'] = preds_y_1[j]
            df_full_dict[a_name].loc[estimation_idx, 'pred_y_0'] = preds_y_0[j]
            df_full_dict[a_name].loc[estimation_idx, 'pred_y'] = preds_y[j]
            pred_y_ood_1_dict[a_name] += list(preds_y_ood_1[j])
            pred_y_ood_0_dict[a_name] += list(preds_y_ood_0[j])
            # pred_y_combined_dict[a_name] += list(preds_y_combined[i])
            pred_y_ood_dict[a_name] += list(preds_y_ood[j])
            ipw_1_weight_ood_dict[a_name] += list(ipw_1_weights_ood[j])
            ipw_0_weight_ood_dict[a_name] += list(ipw_0_weights_ood[j])
            as_ood_dict[a_name] += list(as_ood[j])
            if args.naive == 'true_outcome':
                df_full_dict[a_name].loc[estimation_idx, 'true_y_1'] = trues_y_1[j]
                df_full_dict[a_name].loc[estimation_idx, 'true_y_0'] = trues_y_0[j]
                true_y_ood_1_dict[a_name] += list(trues_y_ood_1[j])
                true_y_ood_0_dict[a_name] += list(trues_y_ood_0[j])

    ipw_estimates = []
    dr_estimates = []
    outcome_estimates = []
    naive_estimates = []
    ipw_SEs = []
    dr_SEs = []
    outcome_SEs = []
    naive_SEs = []

    sigmasq_estimates = []
    nusq_estimates = []

    psi_theta_estimates = []
    psi_sigmasq_estimates = []
    psi_nusq_estimates = []

    for i in range(num_treatments):
        a_name = treatments[i]
        a_true = true_effects.coef[i]
        df_feat = df_full_dict[a_name]
        pred_y_ood_1 = pred_y_ood_1_dict[a_name]
        pred_y_ood_0 = pred_y_ood_0_dict[a_name]
        pred_y_ood = pred_y_ood_dict[a_name]
        ipw_1_weight_ood = ipw_1_weight_ood_dict[a_name]
        ipw_0_weight_ood = ipw_0_weight_ood_dict[a_name]
        a_ood = as_ood_dict[a_name]
        # pred_y_combined = pred_y_combined_dict[a_name]
        n1 = (df_feat[a_name]==1).sum()
        n0 = (df_feat[a_name]==0).sum()
        corp_prob = (df_feat.shape[0]/(df_feat.shape[0]+len(pred_y_ood_1)),
                     len(pred_y_ood_1)/(df_feat.shape[0]+len(pred_y_ood_1)))
        treat_prob = (n1/(n1+n0), n0/(n1+n0))
        indicator_p = np.array([1]*df_feat.shape[0] + [0]*len(pred_y_ood_1))
        # indicator_p = indicators_p[i]
        y_combined = np.concatenate([df_feat[args.outcome].values, np.zeros(len(pred_y_ood_1))])
        # indicator_a = np.concatenate([df_feat[a_name]==1, np.zeros(len(pred_y_ood_1))])
        indicator_a = np.concatenate([df_feat[a_name]==1, a_ood])
        # indicator_a = indicators_a[i]
        pred_y_1_combined = np.concatenate([df_feat['pred_y_1'].values, pred_y_ood_1])
        pred_y_0_combined = np.concatenate([df_feat['pred_y_0'].values, pred_y_ood_0])
        pred_y_combined = np.concatenate([df_feat['pred_y'].values, pred_y_ood])
        ipw_1_weight_combined = np.concatenate([df_feat['ipw_1_weight'].values, ipw_1_weight_ood])
        ipw_0_weight_combined = np.concatenate([df_feat['ipw_0_weight'].values, ipw_0_weight_ood])

        if args.normalize_weights == 'norm_total':
            # ipw_1_weight /= np.nansum(ipw_1_weight)/df.shape[0]
            # ipw_1_weight_ood_ipwonly = ipw_1_weight_ood/(np.nansum(ipw_1_weight_ood)/df_ood.shape[0])
            # ipw_0_weight /= np.nansum(ipw_0_weight)/df.shape[0]
            # ipw_0_weight_ood_ipwonly = ipw_0_weight_ood/(np.nansum(ipw_0_weight_ood)/df_ood.shape[0])
            # ipw_1_weight_combined /= np.nansum(ipw_1_weight_combined*indicator_a)/len(indicator_a)
            # ipw_0_weight_combined /= np.nansum(ipw_0_weight_combined*(1-indicator_a))/len(indicator_a)
            ipw_1_weight_combined /= np.nansum(ipw_1_weight_combined)/len(indicator_a)
            ipw_0_weight_combined /= np.nansum(ipw_0_weight_combined)/len(indicator_a)
        elif args.normalize_weights == 'norm_a':
            ipw_1_weight_combined /= np.nansum(ipw_1_weight_combined*indicator_a)/np.sum(indicator_a == 1)
            ipw_0_weight_combined /= np.nansum(ipw_0_weight_combined*(1-indicator_a))/np.sum(indicator_a == 0)

        if args.naive == 'true_outcome':
            true_y_ood_1 = true_y_ood_1_dict[a_name]
            true_y_ood_0 = true_y_ood_0_dict[a_name]

        # IPW estimate
        # mu1_ipw = np.nanmean(y_combined*ipw_1_weight_combined*indicator_p*indicator_a/(corp_prob[0]*treat_prob[0]))
        # mu0_ipw = np.nanmean(y_combined*ipw_0_weight_combined*indicator_p*(1-indicator_a)/(corp_prob[0]*treat_prob[1]))
        mu1_ipw = np.nanmean(y_combined*ipw_1_weight_combined*indicator_p*indicator_a/(corp_prob[0]))
        mu0_ipw = np.nanmean(y_combined*ipw_0_weight_combined*indicator_p*(1-indicator_a)/(corp_prob[0]))


        # mu1_ipw = np.nansum((df_feat[a_name]==1)*df_feat['ipw_1_weight']*df_feat[args.outcome])/n1
        # mu0_ipw = np.nansum((df_feat[a_name]==0)*df_feat['ipw_0_weight']*df_feat[args.outcome])/n0
        # se1_ipw = np.sqrt(np.var((df_feat['ipw_1_weight']*df_feat[args.outcome])[df_feat[a_name]==1])/n1)
        # se0_ipw = np.sqrt(np.var((df_feat['ipw_0_weight']*df_feat[args.outcome])[df_feat[a_name]==0])/n0)
        
        
        se1_ipw = np.sqrt(np.var((df_feat['ipw_1_weight']*df_feat[args.outcome])[df_feat[a_name]==1])/n1)
        se0_ipw = np.sqrt(np.var((df_feat['ipw_0_weight']*df_feat[args.outcome])[df_feat[a_name]==0])/n0)
        # TODO: SANITY CHECK THIS!!!!
        se1_ipw4 = np.sqrt(np.var(df_feat['ipw_1_weight']*df_feat[args.outcome]*df_feat[a_name])/df_feat.shape[0])
        se0_ipw4 = np.sqrt(np.var(df_feat['ipw_0_weight']*df_feat[args.outcome]*(1-df_feat[a_name]))/df_feat.shape[0])

        ate_ipw = mu1_ipw - mu0_ipw
        ipw_estimates.append(ate_ipw)
        se_ipw = se1_ipw + se0_ipw
        se_ipw4 = se1_ipw4 + se0_ipw4

        psi_ipw = y_combined*ipw_1_weight_combined*indicator_p*indicator_a - \
             y_combined*ipw_0_weight_combined*indicator_p*(1-indicator_a) - \
             ate_ipw*indicator_p

        se_ipw2 = np.sqrt(np.nanmean(psi_ipw[indicator_p==1]**2)/(np.sum(indicator_p)))
        se_ipw3 = np.sqrt(np.nansum(psi_ipw**2))/np.sum(indicator_p)

        # se_ipw_2 = np.sqrt(np.nanmean(
        #     (y_combined*ipw_1_weight_combined*indicator_p*indicator_a/(corp_prob[0]) - 
        #      y_combined*ipw_0_weight_combined*indicator_p*(1-indicator_a)/(corp_prob[0]) - 
        #      ate_ipw)**2)[indicator_p==1]/(len(y_combined)*corp_prob[0]))

        if args.se_type == 'se_original':
            ipw_SEs.append(se_ipw)
        elif args.se_type == 'se2':
            ipw_SEs.append(se_ipw3)
        elif args.se_type == 'se4':
            ipw_SEs.append(se_ipw4)

        # DR2 estimate
        # mu1_dr = np.nanmean((y_combined - pred_y_1_combined)*ipw_1_weight_combined*indicator_p*indicator_a/(corp_prob[0]*treat_prob[0]) +
        #                     pred_y_1_combined*(1-indicator_p)/corp_prob[1])
        # mu0_dr = np.nanmean((y_combined - pred_y_0_combined)*ipw_0_weight_combined*indicator_p*(1-indicator_a)/(corp_prob[0]*treat_prob[1]) +
        #             pred_y_0_combined*(1-indicator_p)/corp_prob[1])
        mu1_dr = np.nanmean((y_combined - pred_y_1_combined)*ipw_1_weight_combined*indicator_p*indicator_a/(corp_prob[0]) +
                            pred_y_1_combined*(1-indicator_p)/corp_prob[1])
        mu0_dr = np.nanmean((y_combined - pred_y_0_combined)*ipw_0_weight_combined*indicator_p*(1-indicator_a)/(corp_prob[0]) +
                    pred_y_0_combined*(1-indicator_p)/corp_prob[1])
        # mu1_dr = np.nansum((df_feat[a_name]==1)*df_feat['ipw_1_weight']*(df_feat[args.outcome] - df_feat['pred_y_1']))/n1 + \
        #     np.nansum(pred_y_ood_1)/len(pred_y_ood_1)
        # mu0_dr = np.nansum((df_feat[a_name]==0)*df_feat['ipw_0_weight']*(df_feat[args.outcome] - df_feat['pred_y_0']))/n0 + \
        #     np.nansum(pred_y_ood_0)/len(pred_y_ood_0)
        # se1_dr = np.sqrt(np.var((df_feat['ipw_1_weight']*(df_feat[args.outcome] - df_feat['pred_y_1']))[df_feat[a_name]==1])/n1 +
        #                  np.var(pred_y_ood_1)/len(pred_y_ood_1))
        # se0_dr = np.sqrt(np.var((df_feat['ipw_0_weight']*(df_feat[args.outcome] - df_feat['pred_y_0']))[df_feat[a_name]==0])/n0 +
        #                  np.var(pred_y_ood_0)/len(pred_y_ood_0))
        se1_dr = np.sqrt(np.var((df_feat['ipw_1_weight']*(df_feat[args.outcome] - df_feat['pred_y_1']))[df_feat[a_name]==1])/n1 +
                         np.var(pred_y_ood_1)/len(pred_y_ood_1))
        se0_dr = np.sqrt(np.var((df_feat['ipw_0_weight']*(df_feat[args.outcome] - df_feat['pred_y_0']))[df_feat[a_name]==0])/n0 +
                         np.var(pred_y_ood_0)/len(pred_y_ood_0))
        # TODO: SANITY CHECK
        se1_dr4 = np.sqrt(np.var(df_feat['ipw_1_weight']*df_feat[a_name]*(df_feat[args.outcome] - df_feat['pred_y_1']))/df_feat.shape[0] +
                         np.var(pred_y_ood_1)/len(pred_y_ood_1))
        se0_dr4 = np.sqrt(np.var(df_feat['ipw_0_weight']*(1-df_feat[a_name])*(df_feat[args.outcome] - df_feat['pred_y_0']))/df_feat.shape[0] +
                         np.var(pred_y_ood_0)/len(pred_y_ood_0))
    
        ate_dr = mu1_dr - mu0_dr

        psi_theta = (y_combined - pred_y_1_combined)*ipw_1_weight_combined*indicator_p*indicator_a/(corp_prob[0]) + \
            pred_y_1_combined*(1-indicator_p)/corp_prob[1] - \
                (y_combined - pred_y_0_combined)*ipw_0_weight_combined*indicator_p*(1-indicator_a)/(corp_prob[0]) + \
                pred_y_0_combined*(1-indicator_p)/corp_prob[1] - ate_dr
        psi_theta_estimates.append(psi_theta)
        # psi_sigmasq = 
        # psi_nusq = 

        dr_estimates.append(ate_dr)
        se_dr = se1_dr + se0_dr
        se_dr4 = se1_dr4 + se0_dr4
        se_dr2 = np.sqrt(np.nanmean(psi_theta**2)/len(psi_theta))
        se_dr3 = np.sqrt(np.nansum(psi_theta**2))/len(psi_theta)
        # se_dr = np.sqrt(np.nanmean(psi_theta**2)/len(y_combined))
        if args.se_type == 'se_original':
            dr_SEs.append(se_dr)
        elif args.se_type == 'se2':
            dr_SEs.append(se_dr3)
        elif args.se_type == 'se4':
            dr_SEs.append(se_dr4)
        # dr_SEs.append(se_dr3)
        
        pred_y_1 = list(df_feat['pred_y_1'].values)
        pred_y_0 = list(df_feat['pred_y_0'].values)
        mu1_outcome = np.nanmean(pred_y_1_combined)
        mu0_outcome = np.nanmean(pred_y_0_combined)
        # mu1_outcome = np.nansum(list(pred_y_1) + list(pred_y_ood_1))/len(list(pred_y_1) + list(pred_y_ood_1))
        # mu0_outcome = np.nansum(list(pred_y_0) + list(pred_y_ood_0))/len(list(pred_y_0) + list(pred_y_ood_0))
        se1_outcome = np.sqrt(np.var((df_feat[args.outcome] - df_feat['pred_y_1'])[df_feat[a_name]==1])/n1 +
                              np.var(pred_y_ood_1)/len(pred_y_ood_1))
        se0_outcome = np.sqrt(np.var((df_feat[args.outcome] - df_feat['pred_y_0'])[df_feat[a_name]==0])/n0 +
                              np.var(pred_y_ood_0)/len(pred_y_ood_0))
        
        se1_outcome2 = np.sqrt(np.var((df_feat[args.outcome] - df_feat['pred_y_1'])[df_feat[a_name]==1])/n1 + \
                               np.var(df_feat['pred_y_1'][df_feat[a_name]==0])/n0 +
                              np.var(pred_y_ood_1)/len(pred_y_ood_1))
        se0_outcome2 = np.sqrt(np.var((df_feat[args.outcome] - df_feat['pred_y_0'])[df_feat[a_name]==0])/n0 + \
                               np.var(df_feat['pred_y_0'][df_feat[a_name]==1]/n1) +
                              np.var(pred_y_ood_0)/len(pred_y_ood_0))

        ate_outcome = mu1_outcome - mu0_outcome
        outcome_estimates.append(ate_outcome)
        se_outcome = se1_outcome + se0_outcome
        se_outcome2 = se1_outcome2 + se0_outcome2

        # psi_outcome = pred_y_1_combined - pred_y_0_combined - ate_outcome
        # psi_outcome2 = (y_combined - pred_y_1_combined)*indicator_p*indicator_a/(corp_prob[0]) + pred_y_1_combined*(1-indicator_p)/corp_prob[1] - (y_combined - pred_y_0_combined)*indicator_p*(1-indicator_a)/(corp_prob[0]) + pred_y_0_combined*(1-indicator_p)/corp_prob[1] 
        # se_outcome3 = np.sqrt(np.nansum(psi_outcome2**2))/len(psi_outcome2)

        outcome_SEs.append(se_outcome)

        sigmasq = np.nanmean(((y_combined - pred_y_combined)**2)*indicator_p/corp_prob[0])
        sigmasq_estimates.append(sigmasq)
        alpha_s = ipw_1_weight_combined*indicator_a - ipw_0_weight_combined*(1-indicator_a)
        if args.nu_type == 'plugin':
            nusq = np.nanmean((alpha_s**2)*indicator_p/corp_prob[0])
        elif args.nu_type == 'dr':
            # nusq = np.nanmean(2*(ipw_1_weight_combined - ipw_0_weight_combined)*(1-indicator_p)/corp_prob[1]) - \
            #                   np.nanmean((alpha_s**2)*indicator_p/corp_prob[0])
            # nusq2 = nusq = np.nanmean(2*(ipw_1_weight_combined - ipw_0_weight_combined)*(1-indicator_p)/corp_prob[1] - 
            #                   np.nanmean((alpha_s**2)*indicator_p/corp_prob[0]))
            nusq = np.nanmean(2*(ipw_1_weight_combined + ipw_0_weight_combined)*(1-indicator_p)/corp_prob[1]) - \
                    np.nanmean((alpha_s**2)*indicator_p/corp_prob[0])
            # print(nusq, nusq2)

        nusq_estimates.append(nusq)

        if args.save_preds:
            # drop_string = ''
            # if args.drop_feat is not None:
            #     drop_string = '_drop{}'.format(args.drop_feat)
            alpha_res_path = os.path.join(args.pred_dir, '{}_clf{}_{}{}{}_type{}_{}{}.npy'.format(
                a_name, args.clf, rep_name, num_features_string, drop_string, args.estimation_strat, ood_string, norm_string))
            # pdb.set_trace()

            g_res_path = os.path.join(args.pred_dir, '{}_g{}_{}{}{}_type{}_{}{}.npy'.format(
                args.outcome, args.g, rep_name, num_features_string, drop_string, args.estimation_strat, ood_string, norm_string))
            
            y_path = os.path.join(args.pred_dir, '{}_true.npy'.format(
                args.outcome))
            
            np.save(g_res_path, pred_y_combined[indicator_p==1])
            np.save(alpha_res_path, alpha_s[indicator_p==1])
            np.save(y_path, y_combined[indicator_p==1])

        if args.plot_weights:
            trunc_plot_string = ''
            if args.truncate is not None:
                trunc_plot_string = '_trunc{}'.format(args.truncate)
            clf_plot_string = args.clf
            if args.clf != args.g:
                clf_plot_string = '{}{}'.format(args.clf, args.g)
            import matplotlib.pyplot as plt
            plt.hist(ipw_1_weight_combined)
            if 'amazon' in args.data_dir:
                ipw1_image = './plots/amazon_synthetic/weights/{}_features{}_type{}_{}ipw1weights{}.png'.format(
                    a_name, args.num_features, args.estimation_strat, clf_plot_string, trunc_plot_string)
                ipw0_image = './plots/amazon_synthetic/weights/{}_features{}_type{}_{}ipw0weights{}.png'.format(
                    a_name, args.num_features, args.estimation_strat, clf_plot_string, trunc_plot_string)
            # elif 'tirzepatide' in args.data_dir:
            #     ipw1_image = './plots/tirzepatide/weights/{}'
            plt.savefig(ipw1_image)
            plt.close()
            plt.hist(ipw_0_weight_combined)
            plt.savefig(ipw0_image)
            print('{:.3f} [{:.3f}, {:.3f}]'.format(
                np.mean(ipw_1_weight_combined),
                np.quantile(ipw_1_weight_combined, 0.025),
                np.quantile(ipw_1_weight_combined, 0.975)
            ))
            print('{:.3f} [{:.3f}, {:.3f}]'.format(
                np.mean(ipw_0_weight_combined),
                np.quantile(ipw_0_weight_combined, 0.025),
                np.quantile(ipw_0_weight_combined, 0.975)
            ))
            print('{:.3f} [{:.3f}, {:.3f}]'.format(
                np.mean(ipw_1_weight_combined[indicator_a==1]),
                np.quantile(ipw_1_weight_combined[indicator_a==1], 0.025),
                np.quantile(ipw_1_weight_combined[indicator_a==1], 0.975)
            ))
            print('{:.3f} [{:.3f}, {:.3f}]'.format(
                np.mean(ipw_0_weight_combined[indicator_a==0]),
                np.quantile(ipw_0_weight_combined[indicator_a==0], 0.025),
                np.quantile(ipw_0_weight_combined[indicator_a==0], 0.975)
            ))
            weights_dict = {'num_features': [args.num_features]*4,
                            'weight_type': ['1/p(a=1|a^c)', 
                                            '1/p(a=0|a^c)',
                                            'a/p(a=1|a^c)', 
                                            '(1-a)/p(a=0|a^c)'],
                            'mean': [np.mean(ipw_1_weight_combined),
                                        np.mean(ipw_0_weight_combined),
                                        np.mean(ipw_1_weight_combined[indicator_a==1]),
                                        np.mean(ipw_0_weight_combined[indicator_a==0])],
                            'lower': [np.quantile(ipw_1_weight_combined, 0.025),
                                        np.quantile(ipw_0_weight_combined, 0.025),
                                        np.quantile(ipw_1_weight_combined[indicator_a==1], 0.025),
                                        np.quantile(ipw_0_weight_combined[indicator_a==0], 0.025)], 
                            'upper': [np.quantile(ipw_1_weight_combined, 0.975),
                                        np.quantile(ipw_0_weight_combined, 0.975),
                                        np.quantile(ipw_1_weight_combined[indicator_a==1], 0.975),
                                        np.quantile(ipw_0_weight_combined[indicator_a==0], 0.975)]}
            df_weights = pd.DataFrame(weights_dict)
            mode = 'w'
            header = True
            if args.append:
                mode = 'a'
                header = False
            df_weights.to_csv('./results/amazon_synthetic/weights/{}_{}_type{}_{}weights{}.csv'.format(
                a_name, args.clf, args.estimation_strat, trunc_plot_string), mode=mode, header=header)
                 
        psi_sigmasq = (((y_combined - pred_y_combined)**2)/corp_prob[0] - sigmasq)*indicator_p
        psi_sigmasq_estimates.append(psi_sigmasq)
        if args.nu_type == 'plugin':
            psi_nusq = (alpha_s**2/corp_prob[0] - nusq)*indicator_p # TODO: Confirm that it's necessary to have the indicator and that indicator should be outside the centering
        elif args.nu_type == 'dr':
            # psi_nusq = 2*(ipw_1_weight_combined - ipw_0_weight_combined)*(1-indicator_p)/corp_prob[1] - (alpha_s**2)*indicator_p/corp_prob[0] - nusq
            psi_nusq = 2*(ipw_1_weight_combined + ipw_0_weight_combined)*(1-indicator_p)/corp_prob[1] - (alpha_s**2)*indicator_p/corp_prob[0] - nusq
        psi_nusq_estimates.append(psi_nusq)

        if args.naive == 'unadjusted':
            mu1_naive = ((df_feat[a_name]==1)*df_feat[args.outcome]).sum()/n1
            mu0_naive = ((df_feat[a_name]==0)*df_feat[args.outcome]).sum()/n0
            se1_naive = np.sqrt(np.var(df_feat[args.outcome][df_feat[a_name]==1])/n1)
            se0_naive = np.sqrt(np.var(df_feat[args.outcome][df_feat[a_name]==0])/n0)
       
        elif args.naive == 'true_outcome':
            true_y_1 = list(df_feat['true_y_1'].values)
            true_y_0 = list(df_feat['true_y_0'].values)
            mu1_naive = np.nansum(list(true_y_1) + list(true_y_ood_1))/len(list(true_y_1) + list(true_y_ood_1))
            mu0_naive = np.nansum(list(true_y_0) + list(true_y_ood_0))/len(list(true_y_0) + list(true_y_ood_0))
            se1_naive = np.sqrt(np.var((df_feat[args.outcome] - df_feat['true_y_1'])[df_feat[a_name]==1])/n1 +
                                np.var(true_y_ood_1)/len(true_y_ood_1))
            se0_naive = np.sqrt(np.var((df_feat[args.outcome] - df_feat['true_y_0'])[df_feat[a_name]==0])/n0 +
                                np.var(true_y_ood_0)/len(true_y_ood_0))

        ate_naive = mu1_naive - mu0_naive
        naive_estimates.append(ate_naive)
        se_naive = se1_naive + se0_naive
        naive_SEs.append(se_naive)

    df_results = true_effects.rename(columns={'coef': 'ate_true', 'feature': 'treatment'})
    df_results['ate_ipw'] = ipw_estimates
    df_results['ate_dr'] = dr_estimates
    df_results['ate_outcome'] = outcome_estimates
    df_results['ate_naive'] = naive_estimates
    df_results['sigmasq'] = sigmasq_estimates
    df_results['nusq'] = nusq_estimates

    max_len = 0
    for psi_theta in psi_theta_estimates:
        max_len = max(max_len, len(psi_theta))
    psi_mask = np.zeros((num_treatments, max_len))
    for j in range(len(psi_theta_estimates)):
        orig_len = len(psi_theta_estimates[j])
        psi_theta_estimates[j] = np.concatenate([psi_theta_estimates[j], np.zeros(max_len - orig_len)])
        psi_sigmasq_estimates[j] = np.concatenate([psi_sigmasq_estimates[j], np.zeros(max_len - orig_len)])
        psi_nusq_estimates[j] = np.concatenate([psi_nusq_estimates[j], np.zeros(max_len - orig_len)])
        psi_mask[j, 0:orig_len] = 1

    psi_theta = np.vstack(psi_theta_estimates)
    psi_sigmasq = np.vstack(psi_sigmasq_estimates)
    psi_nusq = np.vstack(psi_nusq_estimates)

    ipw_lower = np.array(ipw_estimates) - 1.96*np.array(ipw_SEs)
    ipw_upper = np.array(ipw_estimates) + 1.96*np.array(ipw_SEs)
    dr_lower = np.array(dr_estimates) - 1.96*np.array(dr_SEs)
    dr_upper = np.array(dr_estimates) + 1.96*np.array(dr_SEs)
    naive_lower = np.array(naive_estimates) - 1.96*np.array(naive_SEs)
    naive_upper = np.array(naive_estimates) + 1.96*np.array(naive_SEs)
    outcome_lower = np.array(outcome_estimates) - 1.96*np.array(outcome_SEs)
    outcome_upper = np.array(outcome_estimates) + 1.96*np.array(outcome_SEs)

elif args.split_type == 'bootstrap':
    treatments = true_effects.feature.values
    num_treatments = len(treatments)

    # Generate a list of random seeds
    seeds = [random.randint(0, args.bootstrap_iters) for _ in range(args.bootstrap_iters)]
    all_ipw_estimates = np.zeros((args.bootstrap_iters, num_treatments))
    all_dr_estimates = np.zeros((args.bootstrap_iters, num_treatments))
    all_naive_estimates = np.zeros((args.bootstrap_iters, num_treatments))
    all_outcome_estimates = np.zeros((args.bootstrap_iters, num_treatments))
    all_sigmasq_estimates = np.zeros((args.bootstrap_iters, num_treatments))
    all_nusq_estimates = np.zeros((args.bootstrap_iters, num_treatments))

    all_psi_theta_estimates = []
    all_psi_sigmasq_estimates = []
    all_psi_nusq_estimates = []

    # Bootstrapping
    j = 0
    j_extra = 0
    pbar = tqdm(total = args.bootstrap_iters)
    while j < args.bootstrap_iters:
    # for j in tqdm(range(args.bootstrap_iters)):
        try:
            df = df_original.copy()
            df = df.sample(n=df.shape[0], replace=True, random_state=seeds[j+j_extra], ignore_index=True)
            df_nuisance, df = train_test_split(df, test_size=0.8, shuffle=True, random_state=args.seed)
            if args.no_ood:
                df_ood = df.copy()
            else:
                df_ood, df = train_test_split(df, test_size=0.75, shuffle=True, random_state=args.seed)


            ipw_estimates, dr_estimates, naive_estimates, outcome_estimates, sigmasq_estimates, nusq_estimates, psi_theta_estimates, psi_sigmasq_estimates, psi_nusq_estimates = estimate(
                true_effects, df_nuisance, df_ood, df, j)

            all_ipw_estimates[j] = ipw_estimates
            all_dr_estimates[j] = dr_estimates
            all_naive_estimates[j] = naive_estimates
            all_outcome_estimates[j] = outcome_estimates
            all_sigmasq_estimates[j] = sigmasq_estimates
            all_nusq_estimates[j] = nusq_estimates

            all_psi_theta_estimates.append(psi_theta_estimates)
            all_psi_sigmasq_estimates.append(psi_sigmasq_estimates)
            all_psi_nusq_estimates.append(psi_nusq_estimates)

            pbar.update(1)
            j += 1
        except:
            print('Bootstrap sample has overlap issue')
            j_extra += 1
            seeds += [random.randint(0, args.bootstrap_iters)]
            continue

    df_results = true_effects.rename(columns={'coef': 'ate_true', 'feature': 'treatment'})
    df_results['ate_ipw'] = all_ipw_estimates.mean(axis=0)
    df_results['ate_dr'] = all_dr_estimates.mean(axis=0)
    df_results['ate_outcome'] = all_outcome_estimates.mean(axis=0)
    df_results['ate_naive'] = all_naive_estimates.mean(axis=0)
    df_results['sigmasq'] = all_sigmasq_estimates.mean(axis=0)
    df_results['nusq'] = all_nusq_estimates.mean(axis=0)

    psi_theta = np.concatenate(all_psi_theta_estimates, axis=1)
    psi_sigmasq = np.concatenate(all_psi_sigmasq_estimates, axis=1)
    psi_nusq = np.concatenate(all_psi_nusq_estimates, axis=1)

    ipw_lower = np.apply_along_axis(lambda col: quantile_without_inf(col, 0.025), axis=0, arr=all_ipw_estimates)
    ipw_upper = np.apply_along_axis(lambda col: quantile_without_inf(col, 0.975), axis=0, arr=all_ipw_estimates)
    dr_lower = np.apply_along_axis(lambda col: quantile_without_inf(col, 0.025), axis=0, arr=all_dr_estimates)
    dr_upper = np.apply_along_axis(lambda col: quantile_without_inf(col, 0.975), axis=0, arr=all_dr_estimates)
    naive_lower = np.apply_along_axis(lambda col: quantile_without_inf(col, 0.025), axis=0, arr=all_naive_estimates)
    naive_upper = np.apply_along_axis(lambda col: quantile_without_inf(col, 0.975), axis=0, arr=all_naive_estimates)
    outcome_lower = np.apply_along_axis(lambda col: quantile_without_inf(col, 0.025), axis=0, arr=all_outcome_estimates)
    outcome_upper = np.apply_along_axis(lambda col: quantile_without_inf(col, 0.975), axis=0, arr=all_outcome_estimates)

df_results['num_features'] = args.num_features
df_results['covariates'] = args.covariates
df_results['text_col'] = 'NA'
df_results['lm_library'] = 'NA'
df_results['lm_name'] = 'NA'
if args.covariates == 'text':
    df_results['text_col'] = args.text_col
    df_results['lm_library'] = args.lm_library
    df_results['lm_name'] = args.lm_name
df_results['clf'] = args.clf
df_results['g'] = args.g

df_results['ate_ipw_lower'] = ipw_lower
df_results['ate_ipw_upper'] = ipw_upper

df_results['ate_dr_lower'] = dr_lower
df_results['ate_dr_upper'] = dr_upper

df_results['ate_outcome_lower'] = outcome_lower
df_results['ate_outcome_upper'] = outcome_upper

df_results['ate_naive_lower'] = naive_lower
df_results['ate_naive_upper'] = naive_upper

df_results['se'] = np.sqrt(np.nanmean(psi_theta**2, axis=1)/psi_theta.shape[1])
df_results['Ssq'] = df_results['sigmasq']*df_results['nusq']

true_effects.rename(columns={'coef': 'ate'}, inplace=True)
df_results_long = true_effects.copy()
df_results_long['estimate_type'] = 'true'
df_results_long['lower'] = df_results_long['ate']
df_results_long['upper'] = df_results_long['ate']

df_results_ipw = true_effects.copy()
df_results_ipw['ate'] = df_results['ate_ipw']
df_results_ipw['estimate_type'] = 'ipw'
df_results_ipw['lower'] = ipw_lower
df_results_ipw['upper'] = ipw_upper

df_results_dr = true_effects.copy()
df_results_dr['ate'] = df_results['ate_dr']
df_results_dr['estimate_type'] = 'dr'
df_results_dr['lower'] = dr_lower
df_results_dr['upper'] = dr_upper

df_results_outcome = true_effects.copy()
df_results_outcome['ate'] = df_results['ate_outcome']
df_results_outcome['estimate_type'] = 'outcome model'
df_results_outcome['lower'] = outcome_lower
df_results_outcome['upper'] = outcome_upper

df_results_naive = true_effects.copy()
df_results_naive['ate'] = df_results['ate_naive']
df_results_naive['estimate_type'] = 'naive'
df_results_naive['lower'] = naive_lower
df_results_naive['upper'] = naive_upper

df_results['theta_minus'] = df_results['ate_dr'] - np.abs(args.rho)*np.sqrt(np.abs(df_results['Ssq']))*args.Cy*args.Cd
df_results['theta_plus'] = df_results['ate_dr'] + np.abs(args.rho)*np.sqrt(np.abs(df_results['Ssq']))*args.Cy*args.Cd

psi_minus = (psi_theta - np.abs(args.rho)*args.Cy*args.Cd/(2*np.sqrt(np.abs(df_results['Ssq'].values.reshape(-1,1))))*(df_results['sigmasq'].values.reshape(-1,1)*psi_nusq+df_results['nusq'].values.reshape(-1,1)*psi_sigmasq))*psi_mask
psi_plus = (psi_theta + np.abs(args.rho)*args.Cy*args.Cd/(2*np.sqrt(np.abs(df_results['Ssq'].values.reshape(-1,1))))*(df_results['sigmasq'].values.reshape(-1,1)*psi_nusq+df_results['nusq'].values.reshape(-1,1)*psi_sigmasq))*psi_mask

psi_minussq = np.nansum(psi_minus**2, axis=1)/np.nansum(psi_mask, axis=1)
psi_plussq = np.nansum(psi_plus**2, axis=1)/np.nansum(psi_mask, axis=1)

df_results['ovb_ci_lower'] = df_results['theta_minus'] - 1.96*np.sqrt(psi_minussq/psi_theta.shape[1])
df_results['ovb_ci_upper'] = df_results['theta_plus'] + 1.96*np.sqrt(psi_plussq/psi_theta.shape[1])

df_results['psi_theta'] = list(psi_theta)
df_results['psi_nusq'] = list(psi_nusq)
df_results['psi_sigmasq'] = list(psi_sigmasq)

df_results_ovb = true_effects.copy()
df_results_ovb['ate'] = df_results['ate_dr']
df_results_ovb['estimate_type'] = 'ovb'
df_results_ovb['lower'] = df_results['theta_minus']
df_results_ovb['upper'] = df_results['theta_plus']

df_results_ovb_ci = true_effects.copy()
df_results_ovb_ci['ate'] = df_results['ate_dr']
df_results_ovb_ci['estimate_type'] = 'ovb_ci'
df_results_ovb_ci['lower'] = df_results['ovb_ci_lower']
df_results_ovb_ci['upper'] = df_results['ovb_ci_upper']


if args.RV:
    tracking_df = pd.DataFrame({'treatment': df_results['treatment'].values,
                                'Cy': [0]*df_results.shape[0],
                                'Cd': [0]*df_results.shape[0]})

    df_multi = pd.read_csv(args.RV_csv)
    theta_t_lower = df_results['ate_true'] - 1.96*df_results['se']
    theta_t_lower = theta_t_lower.to_frame()
    theta_t_lower.columns = ['theta_t_lower']
    theta_t_lower['treatment'] = df_results['treatment']

    theta_t_upper = df_results['ate_true'] + 1.96*df_results['se']
    theta_t_upper = theta_t_upper.to_frame()
    theta_t_upper.columns = ['theta_t_upper']
    theta_t_upper['treatment'] = df_results['treatment']

    for feat in df_results['treatment']:
        df_multi_sub = df_multi[df_multi['treatment'] == feat]
        df_results_sub = df_results[df_results['treatment'] == feat]
        if args.RV_type == 'bound':
            lower_ref = df_multi_sub['theta_minus']
            upper_ref = df_multi_sub['theta_plus']
        elif args.RV_type == 'bound_ci':
            lower_ref = df_multi_sub['ovb_ci_lower']
            upper_ref = df_multi_sub['ovb_ci_upper']

        if np.any(lower_ref < theta_t_lower[theta_t_lower['treatment'] == feat]['theta_t_lower'].values[0]):
            max_dist_1 = np.max(np.abs(df_results_sub['ate_true'].values[0] - lower_ref.values))
        else:
            max_dist_1 = 0
        if np.any(upper_ref > theta_t_upper[theta_t_upper['treatment'] == feat]['theta_t_upper'].values[0]):
            max_dist_2 = np.max(np.abs(df_results_sub['ate_true'].values[0] - upper_ref.values))
        else:
            max_dist_2 = 0
        max_dist = max(max_dist_1, max_dist_2)
        theta_t_lower.loc[theta_t_lower['treatment'] == feat, 'theta_t_lower'] = df_results_sub['ate_true'] - max_dist
        theta_t_upper.loc[theta_t_upper['treatment'] == feat, 'theta_t_upper'] = df_results_sub['ate_true'] + max_dist


    for feat_idx, feat in df_results['treatment'].items():
        df_results_sub = df_results[df_results['treatment'] == feat]
        Cy, Cd = compute_RV(theta_s = df_results_sub['ate_dr'].values[0], 
                            theta_t = df_results_sub['ate_true'].values[0],
                            theta_t_lower = theta_t_lower[theta_t_lower['treatment'] == feat]['theta_t_lower'].values[0],
                            theta_t_upper = theta_t_upper[theta_t_upper['treatment'] == feat]['theta_t_upper'].values[0],
                            psi_theta = psi_theta[feat_idx], 
                            sigmasq = df_results_sub['sigmasq'].values[0], 
                            nusq = df_results_sub['nusq'].values[0], 
                            Ssq = df_results_sub['Ssq'].values[0], 
                            psi_nusq = psi_nusq[feat_idx], 
                            psi_sigmasq = psi_sigmasq[feat_idx])
        tracking_df.loc[tracking_df['treatment'] == feat, 'Cy'] = Cy
        tracking_df.loc[tracking_df['treatment'] == feat, 'Cd'] = Cd

    df_results['theta_minus'] = df_results['ate_dr'] - np.abs(args.rho)*np.sqrt(np.abs(df_results['Ssq']))*tracking_df['Cy']*tracking_df['Cd']
    df_results['theta_plus'] = df_results['ate_dr'] + np.abs(args.rho)*np.sqrt(np.abs(df_results['Ssq']))*tracking_df['Cy']*tracking_df['Cd']

    df_results_ovb['lower'] = df_results['theta_minus']
    df_results_ovb['upper'] = df_results['theta_plus']

    psi_minus = (psi_theta - np.abs(args.rho)*tracking_df['Cy'].values.reshape(-1, 1)*tracking_df['Cd'].values.reshape(-1, 1)/(2*np.sqrt(np.abs(df_results['Ssq'].values.reshape(-1,1))))*(df_results['sigmasq'].values.reshape(-1,1)*psi_nusq+df_results['nusq'].values.reshape(-1,1)*psi_sigmasq))*psi_mask
    psi_plus = (psi_theta + np.abs(args.rho)*tracking_df['Cy'].values.reshape(-1, 1)*tracking_df['Cd'].values.reshape(-1, 1)/(2*np.sqrt(np.abs(df_results['Ssq'].values.reshape(-1,1))))*(df_results['sigmasq'].values.reshape(-1,1)*psi_nusq+df_results['nusq'].values.reshape(-1,1)*psi_sigmasq))*psi_mask
    
    psi_minussq = np.nansum(psi_minus**2, axis=1)/np.nansum(psi_mask, axis=1)
    psi_plussq = np.nansum(psi_plus**2, axis=1)/np.nansum(psi_mask, axis=1)


    df_results['ovb_ci_lower'] = df_results['theta_minus'] - 1.96*np.sqrt(psi_minussq/psi_theta.shape[1])
    df_results['ovb_ci_upper'] = df_results['theta_plus'] + 1.96*np.sqrt(psi_plussq/psi_theta.shape[1])

    df_results_ovb_ci['lower'] = df_results['ovb_ci_lower']
    df_results_ovb_ci['upper'] = df_results['ovb_ci_upper']

    df_results['RV'] = tracking_df['Cy']*tracking_df['Cd']
    df_results['RV_type'] = args.RV_type

if args.compute_ovb_grid:
    df_grid_list = []
    for Cy in tqdm(np.arange(0, args.Cy_upper, args.RV_interval)):
        Cy_array = np.repeat(Cy, df_results.shape[0])
        for Cd in np.arange(0, args.Cd_upper, args.RV_interval):
            df_grid = df_results[['treatment', 'ate_dr', 'Ssq', 'nusq', 'sigmasq']]
            Cd_array = np.repeat(Cd, df_results.shape[0])
            df_grid['Cy'] = Cy_array
            df_grid['Cd'] = Cd_array

            df_grid['theta_minus'] = df_results['ate_dr'] - np.abs(args.rho)*np.sqrt(np.abs(df_results['Ssq']))*Cy_array*Cd_array
            df_grid['theta_plus'] = df_results['ate_dr'] + np.abs(args.rho)*np.sqrt(np.abs(df_results['Ssq']))*Cy_array*Cd_array

            psi_minus = (psi_theta - np.abs(args.rho)*Cy_array.reshape(-1, 1)*Cd_array.reshape(-1, 1)/(2*np.sqrt(np.abs(df_results['Ssq'].values.reshape(-1,1))))*(df_results['sigmasq'].values.reshape(-1,1)*psi_nusq+df_results['nusq'].values.reshape(-1,1)*psi_sigmasq))*psi_mask
            psi_plus = (psi_theta + np.abs(args.rho)*Cy_array.reshape(-1, 1)*Cd_array.reshape(-1, 1)/(2*np.sqrt(np.abs(df_results['Ssq'].values.reshape(-1,1))))*(df_results['sigmasq'].values.reshape(-1,1)*psi_nusq+df_results['nusq'].values.reshape(-1,1)*psi_sigmasq))*psi_mask
    
            psi_minussq = np.nansum(psi_minus**2, axis=1)/np.nansum(psi_mask, axis=1)
            psi_plussq = np.nansum(psi_plus**2, axis=1)/np.nansum(psi_mask, axis=1)

            df_grid['ovb_ci_lower'] = df_results['theta_minus'] - 1.96*np.sqrt(psi_minussq/psi_theta.shape[1])
            df_grid['ovb_ci_upper'] = df_results['theta_plus'] + 1.96*np.sqrt(psi_plussq/psi_theta.shape[1])

            df_grid_list.append(df_grid)
    
    df_grid = pd.concat(df_grid_list, axis=0).reset_index(drop=True)
    df_grid['num_features'] = args.num_features
    df_grid['covariates'] = args.covariates
    df_grid['text_col'] = 'NA'
    df_grid['lm_library'] = 'NA'
    df_grid['lm_name'] = 'NA'
    if args.covariates == 'text':
        df_grid['text_col'] = args.text_col
        df_grid['lm_library'] = args.lm_library
        df_grid['lm_name'] = args.lm_name
    df_grid['clf'] = args.clf
    df_grid['g'] = args.g

    mode = 'w'
    header = True
    if args.append:
        mode = 'a'
        header = False

    df_grid.to_csv('{}_{}{}{}type{}_nu{}_{}{}{}{}{}_CyCdgrid.csv'.format(
        ''.join(args.output_csv.split('.')[:-1]),
        clf_string, out_string, num_features_string,
        args.estimation_strat, args.nu_type, ood_string, trunc_string, norm_string, args.split_type, num_string), 
        index=False, header=header, mode=mode)

if args.RV0:
    tracking_df = pd.DataFrame({'treatment': df_results['treatment'].values,
                                'Cy': [0]*df_results.shape[0],
                                'Cd': [0]*df_results.shape[0]})

    for feat_idx, feat in df_results['treatment'].items():
        df_results_sub = df_results[df_results['treatment'] == feat]
        Cy, Cd = compute_RV(theta_s = df_results_sub['ate_dr'].values[0], 
                            theta_t = df_results_sub['ate_true'].values[0],
                            theta_t_lower = 0,
                            theta_t_upper = 0,
                            psi_theta = psi_theta[feat_idx], 
                            sigmasq = df_results_sub['sigmasq'].values[0], 
                            nusq = df_results_sub['nusq'].values[0], 
                            Ssq = df_results_sub['Ssq'].values[0], 
                            psi_nusq = psi_nusq[feat_idx], 
                            psi_sigmasq = psi_sigmasq[feat_idx])
        tracking_df.loc[tracking_df['treatment'] == feat, 'Cy'] = Cy
        tracking_df.loc[tracking_df['treatment'] == feat, 'Cd'] = Cd

    df_results['theta_minus'] = df_results['ate_dr'] - np.abs(args.rho)*np.sqrt(np.abs(df_results['Ssq']))*tracking_df['Cy']*tracking_df['Cd']
    df_results['theta_plus'] = df_results['ate_dr'] + np.abs(args.rho)*np.sqrt(np.abs(df_results['Ssq']))*tracking_df['Cy']*tracking_df['Cd']

    df_results_ovb['lower'] = df_results['theta_minus']
    df_results_ovb['upper'] = df_results['theta_plus']

    psi_minus = (psi_theta - np.abs(args.rho)*tracking_df['Cy'].values.reshape(-1, 1)*tracking_df['Cd'].values.reshape(-1, 1)/(2*np.sqrt(np.abs(df_results['Ssq'].values.reshape(-1,1))))*(df_results['sigmasq'].values.reshape(-1,1)*psi_nusq+df_results['nusq'].values.reshape(-1,1)*psi_sigmasq))*psi_mask
    psi_plus = (psi_theta + np.abs(args.rho)*tracking_df['Cy'].values.reshape(-1, 1)*tracking_df['Cd'].values.reshape(-1, 1)/(2*np.sqrt(np.abs(df_results['Ssq'].values.reshape(-1,1))))*(df_results['sigmasq'].values.reshape(-1,1)*psi_nusq+df_results['nusq'].values.reshape(-1,1)*psi_sigmasq))*psi_mask
    
    psi_minussq = np.nansum(psi_minus**2, axis=1)/np.nansum(psi_mask, axis=1)
    psi_plussq = np.nansum(psi_plus**2, axis=1)/np.nansum(psi_mask, axis=1)


    df_results['ovb_ci_lower'] = df_results['theta_minus'] - 1.96*np.sqrt(psi_minussq/psi_theta.shape[1])
    df_results['ovb_ci_upper'] = df_results['theta_plus'] + 1.96*np.sqrt(psi_plussq/psi_theta.shape[1])

    df_results_ovb_ci['lower'] = df_results['ovb_ci_lower']
    df_results_ovb_ci['upper'] = df_results['ovb_ci_upper']

    df_results['RV'] = tracking_df['Cy']*tracking_df['Cd']
    df_results['RV_type'] = args.RV_type

df_results_long = pd.concat([df_results_long, df_results_ipw, df_results_dr,
                             df_results_ovb, df_results_ovb_ci,
                             df_results_outcome, df_results_naive], axis=0)
df_results_long['num_features'] = args.num_features
df_results_long['covariates'] = args.covariates
df_results_long['text_col'] = 'NA'
df_results_long['lm_library'] = 'NA'
df_results_long['lm_name'] = 'NA'
if args.covariates == 'text':
    df_results_long['text_col'] = args.text_col
    df_results_long['lm_library'] = args.lm_library
    df_results_long['lm_name'] = args.lm_name
df_results_long['clf'] = args.clf
df_results_long['g'] = args.g

if args.save_results:

    
    mode = 'w'
    header = True
    if args.append:
        mode = 'a'
        header = False

    if args.output_csv is not None:
        df_results.to_csv('{}_{}{}{}type{}_nu{}_{}{}{}{}{}_wide.csv'.format(
            ''.join(args.output_csv.split('.')[:-1]),
            clf_string, out_string, num_features_string,
            args.estimation_strat, args.nu_type, ood_string, trunc_string, norm_string, args.split_type, num_string), 
            index=False, header=header, mode=mode)
        df_results_long.to_csv('{}_{}{}{}type{}_nu{}_{}{}{}{}{}_long.csv'.format(
            ''.join(args.output_csv.split('.')[:-1]),
            clf_string, out_string, num_features_string,
            args.estimation_strat, args.nu_type, ood_string, trunc_string, norm_string, args.split_type, num_string), 
            index=False, header=header, mode=mode)
    else:
        raise Exception('No output path provided')
        #     df_results.to_csv('./results/amazon_synthetic/pr_label_synthetic_direct1.00.0_iso_effects_type{}_norm_bootstrap{}_wide.csv'.format(
        #         args.estimation_strat, args.bootstrap_iters))
        #     df_results_long.to_csv('./results/amazon_synthetic/pr_label_synthetic_direct1.00.0_iso_effects_type{}_norm_bootstrap{}_long.csv'.format(
        #         args.estimation_strat, args.bootstrap_iters))
        # else:
        # df_results.to_csv('./results/amazon_synthetic/ovb/pr_label_synthetic_direct1.00.0_iso_effects_{}{}{}type{}_nu{}_{}{}{}{}{}_wide.csv'.format(
        #     clf_string, out_string, num_features_string, args.estimation_strat, args.nu_type, ood_string, trunc_string, norm_string, args.split_type, num_string),
        #     index=False, header=header, mode=mode)
        # df_results_long.to_csv('./results/amazon_synthetic/ovb/pr_label_synthetic_direct1.00.0_iso_effects_{}{}{}type{}_nu{}_{}{}{}{}{}_long.csv'.format(
        #     clf_string, out_string, num_features_string, args.estimation_strat, args.nu_type, ood_string, trunc_string, norm_string, args.split_type, num_string),
        #     index=False, header=header, mode=mode)
