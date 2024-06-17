import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern
from sklearn.metrics import mean_squared_error, r2_score

import random

random.seed(1578)

import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

from os import path

mpl.use('cairo')
mpl.rc('font', family='Palatino Linotype', size=18)

class PolymerMLBase:
    def __init__(self) -> None:
        self.highest_test_errors = 5
        self.abbreviation = {'power conversion efficiency': 'PCE'}

    def train_test_split(self, ml_dataset, train_frac, binary_mask=None):
        """Split known data into train and test set"""
        assert train_frac<1 and train_frac>0
        total_points = len(ml_dataset)
        num_points = int(total_points*train_frac)
        if binary_mask is None:
            binary_mask = np.concatenate((np.ones(num_points), np.zeros(total_points-num_points)))
            random.shuffle(binary_mask)
        train_set = [ml_dataset[i] for i in range(total_points) if binary_mask[i]==1]
        test_set = [ml_dataset[i] for i in range(total_points) if binary_mask[i]==0]
        return train_set, test_set, binary_mask

    def fingerprint_data():
        pass

    def vectorize_dict_features():
        pass

    def train_ml_model(self, X_train, y_train):
        """Train ML model and report test set performance and make a test set parity plot"""
        SCALE=.5
        k = 2 ** 2 * Matern(length_scale=SCALE, nu=1.5) + WhiteKernel(
            noise_level=5 ** 2,
            noise_level_bounds=(
                0.01**2, 10 ** 2))
        gp = GaussianProcessRegressor(kernel=k, alpha=0, n_restarts_optimizer=1, normalize_y=True)
        # Fit GPR kernel
        gp.fit(X_train, y_train)
        return gp

    def prediction(self, model, X_train, X_test, Y_train, Y_test, verbose=True, use_log_transform=False):
        '''Given the known y values associated with train and test set and a trained GPR model, compute the predicted values for train and test set and all corresponding error measures'''
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        # Find datapoints with highest test error
        if use_log_transform:
            Y_train = np.exp(Y_train)
            Y_test = np.exp(Y_test)
            Y_train_pred = np.exp(Y_train_pred)
            Y_test_pred = np.exp(Y_test_pred)
        gpr_error_test = np.abs(Y_test_pred - Y_test)

        # Find the k largest indices of gpr_test_error
        k_largest_indices = np.argpartition(gpr_error_test, -self.highest_test_errors)[-self.highest_test_errors:]

        
        # Get error metrics
        tr_error = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
        tt_error = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
        tr_r2 = r2_score(Y_train, Y_train_pred)
        tt_r2 = r2_score(Y_test, Y_test_pred)
        if verbose: return Y_train_pred, Y_test_pred, tr_error, tt_error, tt_r2, tr_r2
        else: return tr_error, tt_error, k_largest_indices
    

    def single_parity_plot(self, gp, property, units,  X_train, X_test, Y_train, Y_test, fig_name, output_dir, use_log_transform=False):
        '''Given train, test data and a model, plots a single parity plot for the given property'''
        Y_train_pred, Y_test_pred, tr_error, tt_error, tt_r2, tr_r2 = self.prediction(gp, X_train, X_test, Y_train, Y_test, verbose=True, use_log_transform=use_log_transform)
        if use_log_transform:
            Y_train = np.exp(Y_train)
            Y_test = np.exp(Y_test)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))#, facecolor='w', edgecolor='k')
        lim_min = min(min(Y_train), min(Y_test))
        lim_max = max(max(Y_train), max(Y_test))
        lim = [lim_min - (lim_max - lim_min) * 0.1, lim_max + (lim_max - lim_min) * 0.1]
        label_train = 'Train RMSE = ' + str('%.2f' % tr_error) + ', r = ' + str(
            '%.2f' % np.sqrt(tr_r2)) + ' (' + str(len(X_train)) + ' points)'
        label_test = 'Test RMSE  = ' + str('%.2f' % tt_error) + ', r = ' + str(
            '%.2f' % np.sqrt(tt_r2)) + ' (' + str(len(X_test)) + ' points)'


        ax.scatter(Y_train, Y_train_pred, s=50, c='#ff0018', edgecolor='#9b000e',
                      marker='o',
                      zorder=4,
                      label=label_train)
        ax.scatter(Y_test, Y_test_pred, s=50, c='#0d78b3', edgecolor='k', marker='s',
                      zorder=5,
                      label=label_test)

        ax.plot(lim, lim, dashes=[5, 3], c='k', lw=2, zorder=0)
        ax.set_xlabel(f"Experimental {property} ({units})", fontsize=18, fontweight='bold')
        ax.set_ylabel(f"ML Predicted {property} ({units})", fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')

        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')
        ax.legend(fontsize=16, loc='upper left')
        ax.grid(False)

        # Ensure axes are visible
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # Optionally set the color of the axes spines
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')

        # Ensure the face color of the plot is not grey
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        fig.savefig(path.join(output_dir, fig_name))


class PolymerSingleMaterialBase(PolymerMLBase):
    def __init__(self) -> None:
        super(PolymerSingleMaterialBase, self).__init__()

    def fingerprint_data(self, data_tuple, use_log_transform=False):
        """Create fingerprints for input data"""
        X = []
        y = []
        for polymer, fp_dict, prop_value in data_tuple: # Assume that polymer is already fingerprinted
            if use_log_transform:
                prop_value = np.log(prop_value)
            if fp_dict:
                X.append(fp_dict)
                y.append(prop_value)
            else:
                print(f'{polymer} had an empty fingerprint dictionary')
        
        return X, y
    
    def vectorize_dict_features(self, X, X_test):
        key_set = set()
        for x in X+X_test:
            for key, _ in x.items():
                key_set.add(key)
        X_out = np.zeros((len(X), len(key_set)))
        for i, x in enumerate(X):
            for j, key in enumerate(key_set):
                if key in x:
                    X_out[i][j]=x[key]

        X_test_out = np.zeros((len(X_test), len(key_set)))
        for i, x in enumerate(X_test):
            for j, key in enumerate(key_set):
                if key in x:
                    X_test_out[i][j]=x[key]

        return list(key_set), X_out, X_test_out


class PolymerDonorAcceptorBase(PolymerMLBase):
    def __init__(self) -> None:
        super(PolymerDonorAcceptorBase, self).__init__()
    
    def generate_fp_dict(self, data_dict_path):
        if path.exists(data_dict_path):
            with open(data_dict_path, 'rb') as fi:
                fp_dict = pickle.load(fi)
        else:
            print('Fingerprint dictionary not found. Please provide a valid path to the dictionary')
        
        return fp_dict

    def fingerprint_data(self, data_tuple, fp_dict, use_median=False, use_log_transform=False):
        """Create fingerprints for input data"""
        X = []
        y = []

        for donor, donor_smile, acceptor, acceptor_smile, prop_value_list, _ in data_tuple:
            if donor_smile and fp_dict[donor] and acceptor_smile and fp_dict[acceptor]:
                if use_log_transform:
                    prop_value_list = [np.log(prop_value) for prop_value in prop_value_list]
                if use_median:
                    X.append((fp_dict[donor], fp_dict[acceptor]))
                    y.append(np.median(prop_value_list))
                else:
                    for prop_value in prop_value_list:
                        X.append((fp_dict[donor], fp_dict[acceptor]))
                        y.append(prop_value)

        return X, y
    
    def vectorize_dict_features(self, X, X_test):
        reshape_output=False
        donor_key_set = set()
        acceptor_key_set = set()
        for donor_dict, acceptor_dict in X+X_test:
            for key, _ in donor_dict.items():
                donor_key_set.add(key)
            for key, _ in acceptor_dict.items():
                acceptor_key_set.add(key)
        
        X_out = np.zeros((len(X), len(donor_key_set)+len(acceptor_key_set)))
        for i, (donor_dict, acceptor_dict) in enumerate(X):
            # Donor and acceptor vectors concatenated
            for j, key in enumerate(donor_key_set):
                if key in donor_dict:
                    X_out[i][j]=donor_dict[key]
            for j, key in enumerate(acceptor_key_set):
                if key in acceptor_dict:
                    X_out[i][j+len(donor_key_set)]=acceptor_dict[key]
            
        X_test_out = np.zeros((len(X_test), len(donor_key_set)+len(acceptor_key_set)))
        for i, (donor_dict, acceptor_dict) in enumerate(X_test):
            # Donor and acceptor vectors concatenated
            for j, key in enumerate(donor_key_set):
                if key in donor_dict:
                    X_test_out[i][j]=donor_dict[key]
            for j, key in enumerate(acceptor_key_set):
                if key in acceptor_dict:
                    X_test_out[i][j+len(donor_key_set)]=acceptor_dict[key]
        
        return list(donor_key_set), list(acceptor_key_set), X_out, X_test_out