import numpy as np
import random

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, RBF, RationalQuadratic
from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import norm

from collections import defaultdict
from os import path

import matplotlib.pyplot as plt

import numpy as np

class SampleEfficiencyBaselines:
    """We will use four baselines, batch a random sample, train model, predict on remaining and then augment the training dataset
       Augment in 2 ways, select batch_size best or best and then random samples, could even be some combination of both
       Best can be best by mean value or mean+predicted uncertainty, The batch size at which the maximum value is added is the saturation step
       It might also be possible to use CB here to select the next batch to train on
    """
    def __init__(self, args, **kwargs):
        super(SampleEfficiencyBaselines, self).__init__()
        self.args = args
        self.verbose = False
        self.batch_size = 1
        self.SCALE=1.0
        self.v2_sampling_fraction = 0.05

        self.improvement_fraction = 0.01
    
    def train_ml_model(self, X_train, y_train):
        """Train ML model and report test set performance and make a test set parity plot"""

        if self.args.kernel=='Matern':
            k = (self.Y_avg) ** 2 * Matern(length_scale=self.SCALE, nu=1.5) + self.args.use_gpr_noise*WhiteKernel(
                noise_level=(self.noise_estimate) ** 2)
        elif self.args.kernel=='RBF':
            k = (self.Y_avg) ** 2 * RBF(length_scale=self.SCALE) + self.args.use_gpr_noise*WhiteKernel(
                noise_level=(self.noise_estimate) ** 2)
        elif self.args.kernel=='RationalQuadratic':
            k = (self.Y_avg) ** 2 * RationalQuadratic(length_scale=self.SCALE) + self.args.use_gpr_noise*WhiteKernel(
                noise_level=(self.noise_estimate) ** 2)
        gp = GaussianProcessRegressor(kernel=k, alpha=0, n_restarts_optimizer=5, normalize_y=True)
        # Fit GPR kernel
        gp.fit(X_train, y_train)
        return gp
        
    def prediction(self, gp, X_test, Y_test):
        '''Given the known y values associated with train and test set and a trained GPR model, compute the predicted values for train and test set and all corresponding error measures'''
        Y_test_pred, gpr_error_test = gp.predict(X_test, return_std=True)
        # Get error metrics
        tt_error = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
        tt_r2 = r2_score(Y_test, Y_test_pred)

        return Y_test_pred, gpr_error_test, tt_error, tt_r2


    def pick_subset(self, picked_subset, X, y):
        X_train, Y_train, X_test, Y_test = [], [], [], []
        num_points=len(y)
        for i in range(num_points):
            if i not in picked_subset:
                X_test.append(X[i])
                Y_test.append(y[i])
            else:
                X_train.append(X[i])
                Y_train.append(y[i])
        
        return X_train, X_test, Y_train, Y_test

    
    def pick_next_predictions(self, X_test, Y_test, Y_test_pred, gpr_test_error, mode, gp_mode, batch_count, max_reward, gp_alt_mode):
        """Pick which of the predicted points will be added to the training dataset to train the next model"""
        # Mode is a string which dictates how the selection should be done, run each mode separately as a baseline

        if len(Y_test)<=self.batch_size:
            return X_test, Y_test, [], []

        if mode=='GP-alt' and batch_count%2==1:
            mode=gp_alt_mode

        # Pick the appropriate subs
        if mode=='prediction':
            top_indices = np.argpartition(Y_test_pred, -self.batch_size)[-self.batch_size:]

        elif mode=='prediction and random':
            # Pick half the samples in batch by using maximum of predictions approach and pick the other half randomly
            top_indices = list(np.argpartition(Y_test_pred, -self.batch_size//2)[-self.batch_size//2:])
            sample_indices = [i for i in range(len(Y_test_pred)) if i not in top_indices]
            random_indices = random.sample(sample_indices, self.batch_size//2)
            top_indices += random_indices

        elif mode=='GP-UCB':
            if gp_mode == 'beta_0.5':
                beta=0.5
            elif gp_mode == 'beta_1.5':
                beta=1.5
            elif gp_mode == 'beta_decay_exp_0.25':
                beta=1.5/(float(batch_count))**0.25
            else:
                beta=1.0

            top_indices = np.argpartition(Y_test_pred+beta*gpr_test_error, -self.batch_size)[-self.batch_size:]

        elif mode=='GP-PI':
            standard_normal_array = np.divide((Y_test_pred-max_reward*(1+self.improvement_fraction)), gpr_test_error)
            probabilities = norm.cdf(standard_normal_array)
            top_indices = np.argpartition(probabilities, -self.batch_size)[-self.batch_size:]

        elif mode=='GP-EI':
            standard_normal_array = np.divide((Y_test_pred-max_reward*(1+self.improvement_fraction)), gpr_test_error)
            cdf_probabilities = norm.cdf(standard_normal_array)
            pdf_probabilities = norm.pdf(standard_normal_array)
            expected_improvement = (Y_test_pred-max_reward*(1+self.improvement_fraction))*cdf_probabilities+gpr_test_error*pdf_probabilities
            top_indices = np.argpartition(expected_improvement, -self.batch_size)[-self.batch_size:]

        elif mode=='GP-alt' and batch_count%2==0:
            top_indices = np.argpartition(gpr_test_error, -self.batch_size)[-self.batch_size:]

        elif mode=='GP-TS':
            sampled_rewards = np.random.normal(loc=Y_test_pred, scale=gpr_test_error)
            top_indices = np.argpartition(sampled_rewards, -self.batch_size)[-self.batch_size:]


        elif mode=='bayesian and random':
            # Pick half the samples in batch by a bayesian approach and pick the other half randomly
            top_indices = list(np.argpartition(Y_test_pred+gpr_test_error, -self.batch_size//2)[-self.batch_size//2:])
            sample_indices = [i for i in range(len(Y_test_pred)) if i not in top_indices]
            random_indices = random.sample(sample_indices, self.batch_size//2)
            top_indices += random_indices

        elif mode=='random': # Pick points at random
            top_indices=random.sample(list(range(len(Y_test))), self.batch_size)

        return top_indices

    
    def run_baseline(self, X, y, material_list, mode, initial_sample=None, gp_mode=None):
        if mode=='GP-UCB':
            assert gp_mode is not None
        y_target = np.max(y)
        # Select subset of data to train on using batch_size
        picked_subset = [initial_sample]
        print(f'Initial picked subset = {picked_subset}')
        X_train, X_test, Y_train, Y_test = self.pick_subset(picked_subset, X, y)
        batch_count=1
        model_stats = defaultdict(dict)
        print(f'Target y value is {y_target}')
        reward_history = [y_elem for index, y_elem in enumerate(y) if index in picked_subset]
        material_history = [material_list[index] for index in picked_subset]
        material_list = delete_material_items(material_list, picked_subset)
        max_reward = 0
        if mode=='random':
            Y_test_pred = []
            gpr_test_error = []

        while True:
            if y_target in Y_train and batch_count==1:
                saturation_steps=batch_count*self.batch_size
                break
            if mode!='random':
                gp = self.train_ml_model(X_train, Y_train)
                Y_test_pred, gpr_test_error, tt_error, tt_r2 = self.prediction(gp, X_test, Y_test)
                model_stats[batch_count-1]['test_rmse_error'] = tt_error # Computing test error would make more sense if we had a fixed dataset. Test error is being computed over all data points here
                model_stats[batch_count-1]['test_r2_error'] = tt_r2
            
            top_indices = self.pick_next_predictions(X_test, Y_test, Y_test_pred, gpr_test_error, mode, gp_mode, batch_count, max_reward, gp_alt_mode='GP-EI') #TODO: Pass this as an argument
            X_add, X_test, Y_add, Y_test = self.pick_subset(top_indices, X_test, Y_test)

            reward_history.extend(Y_add)
            material_history.extend([material_list[index] for index in top_indices])
            material_list = delete_material_items(material_list, top_indices)
            if max(Y_add)>max_reward:
                max_reward = max(Y_add)
            if y_target in Y_add:
                saturation_steps=batch_count*self.batch_size
                break
            else:
                X_train.extend(X_add)
                Y_train.extend(Y_add)
                batch_count+=1
        
        return saturation_steps, reward_history, material_history, model_stats
    
    def run_all_baselines(self, X, y, material_list, starting_indices):
        """One run for each selection method"""
        mode_list = ['GP-TS', 'prediction', 'GP-UCB', 'GP-PI', 'GP-EI', 'random']
        mode_gp_ucb = ['beta_1']
        saturation_steps_dict = defaultdict(list)
        reward_history_dict = defaultdict(list)
        material_history_dict = defaultdict(list)
        model_stats_dict = defaultdict(list)
        num_points = len(y)
        data_sample = np.random.choice(y, int(self.v2_sampling_fraction*num_points), replace=False)
        self.Y_avg = np.mean(data_sample)
        self.noise_estimate = np.std(data_sample)/4

        for mode in mode_list:
            print(f'Now running mode {mode}')
            if mode=='GP-UCB':
                for gp_mode in mode_gp_ucb:
                    mode_key=f'{mode}_{gp_mode}'
                    for i, (index, material_system) in enumerate(starting_indices):
                        print(f'Run {i} of model training for mode {mode} and {gp_mode} starting from material system {material_system}')
                        saturation_steps, reward_history, material_history, model_stats = self.run_baseline(X=X, y=y, material_list=material_list, mode=mode, initial_sample=index, gp_mode=gp_mode)
                        saturation_steps_dict[mode_key].append(saturation_steps)
                        reward_history_dict[mode_key].append(reward_history)
                        material_history_dict[mode_key].append(material_history)
                        model_stats_dict[mode_key].append(model_stats)
            else:
                for i, (index, material_system) in enumerate(starting_indices):
                    print(f'Run {i} of model training for mode {mode} starting from material system {material_system}')
                    saturation_steps, reward_history, material_history, model_stats = self.run_baseline(X=X, y=y, material_list=material_list, mode=mode, initial_sample=index)
                    saturation_steps_dict[mode].append(saturation_steps)
                    reward_history_dict[mode].append(reward_history)
                    material_history_dict[mode].append(material_history)
                    model_stats_dict[mode].append(model_stats)

            # Plot model metrics
            if mode!='random':
                fig, ax = plt.subplots()
                mode_key = f'{mode}_beta_1' if mode=='GP-UCB' else mode
                model_stats = model_stats_dict[mode_key][0]
                test_rmse_error = [model_stats[batch]['test_rmse_error'] for batch, _ in model_stats.items()]
                test_r2_error = [model_stats[batch]['test_r2_error'] for batch, _ in model_stats.items()]
                x = list(range(1, len(test_rmse_error)+1))
                ln1 = ax.plot(x, test_rmse_error , label='Test RMSE history', color='black')
                ax.set_ylabel('RMSE')
                ax.set_ylim((2, 6))
                ax.set_xlabel('Steps')
                ax2 = ax.twinx()
                ln2 = ax2.plot(x, test_r2_error , label='Test $R^2$ history', color='g')
                ax2.set_ylabel('$R^2$', color='g')
                ax2.tick_params(axis='y', colors='g')
                ax2.set_ylim((-3, 0.5))
                lns = ln1+ln2
                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs)
                mode = self.map_keys(mode_key)
                ax.set_title(mode)
                fig.savefig(path.join(self.output_dir, f'Supervised_ML_model_metrics_plot_{mode}_{self.args.property_name.replace(" ", "_")}')) # self.output_dir is defined in the subclass SequentialSelection
            plt.close()

        for mode in saturation_steps_dict:
            print(f'For mode=\"{mode}\" the list of steps required are {saturation_steps_dict[mode]}')
            print(f'For mode=\"{mode}\" the mean saturation steps is {np.mean(saturation_steps_dict[mode])} and the standard deviation is {np.std(saturation_steps_dict[mode]):.2f}')

        return saturation_steps_dict, reward_history_dict, material_history_dict
        # Find the number of saturation steps associated with each run and log it
        # For each mode, the run will have to be repeated several times to give a statistical measure of number of saturation steps due to the initial sampling uncertainty

def delete_material_items(material_list, picked_subset):
    return [
        material
        for index, material in enumerate(material_list)
        if index not in picked_subset
    ]