from collections import defaultdict
from sklearn.preprocessing import StandardScaler

import numpy as np

from base_classes import PolymerSingleMaterialBase, PolymerDonorAcceptorBase

from PolymerSolarCellsML.dataset.dataset_parsing import DatasetParser

import argparse

import matplotlib.pyplot as plt

import seaborn as sns

from os import path
import os

import logging

logger = logging.getLogger()
logging.basicConfig()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--use_median",
    help="Use median of dataset if multiple datapoints with the same value are found. Otherwise consider all values for each material system as a separate datapoint.",
    action="store_true",
)

parser.add_argument(
    "--use_log_transform",
    help="Use log transform to transform the property values. Not used by default.",
    action="store_true",
)

parser.add_argument(
    "--property_name",
    help="Name of property to build models for",
    default='power conversion efficiency'
)
parser.add_argument(
    "--output_dir",
    help="Name of directory where output figures are stored",
    default='../../output/'
)

class DonorAcceptorModel(PolymerDonorAcceptorBase):
    def __init__(self, dataset_parser_output, use_median, property_name, output_dir) -> None:
        super(DonorAcceptorModel, self).__init__()
        self.top_predictions = 10
        self.dataset_parser_output = dataset_parser_output
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.fp_dict_path = '../../metadata/fp_dict.pkl'
        self.use_median = use_median
        self.property_name = property_name

    def find_prop_value(self, ml_dataset, input_donor, input_acceptor):
        """Find the PCE value given donor and acceptor"""
        return next(
            (
                (np.median(prop_value_list), doi_list)
                for donor, donor_smile, acceptor, acceptor_smile, prop_value_list, doi_list in ml_dataset
                if donor == input_donor and acceptor == input_acceptor
            ),
            (None, None),
        )
    
    def plot_grid(self, donor_dict, acceptor_dict, donor_acceptor_pairs, ml_dataset):
        """Plot 2-D grid of all donors Vs acceptors and verify relative frequency of each"""
        # Plot actual median PCE instead of a binary index for these points which will reveal certain trends
        print(f'Number of datapoints available: {len(ml_dataset)}')
        # nan_color = '#d3d3d3'
        donor_list = list(donor_dict.keys())
        acceptor_list = list(acceptor_dict.keys())

        donor_acceptor_frequencies = np.empty((len(acceptor_list), len(donor_list)))
        donor_acceptor_frequencies.fill(np.nan)
        nonzero_count = 0
        donor_count = defaultdict(int)
        acceptor_count = defaultdict(int)
        for column, donor in enumerate(donor_list):
            for row, acceptor in enumerate(acceptor_list):
                if (donor, acceptor) in donor_acceptor_pairs:
                    donor_acceptor_frequencies[row][column] = np.median(donor_acceptor_pairs[(donor, acceptor)][self.property_name]) #donor_organic_acceptor_pairs[(donor, acceptor)]
                    donor_count[(donor, column)]+=1
                    acceptor_count[(acceptor, row)]+=1
                    nonzero_count+=1

        print(f'Fraction of donor acceptor space explored = {(nonzero_count*100/(len(donor_list*len(acceptor_list)))):.2f} %')
        donor_count_sorted = sorted(donor_count.items(), key = lambda x: x[1], reverse=True)
        donor_count_sorted = [(*key, value) for key, value in donor_count_sorted]
        acceptor_count_sorted = sorted(acceptor_count.items(), key = lambda x: x[1], reverse=True)
        acceptor_count_sorted = [(*key, value) for key, value in acceptor_count_sorted]
        print(f'Top {self.top_predictions} donors = {list(donor_count_sorted)[:self.top_predictions]}')
        print(f'Top {self.top_predictions} acceptors = {list(acceptor_count_sorted)[:self.top_predictions]}')
        self._plot_heatmap(
            donor_acceptor_frequencies, 'donor_acceptor_frequencies.png', donor_count_sorted, acceptor_count_sorted, print_top_k=4
        )
        return donor_list, acceptor_list
    
    def donor_acceptor_model(self, ml_dataset, donor_dict, acceptor_dict, use_log_transform=False):
        donor_list, acceptor_list = list(donor_dict.keys()), list(acceptor_dict.keys())
        train_set, test_set, train_test_mask = self.train_test_split(ml_dataset, train_frac=0.85)
        fp_dict = self.generate_fp_dict(self.fp_dict_path)
        X_train, y_train = self.fingerprint_data(train_set, fp_dict, self.use_median, use_log_transform=use_log_transform)
        X_test, y_test = self.fingerprint_data(test_set, fp_dict, self.use_median, use_log_transform=use_log_transform)
        donor_key_set, acceptor_key_set, X_train, X_test = self.vectorize_dict_features(X_train, X_test)

        xscale = StandardScaler()
        X_train = xscale.fit_transform(X_train)
        X_test = xscale.transform(X_test)

        model = self.train_ml_model(X_train, y_train)

        tr_error, tt_error, k_largest_errors = self.prediction(model, X_train, X_test, y_train, y_test, verbose=False, use_log_transform=use_log_transform)

        print(f'Model performance, train error = {tr_error:.3f}, test_error = {tt_error:.3f}')

        print('The predictions with largest error are given below:')

        for index in k_largest_errors:
            print(f'test set point = {test_set[index]}')

        self.single_parity_plot(model, self.abbreviation[self.property_name], '%',  X_train, X_test, y_train, y_test, 'parity_plot_donor_acceptor', self.output_dir, use_log_transform=use_log_transform)

        X_out = self.donor_acceptor_features(donor_list, acceptor_list, fp_dict)
        X_out = self.vectorize_features_test(X_out, donor_key_set, acceptor_key_set)
        X_out = xscale.transform(X_out)
        y_grid = model.predict(X_out)
        if use_log_transform:
            y_grid = np.exp(y_grid)

        y_out = np.zeros((len(acceptor_list), len(donor_list)))
        for j in range(len(donor_list)):
            for i in range(len(acceptor_list)):
                y_out[i][j] = y_grid[j*len(acceptor_list)+i]
        print(f'Number of acceptors = {y_out.shape[0]}')
        print(f'Number of donors = {y_out.shape[1]}')

        # Find top donors and acceptors going by average values

        donor_avg = np.mean(y_out, axis=0)
        acceptor_avg = np.mean(y_out, axis=1)

        donor_avg_sorted = sorted(zip(donor_list, list(range(len(donor_list))), donor_avg), key = lambda x: x[2], reverse=True)

        acceptor_avg_sorted = sorted(zip(acceptor_list, list(range(len(acceptor_list))), acceptor_avg), key = lambda x: x[2], reverse=True)

        print(f'Top {self.top_predictions} donors = {donor_avg_sorted[:self.top_predictions]}')
        print(f'Top {self.top_predictions} acceptors = {acceptor_avg_sorted[:self.top_predictions]}')

        # Find top predicted donor, acceptor combinations

        self._plot_heatmap(
            y_out, 'donor_acceptor_predicted_pce_V2.png', donor_avg_sorted, acceptor_avg_sorted
        )
        
        total_num_points = len(donor_list)*len(acceptor_list)

        top_indices = np.argpartition(y_out, kth=total_num_points-self.top_predictions, axis=None)[(total_num_points-self.top_predictions):]

        # For each index, convert to row column form, obtain the donor, acceptor and find the PCE and make the figure for those donors and acceptors as well?

        # Create a variable for num_donors and num_acceptors
        top_predicted_pairs = []

        for index in top_indices:
            row = int(index/len(donor_list))
            column = int(index%len(donor_list))
            donor = donor_list[column]
            acceptor = acceptor_list[row]
            prop_value, doi_list = self.find_prop_value(ml_dataset, donor, acceptor)
            predicted_prop_value = y_out[row][column]
            top_predicted_pairs.append((donor, acceptor, predicted_prop_value, prop_value, doi_list))

        # sort top predicted pairs by predicted PCE
        top_predicted_pairs = sorted(top_predicted_pairs, key=lambda x: x[2], reverse=True)

        print(f'Top predicted donor acceptor pairs are {top_predicted_pairs}')

        return fp_dict, train_test_mask

    def _plot_heatmap(self, pce, file_name, donor_count, acceptor_count, print_top_k=3):
        """Plot heatmap of PCE values for all donor acceptor pairs"""
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.set(font_scale=2)
        heatmap = sns.heatmap(
            pce,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
            cmap='coolwarm',
            center=0,
        )
        ax.set_xlabel('Donor', fontsize=18, labelpad=3, fontweight="bold")
        ax.set_ylabel('Acceptor', fontsize=18, labelpad=3, fontweight="bold")
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=15)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, fontsize=12)
        offset = 0.13
        ax.arrow(offset, 0.04, 1-offset*3, 0, transform=plt.gcf().transFigure, clip_on=False,
          head_width=0.02, head_length=0.01, fc='black', ec='black')
        ax.arrow(0, offset, 0, 1-offset*2, transform=plt.gcf().transFigure, clip_on=False,
          head_width=0.01, head_length=0.02, fc='black', ec='black')
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('PCE (%)', fontsize=20)
        cbar.ax.tick_params(labelsize=18)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor('black')
        ax.set_xticks(ticks=[donor_count[i][1] for i in range(print_top_k)], labels=[donor_count[i][0] for i in range(print_top_k)], fontsize=14)
        ax.set_yticks(ticks=[acceptor_count[i][1] for i in range(print_top_k)], labels=[acceptor_count[i][0] for i in range(print_top_k)], fontsize=14)
        fig.savefig(
                    path.join(self.output_dir, file_name), format='png', bbox_inches='tight'
                   )
        plt.close()
        
    def donor_acceptor_features(self, donor_list, acceptor_list, fp_dict):
        """Use trained model to predict on donor acceptor pairs not in training dataset"""
        X = []
        for donor in donor_list:
            X.extend((fp_dict[donor], fp_dict[acceptor]) for acceptor in acceptor_list)
        return X
    
    def vectorize_features_test(self, X, donor_key_set, acceptor_key_set):
        X_out = np.zeros((len(X), len(donor_key_set)+len(acceptor_key_set)))
        for i, (donor_dict, acceptor_dict) in enumerate(X):
            if donor_dict and acceptor_dict:
                # Donor and acceptor vectors concatenated
                for j, key in enumerate(donor_key_set):
                    if key in donor_dict:
                        X_out[i][j]=donor_dict[key]
                for j, key in enumerate(acceptor_key_set):
                    if key in acceptor_dict:
                        X_out[i][j+len(donor_key_set)]=acceptor_dict[key]
        
        return X_out

    def sort_dict(self, donor_dict, acceptor_dict, donor_acceptor_pairs):
        """Arrange each dictionary such that the most frequent donors and acceptors are even distributed over the keys instead of being clumped together"""
        donor_list = list(donor_dict.keys())
        acceptor_list = list(acceptor_dict.keys())
        donor_count = defaultdict(int)
        acceptor_count = defaultdict(int)
        l2_donor_index = -1
        for column, donor in enumerate(donor_list):
            if donor=='L2':
                l2_donor_index = column
            for row, acceptor in enumerate(acceptor_list):
                if (donor, acceptor) in donor_acceptor_pairs:
                    donor_count[(donor, column)]+=1
                    acceptor_count[(acceptor, row)]+=1
        donor_count_sorted = sorted(donor_count.items(), key = lambda x: x[1], reverse=True)
        acceptor_count_sorted = sorted(acceptor_count.items(), key = lambda x: x[1], reverse=True)
        top_k = 4
        idx_start = 20
        idx_end = len(donor_count_sorted)
        # Swap the top k donors evenly across the key list
        for i in range(top_k):
            (donor, idx), count = donor_count_sorted[i]
            swap_idx = (idx_end - idx_start)*(i)//top_k+idx_start
            donor_list[idx], donor_list[swap_idx] = donor_list[swap_idx], donor_list[idx]
        
        donor_list[l2_donor_index], donor_list[idx_end//2] = donor_list[idx_end//2], donor_list[l2_donor_index]
        idx_end = len(acceptor_count_sorted)
        for i in range(top_k):
            (donor, idx), count = acceptor_count_sorted[i]
            swap_idx = (idx_end - idx_start)*(i)//top_k+idx_start
            acceptor_list[idx], acceptor_list[swap_idx] = acceptor_list[swap_idx], acceptor_list[idx]

        # Create a new dictionary with reordered keys
        donor_dict_new = {donor: donor_dict[donor] for donor in donor_list}
        acceptor_dict_new = {acceptor: acceptor_dict[acceptor] for acceptor in acceptor_list}

        return donor_dict_new, acceptor_dict_new
        
    def run(self, use_log_transform=False):
        # Show some statistics of the dataset
        ml_dataset, donor_dict, acceptor_dict, donor_acceptor_pairs = self.dataset_parser_output

        donor_dict, acceptor_dict = self.sort_dict(donor_dict, acceptor_dict, donor_acceptor_pairs)

        self.plot_grid(donor_dict, acceptor_dict, donor_acceptor_pairs, ml_dataset)

        fp_dict, train_test_mask = self.donor_acceptor_model(ml_dataset, donor_dict, acceptor_dict, use_log_transform=use_log_transform)

        return fp_dict, train_test_mask

class DonorModel(PolymerSingleMaterialBase):

    def __init__(self, ml_dataset, fp_dict, use_median, property_name, output_dir) -> None:
        super(DonorModel, self).__init__()
        self.ml_dataset = ml_dataset
        self.fp_dict = fp_dict
        self.use_median = use_median
        self.property_name = property_name
        self.output_dir = output_dir
    
    def filter_dataset(self):
        donor_dataset = []
        for donor, donor_smile, acceptor, acceptor_smile, prop_value_list, doi_list in ml_dataset:
            if donor_smile and fp_dict.get(donor, ''):
                if self.use_median:
                    donor_dataset.append((donor, self.fp_dict[donor], np.median(prop_value_list)))
                else:
                    donor_dataset.extend(
                        (donor, self.fp_dict[donor], prop_value)
                        for prop_value in prop_value_list
                    )
            else:
                logger.info(f'Donor {donor} not in fp_dict or donor_smile is empty')

        return donor_dataset
    
    def donor_pce_model(self, train_test_mask, use_log_transform=False):
        """Train a model to predict PCE's based on donor structure only either include fullerene label or don't"""

        donor_dataset = self.filter_dataset()
        print(f'Number of datapoints = {len(donor_dataset)}')

        # Can store and analyze data_dict
        
        # Split into train and test
        print('Splitting data')
        train_set, test_set, _ = self.train_test_split(donor_dataset, train_frac=0.85, binary_mask=train_test_mask)
        print('Fingerprinting data')
        # Fingerprint and vectorize features
        X_train, y_train = self.fingerprint_data(train_set, use_log_transform=use_log_transform)
        X_test, y_test = self.fingerprint_data(test_set, use_log_transform=use_log_transform)

        key_set, X_train, X_test = self.vectorize_dict_features(X_train, X_test)
        
        xscale = StandardScaler()
        X_train = xscale.fit_transform(X_train)
        X_test = xscale.transform(X_test)
        print(X_train.shape)
        model = self.train_ml_model(X_train, y_train)

        tr_error, tt_error, k_largest_errors = self.prediction(model, X_train, X_test, y_train, y_test, verbose=False, use_log_transform=use_log_transform)

        print(f'Model performance, train error = {tr_error:.3f}, test_error = {tt_error:.3f}')
        print(f'Max {self.property_name} = {np.max(y_train)}')
        print(f'Min {self.property_name} = {np.min(y_train)}')
        self.single_parity_plot(model, self.abbreviation[self.property_name], '%',  X_train, X_test, y_train, y_test, 'parity_plot', self.output_dir, use_log_transform=use_log_transform)


if __name__ == '__main__':
    args = parser.parse_args()

    dataset_parser = DatasetParser(args.property_name)
    dataset_parser_output = dataset_parser.data_statistics()
    ml_dataset, donor_dict, acceptor_dict, donor_acceptor_pairs = dataset_parser.auxiliary_data()

    dataset_parser_output = ml_dataset, donor_dict, acceptor_dict, donor_acceptor_pairs

    donor_acceptor_model = DonorAcceptorModel(dataset_parser_output, args.use_median, args.property_name, args.output_dir)
    fp_dict, train_test_mask = donor_acceptor_model.run(args.use_log_transform)

    donor_model = DonorModel(ml_dataset, fp_dict, args.use_median, args.property_name, args.output_dir) # Use fp_dict as input
    donor_model.donor_pce_model(train_test_mask, args.use_log_transform)
