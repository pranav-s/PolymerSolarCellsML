from PolymerSolarCellsML.sequential_selection.contextual_bandit_trainer import ThomSampPolymerSingle
from PolymerSolarCellsML.sequential_selection.baseline_trainer import SampleEfficiencyBaselines
from PolymerSolarCellsML.dataset.dataset_parsing import DatasetParser
from PolymerSolarCellsML.property_prediction.base_classes import PolymerDonorAcceptorBase
from PolymerSolarCellsML.sequential_selection.parse_args import parse_args

import random
import json
import logging
from collections import defaultdict, Counter
import multiprocessing as mp

from time import time

from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm

from typing import Dict

import sys

from os import path
import os
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use('cairo')
mpl.rc('font', family='Liberation Sans', size=18, weight='bold')

import umap

from rdkit import Chem
from rdkit.Chem import Draw

import numpy as np

logger = logging.getLogger()
logging.basicConfig()
logger.setLevel(logging.INFO)

class PSCMaterialSelection(ThomSampPolymerSingle, SampleEfficiencyBaselines): # First time properly using multiple inheritance :D
    def __init__(self, property_metadata, args):
        super(PSCMaterialSelection, self).__init__(args=args)
        self.args = args
        self.property_metadata = property_metadata
        self.dataset_parser = DatasetParser(property_name=args.property_name)
        self.fp_dict_path = '../../metadata/fp_dict.pkl'
        self.donor_acceptor_base = PolymerDonorAcceptorBase()

        self.output_dir = path.join(self.args.figures_dir, f'{self.args.acceptor_type}_{self.args.filtering_criteria}_{self.args.plotting_criteria}')
        self._create_folder(self.output_dir)
    
    def _create_folder(self, folder_path):
        if not path.exists(folder_path):
            logger.info(f'Creating output directory {folder_path}')
            os.makedirs(folder_path)
        else:
            # Delete the contents of the directory
            logger.info(f'Output directory {folder_path} already exists, deleting contents')
            for file in os.listdir(folder_path):
                os.remove(path.join(folder_path, file))
            
    
    def filter_dataset(self, dataset: Dict, add_random_nfa=False):
        """Filter the dataset based on the acceptor type and the filtering_criteria which will pick a single property out of a list of property values"""
        new_dataset = defaultdict(dict)
        max_pce = 0
        for (donor, acceptor), value_dict in dataset.items():
            if (self.args.acceptor_type == 'FA' and not value_dict['fullerene_acceptor']) or (self.args.acceptor_type == 'NFA' and value_dict['fullerene_acceptor']):
                continue
            pce_list = value_dict[self.args.property_name]
            year_list = value_dict['year']
            pce_new, year_new, index = self.generate_filtered_list(pce_list, year_list)
            new_dataset[(donor, acceptor)] = {key: value for key, value in value_dict.items() if key not in ['year', self.args.property_name]}
            new_dataset[(donor, acceptor)]['year'] = year_new[index]
            new_dataset[(donor, acceptor)][self.args.property_name] = pce_new[index]
            max_pce = max(max_pce, pce_new[index])
        
                    
        return new_dataset
    
    def generate_filtered_list(self, pce_list, year_list):
        if self.args.filtering_criteria == 'median':
            pce_new, year_new = zip(*sorted(zip(pce_list, year_list), key=lambda x: x[0]))
            index = self.median_(pce_new)
            
        elif self.args.filtering_criteria == 'earliest':
            pce_new, year_new = zip(*sorted(zip(pce_list, year_list), key=lambda x: x[1]))
            index = 0
        
        elif self.args.filtering_criteria == 'latest':
            pce_new, year_new = zip(*sorted(zip(pce_list, year_list), key=lambda x: x[1]))
            index = -1

        elif self.args.filtering_criteria == 'smallest':
            pce_new, year_new = zip(*sorted(zip(pce_list, year_list), key=lambda x: x[0]))
            index = 0
        
        elif self.args.filtering_criteria == 'largest':
            pce_new, year_new = zip(*sorted(zip(pce_list, year_list), key=lambda x: x[0]))
            index = -1
        
        else:
            raise NotImplementedError(f'Filtering criteria {self.args.filtering_criteria} not implemented')
    
        return pce_new, year_new, index

    
    def median_(self, items):
        """Calculate the median value for a list such that the value is always in the list"""
        return len(items)//2 if len(items) % 2 == 1 else len(items)//2-1

    def construct_feature_matrix(self, dataset, fp_dict):
        """Construct the feature matrix from the dataset"""

        X, y, material_list = [], [], []
        donor_key_set = set()
        acceptor_key_set = set()
        for (donor, acceptor), value_dict in dataset.items():
            X.append((fp_dict[donor], fp_dict[acceptor]))
            y.append(value_dict[self.args.property_name])
            material_list.append((donor, acceptor))
        
        for donor_dict, acceptor_dict in X:
            for key, _ in donor_dict.items():
                donor_key_set.add(key)
            for key, _ in acceptor_dict.items():
                acceptor_key_set.add(key)
        donor_key_set, acceptor_key_set = list(donor_key_set), list(acceptor_key_set)
        X_out = np.zeros((len(X), len(donor_key_set)+len(acceptor_key_set)))
        for i, (donor_dict, acceptor_dict) in enumerate(X):
            # Donor and acceptor vectors concatenated
            for j, key in enumerate(donor_key_set):
                if key in donor_dict:
                    X_out[i][j]=donor_dict[key]
            for j, key in enumerate(acceptor_key_set):
                if key in acceptor_dict:
                    X_out[i][j+len(donor_key_set)]=acceptor_dict[key]
        
        xscale = MinMaxScaler()
        X_out = xscale.fit_transform(np.array(X_out))
        
        return X_out, np.array(y), donor_key_set, acceptor_key_set, material_list

    def augment_fp_dict(self):
        """For donor or acceptor not in the fp_dict, add the corresponding fingerprint"""
        return self.donor_acceptor_base.generate_fp_dict(
                                                         data_dict_path=self.fp_dict_path, 
                                                         )

    def run_contextual_bandits(self, X, y, material_list, starting_index, queue=None):
        # Use the Thompson sampler to run the contextual bandit algorithm
        variance_init = self.args.alpha
        max_step_list, reward_history_list, material_history_list = self.run_multiple_bandits(X, y, material_list, variance_init, starting_index)
        if queue is not None:
            queue.put((max_step_list, reward_history_list, material_history_list))
        else:
            return max_step_list, reward_history_list, material_history_list

    def run_baselines(self, X, y, material_list, starting_index, queue=None):
        # Use the baselines to run the GPR algorithms
        saturation_steps_dict, reward_history_dict, material_history_dict = self.run_all_baselines(X, y, material_list, starting_index)
        if queue is not None:
            queue.put((saturation_steps_dict, reward_history_dict, material_history_dict))
        else:
            return saturation_steps_dict, reward_history_dict, material_history_dict
    
    def sort_ground_truth(self, dataset):
        """Sort the dataset according to time in the format required for plotting"""
        sorted_dataset = dict(sorted(dataset.items(), key=lambda x: x[1]['year']))
        max_property_value = max(
            value[self.args.property_name] for value in sorted_dataset.values()
        )
        for index, (key, value) in enumerate(sorted_dataset.items()):
            if value[self.args.property_name] == max_property_value:
                max_step = index
                break

        reward_history = [value[self.args.property_name] for value in sorted_dataset.values()]
        material_history = [(key[0], key[1]) for key in sorted_dataset]
        year_history = [value['year'] for value in sorted_dataset.values()]

        return max_step, reward_history, material_history, year_history
    
    def select_seed(self, dataset, num_seeds=5):
        """Select initial data points with which to seed the model"""
        datapoints_sorted = sorted(list(enumerate(dataset.keys())), key=lambda x: dataset[x[1]]['year'])
        return datapoints_sorted[:num_seeds] # Return index as well as material pair
    
    def run(self):
        ml_dataset, donor_dict, acceptor_dict, donor_acceptor_dataset = self.dataset_parser.auxiliary_data()
        fp_dict = self.augment_fp_dict()
        # Construct conventional dataset filtering
        filtered_dataset = self.filter_dataset(donor_acceptor_dataset)
        output_data = {'starting_indices': []}

        # Construct X and y
        X, y, donor_key_set, acceptor_key_set, material_list = self.construct_feature_matrix(dataset=filtered_dataset, fp_dict=fp_dict)

        starting_indices = self.select_seed(filtered_dataset, num_seeds=self.args.gp_runs)

        for _, iteration_starting_point in starting_indices:
            logger.info(f'Starting indices: {iteration_starting_point}, details: {filtered_dataset[iteration_starting_point]}')
            output_data['starting_indices'].append(iteration_starting_point)

        # Create two parallel processes to run run_baselines and run_contextual_bandits
        if self.args.run_parallel_paths:
            queue = mp.Queue()
            args = (X, y, material_list, starting_indices, queue)
            process1 = mp.Process(target=self.run_baselines, args=args)
            process2 = mp.Process(target=self.run_contextual_bandits, args=args)
            process1.start()
            process2.start()
            max_steps_dict, reward_history_dict, material_history_dict = queue.get()
            max_steps_array, reward_history_list, material_history_list = queue.get()
            process1.join()
            process2.join()
            if type(material_history_dict)==list:
                # Swap the two sets of variables, needed as it is not known which process will finish first
                max_steps_array, max_steps_dict = max_steps_dict, max_steps_array
                reward_history_list, reward_history_dict = reward_history_dict, reward_history_list
                material_history_list, material_history_dict = material_history_dict, material_history_list
        else:
            # Run the baselines
            max_steps_dict, reward_history_dict, material_history_dict = self.run_baselines(X, y, material_list, starting_indices)

            # Run the contextual bandit algorithm
            max_steps_array, reward_history_list, material_history_list = self.run_contextual_bandits(X, y, material_list, starting_indices)

        max_steps_dict['contextual_bandits'] = max_steps_array
        reward_history_dict['contextual_bandits'] = reward_history_list
        material_history_dict['contextual_bandits'] = material_history_list

        max_steps_ground, reward_history_ground, material_history_ground, year_list = self.sort_ground_truth(filtered_dataset)

        max_steps_dict['ground_truth'] = [max_steps_ground]
        reward_history_dict['ground_truth'] = [reward_history_ground]
        material_history_dict['ground_truth'] = [material_history_ground]
        output_data['material_history'] = material_history_dict
        output_data['reward_history'] = reward_history_dict
        output_data['max_steps'] = max_steps_dict

        # Plot the results
        self.plot_results(X, material_list, filtered_dataset, max_steps_dict, reward_history_dict, material_history_dict, year_list, max_y=np.max(y))

        # Write output_data to a json file
        if self.args.write_output:
            with open(path.join(self.output_dir, 'output_data.json'), 'w') as f:
                json.dump(output_data, f, indent=4)
            run_params = vars(self.args)
            with open(path.join(self.output_dir, 'run_params.json'), 'w') as f:
                json.dump(run_params, f, indent=4)

        
    def plot_results(self, X, material_list, dataset, max_steps_dict, reward_history_dict, material_history_dict, year_list, max_y):
        """Plot the results of the selection algorithms"""
        def barchart(ax, data_dict, plt):
            for key, speedup_list in data_dict.items():
                ax.bar(key, np.mean(speedup_list), yerr=np.std(speedup_list), capsize=3)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
        def violin_plot(ax, data_dict, plt=None):
            data_values = list(data_dict.values())
            keys = list(data_dict.keys())
            # violin_parts = ax.violinplot(data_values, showmeans=False, showmedians=False, showextrema=False)
            # # Customize violin plot colors
            # for pc in violin_parts['bodies']:
            #     pc.set_facecolor('#1f77b4')
            #     pc.set_edgecolor('black')
            #     pc.set_alpha(1)

            # Create box plots
            meanpointprops = dict(marker='x', markeredgecolor='black',
                      markerfacecolor='firebrick', markersize=7)
            boxprops = dict(linestyle='-', linewidth=2, facecolor='tan')
            medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
            box_parts = ax.boxplot(data_values, patch_artist=True, showmeans=True, medianprops=medianprops, whiskerprops=dict(color='black', linewidth=2), capprops=dict(color='black', linewidth=2), boxprops=boxprops, meanprops=meanpointprops)
            ax.yaxis.grid(True)
            # Set the x-ticks to correspond to the keys in the dictionary
            ax.set_xticks(range(1, len(keys) + 1))
            ax.set_xticklabels(keys, rotation=45, ha="right", rotation_mode="anchor")

        self.plot_histogram_year_ratio(max_steps_dict, year_list, plot_type=barchart)
        self.plot_histogram_year_ratio(max_steps_dict, year_list, plot_type=violin_plot)
        self.plot_material_path(X=X, material_list=material_list, material_history_dict=material_history_dict, reward_history_dict=reward_history_dict)
        self.plot_output(reward_history_dict=reward_history_dict,year_list=year_list, max_y=max_y)
        # self.plot_maximum_common_substructure(dataset, material_history_dict, reward_history_dict)
        self.plot_histogram_step_ratio(max_steps_dict)
    
    def select_path(self, reward_history_list):
        """Select the path to plot from the list of paths"""
        if self.args.plotting_criteria == 'earliest':
            return reward_history_list[0]
        # Sort the paths by length
        reward_history_list.sort(key=lambda x: len(x))
        if self.args.plotting_criteria == 'median':
            index = self.median_(reward_history_list)
        elif self.args.plotting_criteria == 'shortest':
            index = 0
        elif self.args.plotting_criteria == 'longest':
            index = -1
        
        return reward_history_list[index]
    
    def plot_output(self, reward_history_dict, year_list, max_y):
        """Plot the successive rewards of the entries picked by the algorithm"""
        plt.close()
        fig, ax = plt.subplots()
        keep_list = ['prediction', 'ground_truth']
        ax.plot(year_list, [max_y]*len(reward_history_dict['ground_truth'][0]), '--', label='Maximum value')
        year_index = reward_history_dict['ground_truth'][0].index(max_y)
        
        for key, value_list in reward_history_dict.items():
            if key not in keep_list:
                continue
            reward_history = self.select_path(value_list)
            speedup_fraction = (year_list[len(reward_history)-1]-year_list[0])/(year_list[year_index]-year_list[0])
            logger.info(f'Speedup fraction for {key}: {speedup_fraction}')
            key = self.map_keys(key)
            ax.plot(year_list[:len(reward_history)], reward_history, label=key)
        ax.set_xlabel('Year')
        # Set xticklabels to be every 4 years
        min_year = int(year_list[0])
        max_year = int(year_list[-1])+1
        ax.set_xticks(list(range(min_year, max_year+1, 3)))
        ax.set_ylim((0, 17)) # Hardcoded for now as 17 is the maximum value of PCE in the dataset
        ax.set_ylabel(f'{self.args.property_name.title()} ({self.property_metadata[self.args.property_name]["unit"]})')
        ax.legend(loc='upper center')
        title_map = {'FA': 'Fullerene acceptor', 'NFA': 'Non-fullerene acceptor', 'both': 'Fullerene and non-fullerene acceptor'}
        ax.set_title(f'{title_map[self.args.acceptor_type]}')
        fig.savefig(path.join(self.output_dir, f'reward_plot_comparison_{self.args.acceptor_type}.png'), format='png')
        plt.close()
    
    def plot_structure_history(self, material_history_dict, reward_history_dict, dataset):
        """Draw the sequence of structures for which the reward increases"""

        # Iterate over all keys of the material_history_dict and draw the corresponding molecules and save a figure for each
        # Find the set of materials for which the reward increases
        exclude_key_list = ['random', 'GP-UCB_beta_0.5', 'GP-UCB_beta_1.5', 'GP-UCB_beta_decay_exp_0.25', 'GP-EI', 'GP-TS', 'GP-alt']
        for key, value_list in reward_history_dict.items():
            max_value = 0
            increasing_reward_history_materials = []
            if key in exclude_key_list:
                continue
            reward_history = self.select_path(value_list)
            material_history = self.select_path(material_history_dict[key])

            for index, value in enumerate(reward_history):
                if value > max_value:
                    max_value = value
                    increasing_reward_history_materials.append((index, material_history[index]))

            increasing_reward_history_materials = [(step, material_pair) for step, material_pair in increasing_reward_history_materials if '{' not in dataset[material_pair]['donor_smiles'] and '{' not in dataset[material_pair]['acceptor_smiles']] # Since we can't do a satisfactory visualization of copolymer structures, we will skip them
            plt.subplots_adjust(wspace=0.1, hspace=0.25)
            fig, ax = plt.subplots(nrows=len(increasing_reward_history_materials), ncols=2)
            fontsize = 6 if len(increasing_reward_history_materials) > 10 else 8
            figsize = 400
            for reward_index, (step, (donor, acceptor)) in enumerate(increasing_reward_history_materials):
                donor_smiles = dataset[(donor, acceptor)]['donor_smiles']
                acceptor_smiles = dataset[(donor, acceptor)]['acceptor_smiles']
                # if '{' in donor_smiles or '{' in acceptor_smiles: # Since we can't do a satisfactory visualization of copolymer structures, we will skip them
                #     continue
                donor_mol = Chem.MolFromSmiles(donor_smiles)
                donor_image = Draw.MolToImage(donor_mol, size=(figsize, figsize))
                acceptor_mol = Chem.MolFromSmiles(acceptor_smiles)
                acceptor_image = Draw.MolToImage(acceptor_mol, size=(figsize, figsize))
                self._structure_plot_configuration(
                    ax, reward_index, idx=0, image=donor_image
                )
                if len(donor)>20:
                    donor = f'{donor[:20]}...'
                if len(acceptor)>20:
                    acceptor = f'{acceptor[:20]}...'
                ax[reward_index][0].annotate(donor, xy=(1,-3), size=fontsize)
                ax[reward_index][0].annotate(f'Step {step} {self.property_metadata[self.args.property_name]["short_name"]} = {reward_history[step]:.2f} {self.property_metadata[self.args.property_name]["unit"]}', xy=(400, 250), size=fontsize)
                self._structure_plot_configuration(
                    ax, reward_index, idx=1, image=acceptor_image
                )
                ax[reward_index][1].annotate(acceptor, xy=(1,-3), size=fontsize)
            fig.savefig(path.join(self.output_dir, f'structure_plot_{key}.eps'), format='eps', dpi=500)
            plt.close()

    def _structure_plot_configuration(self, ax, reward_index, idx, image):
        ax[reward_index][idx].imshow(image)
        ax[reward_index][idx].xaxis.set_visible(False)
        ax[reward_index][idx].yaxis.set_visible(False)
        ax[reward_index][idx].spines['top'].set_linewidth(1)
        ax[reward_index][idx].spines['bottom'].set_linewidth(1)
        ax[reward_index][idx].spines['left'].set_linewidth(1)
        ax[reward_index][idx].spines['right'].set_linewidth(1)
        ax[reward_index][idx].tick_params(left = False, right = False , labelleft = False,
                                     labelbottom = False, bottom = False)

    def plot_maximum_common_substructure(self, dataset, material_history_dict, reward_history_dict):
        """Plot the maximum common substructure of the molecules in the dataset"""
        from rdkit import Chem
        from rdkit.Chem import rdFMCS
        MAX_COMPARE = 10
        top_k = 3
        exclude_key_list = ['random', 'GP-UCB_beta_0.5', 'GP-UCB_beta_1.5', 'GP-UCB_beta_decay_exp_0.25', 'GP-alt']
        for key, material_history_list in material_history_dict.items():
            donor_list = []
            acceptor_list = []
            if key in exclude_key_list:
                continue
            material_history = self.select_path(material_history_list)[:MAX_COMPARE]
            reward_history = self.select_path(reward_history_dict[key])[:MAX_COMPARE]
            logger.info(f'Material history for {key} has length {len(material_history)}: {material_history}\n')
            increasing_reward_history_materials = []
            max_value = 0
            for index, value in enumerate(reward_history):
                if value > max_value:
                    max_value = value
                    increasing_reward_history_materials.append((index, material_history[index]))
            material_history = [material_pair for step, material_pair in increasing_reward_history_materials if '{' not in dataset[material_pair]['donor_smiles'] and '{' not in dataset[material_pair]['acceptor_smiles']] # Since we can't do a satisfactory visualization of copolymer structures, we will skip them
            fig, ax = plt.subplots(nrows=2, ncols=3)
            # limit = min(len(material_history), MAX_COMPARE)
            limit = len(material_history)
            logger.info(f'Material history for {key} has length {len(material_history)}: {material_history}\n')
            for i in range(1, limit):
                donor_smiles_i = dataset[material_history[i]]['donor_smiles']
                acceptor_smiles_i = dataset[material_history[i]]['acceptor_smiles']
                donor_mol_i = Chem.MolFromSmiles(donor_smiles_i)
                acceptor_mol_i = Chem.MolFromSmiles(acceptor_smiles_i)
                for j in range(i):
                    donor_smiles_j = dataset[material_history[j]]['donor_smiles']
                    acceptor_smiles_j = dataset[material_history[j]]['acceptor_smiles']
                    if donor_smiles_i != donor_smiles_j:
                        donor_mol_j = Chem.MolFromSmiles(donor_smiles_j)
                        start = time()
                        donor_mcs = rdFMCS.FindMCS([donor_mol_i, donor_mol_j], timeout=20)
                        end = time()
                        logger.info(f'Time taken to find MCS for donors {material_history[i][0]} and {material_history[j][0]}: {(end-start):.3f} s')
                        donor_list.append(donor_mcs.smartsString)
                    if acceptor_smiles_i != acceptor_smiles_j:
                        acceptor_mol_j = Chem.MolFromSmiles(acceptor_smiles_j)
                        start = start = time()
                        acceptor_mcs = rdFMCS.FindMCS([acceptor_mol_i, acceptor_mol_j], timeout=15)
                        end = time()
                        logger.info(f'Time taken to find MCS for acceptors {material_history[i][1]} and {material_history[j][1]}: {(end-start):.3f} s')
                        acceptor_list.append(acceptor_mcs.smartsString)

            # Top K most common substructures in donor_list and acceptor_list
            donor_counter = Counter(donor_list)
            acceptor_counter = Counter(acceptor_list)
            logger.info(f'Donor counter: {donor_counter}')
            logger.info(f'Acceptor counter: {acceptor_counter}')
            if len(donor_counter) < top_k or len(acceptor_counter) < top_k:
                logger.info(f'Not enough unique substructures for {key}, skipping')
                continue
            top_donor = donor_counter.most_common(top_k)
            top_acceptor = acceptor_counter.most_common(top_k)
            for i in range(top_k):
                donor_mcs = Chem.MolFromSmarts(top_donor[i][0])
                acceptor_mcs = Chem.MolFromSmarts(top_acceptor[i][0])
                donor_image = Draw.MolToImage(donor_mcs)
                acceptor_image = Draw.MolToImage(acceptor_mcs)
                self._structure_plot_configuration(ax, reward_index=0, idx=i, image=donor_image)
                self._structure_plot_configuration(ax, reward_index=1, idx=i, image=acceptor_image)
            ax[0][0].set_title('Donors maximum common substructures', fontsize=16, fontweight='bold')
            ax[1][0].set_title('Acceptors maximum common substructures', fontsize=16, fontweight='bold')
            fig.savefig(path.join(self.output_dir, f'mcs_donor_acceptor_{key}.png'))
            

    def plot_histogram_year_ratio(self, max_steps_dict, year_list, plot_type):
        """Compare the mean steps to find the desired property across settings"""
        fig, ax = plt.subplots()
        include_list = ['GP-PI', 'GP-EI', 'GP-UCB_beta_1', 'prediction', 'contextual_bandits', 'GP-TS', 'random']
        max_year_index = max_steps_dict['ground_truth'][0]
        filtered_dict = dict()
        for key, value_list in max_steps_dict.items():
            if key in include_list:
                speedup_list = [(year_list[max_year_index]-year_list[0])/(year_list[item]-year_list[0]) for item in value_list if item>3] # Arbitrary lower threshold introduced to prevent high variance values from dominating the plot
                key = self.map_keys(key)
                filtered_dict[key] = speedup_list
                # ax.bar(key, np.mean(speedup_list), yerr=np.std(speedup_list), capsize=3)
        plot_type(ax, filtered_dict, plt) # Make a barchart or violin plot with box plot depending on the plot_type that is injected
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_ylabel('Discovery time speedup factor')
        ax.set_ylim((0, 5))
        fig.savefig(path.join(self.output_dir, f'final_comparison_years_plot_{plot_type.__name__}_{self.args.property_name.replace(" ", "_")}_{self.args.acceptor_type}'), bbox_inches="tight")
        plt.close()
    
    def plot_histogram_step_ratio(self, max_steps_dict):
        """Compare the mean steps to find the desired property across settings"""
        fig, ax = plt.subplots()
        include_list = ['GP-PI', 'GP-EI', 'GP-UCB_beta_1', 'prediction', 'contextual_bandits', 'GP-TS', 'random']
        for key, value_list in max_steps_dict.items():
            if key in include_list:
                speedup_list = [max_steps_dict['ground_truth'][0]/item for item in value_list if item>3] # Arbitrary lower threshold introduced to prevent high variance values from dominating the plot
                key = self.map_keys(key)
                ax.bar(key, np.mean(speedup_list), yerr=np.std(speedup_list), capsize=3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_ylabel('Discovery steps speedup ratio')
        fig.savefig(path.join(self.output_dir, f'final_comparison_time_plot_{self.args.property_name.replace(" ", "_")}_{self.args.acceptor_type}'), bbox_inches="tight")
        plt.close()
    
    def find_clusters(self, umap, material_list):
        """Find the materials corresponding to certain clusters in the plot"""
        cluster_points = defaultdict(list)
        TOP_K = 10
        for index, (x, y) in enumerate(umap):
            if x < -5:
                cluster_points[0].append(material_list[index])
            elif x> 5:
                cluster_points[1].append(material_list[index])
            elif x>-5 and x<5 and y>8:
                cluster_points[2].append(material_list[index])
            elif x>-5 and x<5 and y<6:
                cluster_points[3].append(material_list[index])
        direction = ['left', 'right', 'top', 'bottom']
        for key, value in cluster_points.items():
            logger.info(f'Cluster {key} for {direction[key]}: {value[:TOP_K]}')
    
    def plot_material_path(self, X, material_list, material_history_dict, reward_history_dict):
        """Plot the path taken by different methods in embedding space and show the actual structure on the material on the same plot for clarity. Compare against ground truth"""
        model_list = ['GP-TS', 'GP-PI', 'GP-EI', 'GP-UCB_beta_1', 'prediction', 'contextual_bandits']
        marker_size=20
        # Generate 2-D embedding of the material embeddings
        reducer = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.1, random_state=42)

        # Fit and transform the data
        umap_result = reducer.fit_transform(X)

        self.find_clusters(umap_result, material_list)

        # Plot the path taken by the algorithm in the embedding space
        for key, value_list in material_history_dict.items():
            total_steps = len(value_list[0])
            if key not in model_list:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            cmap = cm.get_cmap('cool')
            ax.scatter(umap_result[:, 0], umap_result[:, 1], c='gray', label='Data Points', s=marker_size)

            x_list, y_list = [], []
            reward_array = []
            # Need to systematically downsample the points to make the plot more readable
            for step, material_pair in enumerate(value_list[0]):
                    idx = material_list.index(material_pair)
                    x_list.append(umap_result[idx][0])
                    y_list.append(umap_result[idx][1])
                    reward_array.append(reward_history_dict[key][0][step])
                    ax.annotate(str(step+1), (umap_result[idx][0]+0.2, umap_result[idx][1]-0.2), size=7)
            sc = ax.scatter(x_list, y_list, c=reward_array, cmap=cmap, marker='o', s=marker_size)
            key = self.map_keys(key)
            ax.plot(x_list, y_list, label=key, c='gray', linestyle='--')
            # Create a color bar to display the value scale
            cbar = fig.colorbar(sc, orientation='vertical')
            cbar.set_label('Power Conversion Efficiency (%)')
            ax.set_xlabel('UMAP Dimension 1')
            ax.set_ylabel('UMAP Dimension 2')
            ax.set_title(key)

            fig.savefig(path.join(self.output_dir, f'material_path_plot_{key.replace(".", "_")}_{self.args.property_name.replace(" ", "_")}_{self.args.acceptor_type}'), bbox_inches="tight")

            plt.close()

    def map_keys(self, key):
        """Map names of certain keys to what we would like it to be in the paper"""
        out_key = key
        key_map = {'prediction': 'Greedy',
                   'GP-UCB_beta_decay_exp_0.25': 'GP-UCB',
                   'GP-UCB_beta_1': 'GP-UCB',
                   'contextual_bandits': 'Contextual Bandits',
                   'ground_truth': 'Experimental data',
                   'random': 'Random'}
        if key in key_map:
            out_key = key_map[key]
        return out_key
    
    def plot_saturation(self, reward_dict, steps_dict):
        """Plot the steps to saturation for each variance value"""
        plt.close()
        fig, ax = plt.subplots()
        steps_mean, steps_std = [], []
        for _, value in steps_dict.items():
            steps_mean.append(value["mean"])
            steps_std.append(value["std"])
        
        rewards_mean, rewards_std = [], []
        
        for _, value in reward_dict.items():
            rewards_mean.append(value["mean"])
            rewards_std.append(value["std"])
        
        ax.errorbar(steps_dict.keys(), steps_mean, yerr=steps_std, capsize=3, fmt='o')
        ax.set_xlabel('Variance')
        ax.set_ylabel('Steps to maximum reward')

        fig.savefig(path.join(self.output_dir, f'saturation_steps_plot_{args.property_name.replace(" ", "_")}'))
        plt.close()

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    args = args[0]
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    property_metadata = {'power conversion efficiency': {'unit': '%', 'v2': 0.1, 'short_name': 'PCE'},
                        }
    psc_material_selector = PSCMaterialSelection(property_metadata, args)
    psc_material_selector.run()