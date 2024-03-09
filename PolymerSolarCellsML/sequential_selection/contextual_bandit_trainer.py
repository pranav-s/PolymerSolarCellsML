import random
import numpy as np
from os import path

import matplotlib as mpl
mpl.use('Cairo')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from IPython.display import SVG

from PolymerSolarCellsML.utils import config_plots
config_plots(mpl)


def moltosvg(mol, molSize = (100,100), kekulize = True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return SVG(svg)

class ThomSampPolymerSingle:
    def __init__(self, args, **kwargs):
        """
            See Algorithm 1 and section 2.2 from paper:
            "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
            n_arms: int, the number of different arms/ actions the algorithm can take
            features: list of strings, contains the patient features to use
            alpha: float, hyperparameter for step size.
        
        """
        super(ThomSampPolymerSingle, self).__init__(args=args)
        self.args = args
        self.time_steps = 300 # Arbitrary number chosen for test purposes, can also pick as a fraction of the total dataset
        self.features_to_plot = 10

        
    def model_params(self, d):
        self.B = np.identity(d)
        self.f = np.zeros(d)
        self.mu = np.zeros(d)
    
    def choose(self, X, v2, count):
        """
        See Algorithm 1 and section 2.2 from paper:
            "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
            X: Numpy matrix containing the material fingerprints
        Returns:
            output: int containing the selected material

        """


        # Only when choosing the arm very first time do the below quantities need to be computed,
        # the update step has it pre-computed for all future steps
        if not hasattr(self, 'B_inv') or not hasattr(self, 'mu'):
            self.B_inv = np.linalg.inv(self.B)
            self.mu = np.matmul(self.B_inv, self.f)

        self.mu_sample = np.random.multivariate_normal(self.mu, v2*self.B_inv)
        arm_values = np.matmul(X, self.mu_sample)
        return np.argmax(arm_values)

    def update(self, X, a, r):
        """
        See Algorithm 1 and section 2.2 from paper:
            "Thompson Sampling for Contextual Bandits with Linear Payoffs"
            
        Args:
            x: Dictionary containing the possible patient features.
            a: int, indicating the action/index of material your algorithm chose
            r: the reward you recieved for that action
        Returns:
            Nothing

        """

        self.B = self.B+np.outer(X[a], X[a])
        self.f = self.f+r*X[a]
        self.B_inv = np.linalg.inv(self.B)
        self.mu = np.matmul(self.B_inv, self.f)

    def matrix_visualizer(self, covariance, key_list):
        """Visualize the covariance matrix being updated as part of the Thompson sampling process"""
        # The objective is to analyze what physics is being learned as part of the learning process
        # Get the top K submatrix based on the diagonal values
        top_components = 10
        diagonals = np.diagonal(covariance)
        max_indices = np.argsort(diagonals)[-top_components:] # Assume most important components have large diagonal value
        max_indices.sort()
        # Construct the matrix
        sub_matrix = np.array([[covariance[index_r][index_c] for index_c in max_indices] for index_r in max_indices])
        
        # Get the names of the corresponding fields
        fields = [key_list[i] for i in max_indices]
  
        # Create a visualization for the same
        plt.close()
        fig, ax = plt.subplots()
        heatmap = sns.heatmap(sub_matrix, xticklabels=fields, yticklabels=fields, ax=ax)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation = 0, fontsize = 12)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 90, fontsize = 12)
        fig.savefig(path.join(self.args.figures_dir, f'feature_heatmap_plot_{self.v2:.2f}_{self.args.property_name.replace(" ", "_")}.png'), format='png')
        plt.close()

    
    def saturation_checker(self, num_list):
        sat_limit = 0.1
        max_val = stats.mode(num_list[-int(self.time_steps*sat_limit):]) # Assumption is that the value saturates in the last sat_limit fraction of the run
        return num_list.index(max_val.mode[0])
    
    def run_bandits(self,X, y, material_list, v2, set_random_seed, initial_arm=None):
        if set_random_seed:
            np.random.seed = random.randint(1, 5000)
            print(f'Random seed is set to be {np.random.seed}')
        d = X.shape[1]
        max_val = np.max(y)
        self.model_params(d=d)
        log_interval = 100
        max_reward = 0
        material_history = []
        reward_history = []
        measurement_count = 0
        print(f'Using variance = {v2}')
        while X.shape[0]>0:
            if initial_arm is not None and measurement_count==0:
                arm = initial_arm
            else:
                arm = self.choose(X, v2, measurement_count)
            if set_random_seed: # Print statements for debugging purposes
                print(f'The arm picked is {arm} for measurement_count={measurement_count}')
                print(f'The dimension of the vector is {X[arm].shape}')
                print(f'The number of non-zero components is {np.count_nonzero(X[arm])}')
            reward = y[arm]
            measurement_count+=1
            if reward>max_reward: # Need to incorporate an early termination condition if the max value is reached and can then increase the time_steps further
                max_reward=reward
                max_step = measurement_count
            
            material = material_list[arm]
            material_history.append(tuple(material))
            reward_history.append(reward)
            if reward==max_val:
                break
                # No more iterations needed
            self.update(X, arm, reward)

            X = np.delete(X, (arm), axis=0)
            y = np.delete(y, (arm), axis=0)
            material_list = np.delete(material_list, (arm), axis=0)
            if measurement_count%log_interval==0:
                print(f'Done with {measurement_count} iterations')
        
        return max_step, reward_history, material_history
    
    def run_multiple_bandits(self, X, y, material_list, v2, starting_indices=None):
        """Run the simulation multiple times in order to obtain the expected number of steps to reach the final reward over a finite time horizon"""
        max_step_list = []
        reward_history_list = []
        material_history_list = []
        for i, (index, material) in enumerate(starting_indices):
            svd_not_converge = True
            rand_seed = False
            print(f'Iteration number {i}, starting with material {material}')
            while svd_not_converge: # Necessary to keep external count the same
                try:
                    max_step, reward_history, material_history = self.run_bandits(X=X, y=y, material_list=material_list, v2=v2, set_random_seed=rand_seed, initial_arm=index)
                    svd_not_converge = False # Can potentially lead to an infinite loop
                except np.linalg.LinAlgError:
                    print('SVD failed to converge, trying again...')
                    print(f'The iteration number is {i}')
                    rand_seed = True
                    # If SVD does not converge during sampling,
            max_step_list.append(max_step)
            reward_history_list.append(reward_history)
            material_history_list.append(material_history)
        

        print(f'For contextual bandits, the list of number of steps required is {max_step_list}')
        print(f'For contextual bandits, the mean saturation steps is {np.mean(max_step_list)} and the standard deviation is {np.std(max_step_list)}')
        return max_step_list, reward_history_list, material_history_list
