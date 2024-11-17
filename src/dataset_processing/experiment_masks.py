import numpy as np
import matplotlib.pyplot as plt

from typing import List
from dataset import AbstractSAMDataset

def experiment_loose_masks(dataset : AbstractSAMDataset, 
                           truth_mask_list : List[np.ndarray], 
                           looseness_list : List[int], 
                           noise_level_list : List[int], 
                           iterations_list : List[int],
                           combined_parameters : bool
                           ):
    for i, truth_mask in enumerate(truth_mask_list):
        if combined_parameters:
            fig, axes = plt.subplots(1, 1, figsize = (5, 5))
        
            axes.imshow(truth_mask, cmap = 'gray')
            axes.set_title("Original Mask")
            axes.axis('off')
            plt.show()

            for looseness in looseness_list:
                for noise_level in noise_level_list:
                    for iterations in iterations_list:
                        new_mask = dataset._get_mask_loose_dilation(truth_mask, looseness, noise_level, iterations)
                        fig, ax = plt.subplots(1, 1, figsize = (5, 5))

                        ax.imshow(new_mask, cmap = 'gray')
                        ax.set_title(f"Looseness: {looseness}, Noise: {noise_level}, Iter: {iterations}")
                        ax.axis('off')
                        plt.show()
            
        else:
            fixed_looseness = looseness_list[0]
            fixed_noise_level = noise_level_list[0]
            fixed_iterations = iterations_list[0]

            fig, axes = plt.subplots(1, len(looseness_list) + 1, figsize = (20, 20))
            axes[0].imshow(truth_mask, cmap = 'gray')
            axes[0].set_title("Original Mask")
            axes[0].axis('off')
            
            for i, looseness in enumerate(looseness_list):
                new_mask = dataset._get_mask_loose_dilation(truth_mask, looseness, fixed_noise_level, fixed_iterations)
                
                axes[i + 1].imshow(new_mask, cmap = 'gray')
                axes[i + 1].set_title(f"Looseness: {looseness}")
                axes[i + 1].axis('off')
            
            plt.show()
            
            fig, axes = plt.subplots(1, len(noise_level_list) + 1, figsize = (20, 20))
            axes[0].imshow(truth_mask, cmap = 'gray')
            axes[0].set_title("Original Mask")
            axes[0].axis('off')
            
            for i, noise_level in enumerate(noise_level_list):
                new_mask = dataset._get_mask_loose_dilation(truth_mask, fixed_looseness, noise_level, fixed_iterations)
                
                axes[i + 1].imshow(new_mask, cmap = 'gray')
                axes[i + 1].set_title(f"Noise Level: {noise_level}")
                axes[i + 1].axis('off')
            
            plt.show()
            
            fig, axes = plt.subplots(1, len(iterations_list) + 1, figsize = (20, 20))
            axes[0].imshow(truth_mask, cmap = 'gray')
            axes[0].set_title("Original Mask")
            axes[0].axis('off')
            
            for i, iterations in enumerate(iterations_list):
                new_mask = dataset._get_mask_loose_dilation(truth_mask, fixed_looseness, fixed_noise_level, iterations)
                
                axes[i + 1].imshow(new_mask, cmap = 'gray')
                axes[i + 1].set_title(f"Iterations: {iterations}")
                axes[i + 1].axis('off')
            
            plt.show()