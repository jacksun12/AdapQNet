import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from tqdm import tqdm
from .visualization_utils import plot_training_curves, plot_quick_finetune_results
from .training_utils import evaluate_model


def save_dnas_results(best_config, best_acc, training_history, save_dir='dnas_results', test_acc=0.0):
    """Save DNAS search results"""
    logger = logging.getLogger(__name__)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save best configuration
    config_path = os.path.join(save_dir, 'best_precision_config.json')
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save training curves
    plot_training_curves(
        training_history['epochs'],
        training_history['accuracies'],
        training_history['ce_losses'],
        training_history['hw_penalties'],
        training_history['total_losses'],
        os.path.join(save_dir, 'training_curves.png')
    )
    
    # Save quick finetune results curves
    plot_quick_finetune_results(training_history, os.path.join(save_dir, 'quick_finetune_curves.png'))
    
    # Save results summary
    summary_path = os.path.join(save_dir, 'dnas_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"DNAS Search Results Summary\n")
        f.write(f"==========================\n")
        f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Total Epochs: {len(training_history['epochs'])}\n")
        f.write(f"Final Temperature: {training_history['temperatures'][-1]:.4f}\n")
        f.write(f"\nPrecision Distribution:\n")
        for layer, precision in best_config.items():
            f.write(f"  {layer}: {precision}\n")
        
        # Add quick finetune results
        if 'quick_finetune_results' in training_history and training_history['quick_finetune_results']:
            f.write(f"\nQuick Finetune Results:\n")
            f.write(f"  Total Quick Finetune Sessions: {len(training_history['quick_finetune_results'])}\n")
            
            improvements = [r['test_acc_after_finetune'] - r['validation_acc'] 
                           for r in training_history['quick_finetune_results']]
            avg_improvement = sum(improvements) / len(improvements)
            max_improvement = max(improvements)
            min_improvement = min(improvements)
            
            f.write(f"  Average Improvement: {avg_improvement:.2f}%\n")
            f.write(f"  Max Improvement: {max_improvement:.2f}%\n")
            f.write(f"  Min Improvement: {min_improvement:.2f}%\n")
            
            f.write(f"\n  Detailed Results:\n")
            for result in training_history['quick_finetune_results']:
                f.write(f"    Epoch {result['epoch']}: Val={result['validation_acc']:.2f}%, "
                       f"Test={result['test_acc_after_finetune']:.2f}%, "
                       f"Improvement={result['test_acc_after_finetune'] - result['validation_acc']:.2f}%\n")
    
    logger.info(f"DNAS results saved to {save_dir}/ directory")
