import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题


def plot_training_curves(epochs, accuracies, ce_losses, hw_penalties, total_losses, save_path='training_curves.png'):
    """绘制训练曲线"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 准确率曲线
    ax1.plot(epochs, accuracies, 'b-', linewidth=2, label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Validation Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # CE Loss曲线
    ax2.plot(epochs, ce_losses, 'r-', linewidth=2, label='Cross Entropy Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Cross Entropy Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Hardware Penalty曲线
    ax3.plot(epochs, hw_penalties, 'g-', linewidth=2, label='Hardware Penalty')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Penalty')
    ax3.set_title('Hardware Penalty')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Total Loss曲线
    ax4.plot(epochs, total_losses, 'm-', linewidth=2, label='Total Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Total Loss')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_quick_finetune_results(training_history, save_path='quick_finetune_curves.png'):
    """绘制快速微调结果曲线"""
    if 'quick_finetune_results' not in training_history or not training_history['quick_finetune_results']:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = [r['epoch'] for r in training_history['quick_finetune_results']]
    val_accs = [r['validation_acc'] for r in training_history['quick_finetune_results']]
    test_accs = [r['test_acc_after_finetune'] for r in training_history['quick_finetune_results']]
    improvements = [r['test_acc_after_finetune'] - r['validation_acc'] for r in training_history['quick_finetune_results']]
    
    # 准确率对比
    ax1.plot(epochs, val_accs, 'b-o', linewidth=2, markersize=6, label='Validation Accuracy')
    ax1.plot(epochs, test_accs, 'r-s', linewidth=2, markersize=6, label='Test Accuracy (After Finetune)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Quick Finetune Results')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 提升幅度
    ax2.bar(epochs, improvements, color='green', alpha=0.7, label='Improvement')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Finetune Improvement')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 添加平均线
    avg_improvement = sum(improvements) / len(improvements)
    ax2.axhline(y=avg_improvement, color='red', linestyle='--', linewidth=2, 
                label=f'Average: {avg_improvement:.2f}%')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 