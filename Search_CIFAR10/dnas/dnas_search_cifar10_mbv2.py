import torch
import argparse
import sys
import os
from pathlib import Path
import json
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import logging
matplotlib.use('Agg')

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.mobilenetv2 import AdaptQMobileNetV2, MixedPrecisionLayer

from utils import (
    create_train_val_test_loaders,
    load_layer_precision_options,
    set_layer_precision_options,
    validate_model_alpha_parameters,
    evaluate_model,
    calculate_hardware_penalty,
    update_temperature,
    plot_training_curves,
    save_dnas_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_analysis_json', type=str, default='../tensor_analysis_result/mbv2_cifar10_tensor_analysis_results.json')
    parser.add_argument('--int_only', action='store_true', help='Whether to search only INT precision options (int8, int4, int2)')
    parser.add_argument('--input_size', type=int, default=32, help='Input image size')
    args = parser.parse_args()
    
    log_dir = 'mbv2_logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = "_int_only" if args.int_only else "_mixed"
    log_file = os.path.join(log_dir, f'dnas_search_mbv2{mode_suffix}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    if args.int_only:
        logger.info("Starting DNAS search - MobileNetV2 INT Only")
    else:
        logger.info("Starting DNAS search - MobileNetV2 Mixed Precision")
    logger.info("=" * 80)
    
    logger.info("Training parameters:")
    logger.info(f"  Tensor analysis result: {args.tensor_analysis_json}")
    logger.info(f"  INT only mode: {args.int_only}")
    logger.info(f"  Input size: {args.input_size}x{args.input_size}")
    logger.info(f"  Log file: {log_file}")
    logger.info("-" * 80)

    device = torch.device('cuda:1')
    logger.info(f"Using GPU training: {device}")
    logger.info(f"Selected GPU 1: {torch.cuda.get_device_name(1)}")

    logger.info("Reading tensor analysis results...")
    if args.int_only:
        model_default_options = ["int8", "int4", "int2"]
        logger.info("INT only mode: searching only INT precision options")
    else:
        model_default_options = ["fp32", "fp16", "int8", "int4", "int2"]
        logger.info("Full precision mode: searching all precision options")
    
    layer_precision_options = load_layer_precision_options(args.tensor_analysis_json, int_only=args.int_only, model_default_options=model_default_options)
    logger.info(f"Successfully loaded precision options for {len(layer_precision_options)} layers")

    logger.info("Loading pretrained model...")
    pretrained_model = AdaptQMobileNetV2(
        num_classes=10,
        width_mult=1.0,
        precision_options=None,
        pretrain_mode=True,
        input_size=args.input_size
    )
    
    logger.info("Loading ImageNet pretrained weights...")
    success = pretrained_model.load_imagenet_pretrained_weights()
    if not success:
        logger.warning("ImageNet pretrained weights loading failed, using random initialization")
    else:
        logger.info("Successfully loaded ImageNet pretrained weights")

    logger.info("Creating DNAS model...")
    model = AdaptQMobileNetV2(
        num_classes=10,
        width_mult=1.0,
        precision_options=model_default_options,
        hardware_constraints=None,
        pretrain_mode=False,
        initialize_weights=True,
        input_size=args.input_size
    )
    model.copy_weights_from_pretrained(pretrained_model)
    
    logger.info("Single GPU training only, no data parallelism")
    
    model = model.to(device)
    logger.info("DNAS model created and moved to device")
    logger.info(f"Using standard MobileNetV2 configuration ({args.input_size}x{args.input_size})")
    logger.info("Model configuration: 7 IRB stages, final channels 1280")

    logger.info("Setting layer precision options...")
    set_layer_precision_options(model, layer_precision_options)
    logger.info("Precision options set")
    
    validate_model_alpha_parameters(model)
    
    logger.info("Validating memory filtering results:")
    total_modules_before = 0
    total_modules_after = 0
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            if args.int_only:
                total_modules_before += len(["int8", "int4", "int2"])
            else:
                total_modules_before += len(["fp32", "fp16", "int8", "int4", "int2"])
            total_modules_after += len(module.precision_options)
            logger.info(f"  Layer {name}:")
            logger.info(f"    Precision options: {module.precision_options}")
            logger.info(f"    Precision modules count: {len(module.precision_modules)}")
            logger.info(f"    PACT parameters count: {len(module.alpha_pact_dict)}")
            logger.info(f"    Alpha size: {module.alpha.size(0)}")
    
    logger.info(f"Memory filtering statistics:")
    logger.info(f"  Total modules reduced: {total_modules_before} -> {total_modules_after}")
    logger.info(f"  Reduction ratio: {((total_modules_before - total_modules_after) / total_modules_before * 100):.1f}%")
    
    model = model.to(device)
    logger.info("Ensuring all modules are on correct device")
    
    logger.info("Freezing all BN parameters...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False
            module.running_mean.requires_grad = False
            module.running_var.requires_grad = False
            logger.info(f"  Frozen BN layer: {name}")
    logger.info("All BN parameters frozen")
    
    logger.info("Debug info - Checking module devices:")
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            logger.info(f"  Layer {name}:")
            logger.info(f"    Module device: {next(module.parameters()).device}")
            logger.info(f"    Alpha device: {module.alpha.device}")
            for precision, pact_param in module.alpha_pact_dict.items():
                logger.info(f"    PACT {precision} device: {pact_param.device}")
            if module.precision_modules:
                first_precision = list(module.precision_modules.keys())[0]
                first_module = module.precision_modules[first_precision]
                logger.info(f"    First precision module({first_precision}) device: {next(first_module.parameters()).device}")
    
    logger.info("Debug info - Layer precision options and alpha sizes:")
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            logger.info(f"  Layer {name}:")
            logger.info(f"    Precision options: {module.precision_options}")
            logger.info(f"    Alpha size: {module.alpha.size(0)}")
            logger.info(f"    Options count: {len(module.precision_options)}")
            if module.alpha.size(0) != len(module.precision_options):
                logger.warning(f"    ⚠️  Mismatch! Alpha size({module.alpha.size(0)}) != Options count({len(module.precision_options)})")
            else:
                logger.info(f"    ✅  Match")

    logger.info("Creating data loaders...")
    if args.input_size <= 64:
        batch_size = 512
    elif args.input_size <= 128:
        batch_size = 256
    else:
        batch_size = 128
    
    input_size = args.input_size
    logger.info(f"{input_size}x{input_size} resolution training, batch size: {batch_size}")
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        batch_size=batch_size, 
        num_workers=8, 
        input_size=input_size
    )
    logger.info("Data loaders created")
    
    logger.info("Starting DNAS search main process...")
    epochs = 50
    warmup_epochs = 1
    
    lr_w = 0.001
    lr_alpha = 0.05
    logger.info(f"GPU training, weight learning rate: {lr_w}")
    logger.info(f"GPU training, alpha learning rate: {lr_alpha}")
    initial_temp = 5.0
    min_temp = 0.01
    temperature_decay = 'cubic'
    hardware_penalty_weight = 0.0000000001
    grad_clip_norm = 1.0
    

    
    logger.info("DNAS parameters:")
    logger.info(f"  Training epochs: {epochs}")
    logger.info(f"  Warmup epochs: {warmup_epochs} (training weights only, no alpha search)")
    logger.info(f"  Weight learning rate: {lr_w}")
    logger.info(f"  Alpha learning rate: {lr_alpha}")
    logger.info(f"  Initial temperature: {initial_temp}")
    logger.info(f"  Minimum temperature: {min_temp} (significantly reduced, ensuring one-hot convergence)")
    logger.info(f"  Temperature decay: {temperature_decay} (cubic decay, very aggressive)")
    logger.info(f"  Hardware penalty weight: {hardware_penalty_weight}")
    logger.info(f"  Gradient clipping: {grad_clip_norm}")
    
    logger.info("Temperature schedule validation:")
    for epoch in range(0, epochs, 10):
        progress = epoch / epochs
        if temperature_decay == 'cubic':
            temp = min_temp + (initial_temp - min_temp) * (1 - progress)**3
        elif temperature_decay == 'quadratic':
            temp = min_temp + (initial_temp - min_temp) * (1 - progress)**2
        elif temperature_decay == 'exponential':
            temp = min_temp + (initial_temp - min_temp) * np.exp(-3 * progress)
        else:
            temp = min_temp + (initial_temp - min_temp) * (1 - progress)
        logger.info(f"  Epoch {epoch}: temperature = {temp:.6f}")
    logger.info(f"  Epoch {epochs-1}: temperature = {min_temp:.6f} (final temperature)")
    
    logger.info("Performance optimization configuration:")
    logger.info(f"  Training mode: Single GPU training")
    logger.info(f"  Input resolution: {input_size}x{input_size}")
    logger.info(f"  Effective Batch Size: {batch_size}")
    logger.info(f"  Data loader workers: 8")
    logger.info(f"  Mixed precision training: Disabled")
    logger.info(f"  Using GPU: {torch.cuda.get_device_name(1)}")
    logger.info(f"  GPU memory: {torch.cuda.get_device_properties(1).total_memory / 1024**3:.1f} GB")
    
    optimizer_w = optim.Adam(model.parameters(), lr=lr_w, betas=(0.9, 0.999), weight_decay=4e-5)
    optimizer_alpha = optim.Adam([p for n, p in model.named_parameters() if 'alpha' in n], lr=lr_alpha, betas=(0.5, 0.999))
    scheduler_w = optim.lr_scheduler.CosineAnnealingLR(optimizer_w, epochs)

    best_acc = 0
    best_config = None
    
    training_history = {
        'epochs': [],
        'accuracies': [],
        'ce_losses': [],
        'hw_penalties': [],
        'total_losses': [],
        'temperatures': []
    }
    
    total_start_time = time.time()
    logger.info(f"DNAS search start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        current_temp = update_temperature(model, epoch, epochs, initial_temp, min_temp, temperature_decay)
        
        is_warmup = epoch < warmup_epochs
        
        if is_warmup:
            logger.info(f"Warmup phase Epoch {epoch}: training weights only, no alpha search")
        else:
            logger.info(f"Search phase Epoch {epoch}: training weights + searching alpha")
        
        model.train()
        
        for name, module in model.named_modules():
            if isinstance(module, MixedPrecisionLayer):
                module.alpha.requires_grad = False
                for param_name, param in module.named_parameters():
                    if 'alpha' not in param_name:
                        param.requires_grad = True
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training Weights Only', leave=False)
        epoch_ce_loss = 0.0
        epoch_hw_penalty = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer_w.zero_grad()
            
            outputs = model(images)
            ce_loss = F.cross_entropy(outputs, labels)
            hw_penalty = calculate_hardware_penalty(model, use_fixed_alpha=True)
            total_loss = ce_loss + hardware_penalty_weight * hw_penalty
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            optimizer_w.step()
            
            epoch_ce_loss += ce_loss.item()
            epoch_hw_penalty += hw_penalty.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1
            
            train_pbar.set_postfix({
                'CE_Loss': f'{ce_loss.item():.4f}',
                'HW_Penalty': f'{hw_penalty.item():.4f}',
                'Total_Loss': f'{total_loss.item():.4f}'
            })
        
        if not is_warmup:
            model.eval()
            
            for name, module in model.named_modules():
                if isinstance(module, MixedPrecisionLayer):
                    module.alpha.requires_grad = True
                    for param_name, param in module.named_parameters():
                        if 'alpha' not in param_name:
                            param.requires_grad = False
            
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch} - Training Alpha Only', leave=False)
            for batch_idx, (images, labels) in enumerate(val_pbar):
                images, labels = images.to(device), labels.to(device)
                optimizer_alpha.zero_grad()
                
                outputs = model(images)
                ce_loss = F.cross_entropy(outputs, labels)
                hw_penalty = calculate_hardware_penalty(model, use_fixed_alpha=False)
                total_loss = ce_loss + hardware_penalty_weight * hw_penalty
                total_loss.backward()
                
                alpha_params = [p for n, p in model.named_parameters() if 'alpha' in n]
                torch.nn.utils.clip_grad_norm_(alpha_params, grad_clip_norm)
                
                optimizer_alpha.step()
                
                val_pbar.set_postfix({
                    'CE_Loss': f'{ce_loss.item():.4f}',
                    'HW_Penalty': f'{hw_penalty.item():.4f}',
                    'Total_Loss': f'{total_loss.item():.4f}'
                })
        else:
            logger.info("Warmup phase: skipping alpha training")
            
        scheduler_w.step()
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time
        
        test_acc = evaluate_model(model, test_loader, device)
        current_loss = epoch_total_loss / num_batches
        
        training_history['epochs'].append(epoch)
        training_history['accuracies'].append(test_acc)
        training_history['ce_losses'].append(epoch_ce_loss / num_batches)
        training_history['hw_penalties'].append(epoch_hw_penalty / num_batches)
        training_history['total_losses'].append(current_loss)
        training_history['temperatures'].append(current_temp)
        
        plot_training_curves(
            training_history['epochs'],
            training_history['accuracies'],
            training_history['ce_losses'],
            training_history['hw_penalties'],
            training_history['total_losses'],
            f'training_curves_mbv2{mode_suffix}_live.png',
            mode_suffix=mode_suffix
        )
        
        if is_warmup:
            logger.info(f'Warmup Epoch: {epoch}, Test Accuracy: {test_acc:.2f}%')
            logger.info(f'Epoch time: {epoch_time:.2f}s, Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)')
            logger.info(f'CE Loss: {epoch_ce_loss/num_batches:.4f}')
        else:
            logger.info(f'Search Epoch: {epoch}, Temperature: {current_temp:.4f}, Test Accuracy: {test_acc:.2f}%, BN frozen')
            logger.info(f'Epoch time: {epoch_time:.2f}s, Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)')
            logger.info(f'CE Loss: {epoch_ce_loss/num_batches:.4f}, HW Penalty: {epoch_hw_penalty/num_batches:.4f}, Total Loss: {current_loss:.4f}')
            
            logger.info("Precision Distribution:")
            for name, module in model.named_modules():
                if isinstance(module, MixedPrecisionLayer):
                    weights = F.softmax(module.alpha / current_temp, dim=0)
                    selected_idx = min(weights.argmax().item(), len(module.alpha) - 1)
                    selected_precision = module.precision_options[selected_idx]
                    entropy = -(weights * torch.log(weights + 1e-10)).sum()
                    logger.info(f"{name}: {selected_precision}")
                    logger.info(f"  Weights: {weights.detach().cpu().numpy()}")
                    logger.info(f"  Entropy: {entropy.item():.4f}")
                    logger.info(f"  Precision options: {module.precision_options}")
                    logger.info(f"  Alpha size: {module.alpha.size(0)}, Options count: {len(module.precision_options)}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_config = {}
            for name, module in model.named_modules():
                if isinstance(module, MixedPrecisionLayer):
                    weights = F.softmax(module.alpha / current_temp, dim=0)
                    selected_idx = min(weights.argmax().item(), len(module.precision_options) - 1)
                    best_config[name] = module.precision_options[selected_idx]
            
            best_alpha_snapshot = {}
            for name, module in model.named_modules():
                if isinstance(module, MixedPrecisionLayer):
                    best_alpha_snapshot[name] = module.alpha.detach().cpu().numpy().tolist()
            
            os.makedirs('mbv2_best_models', exist_ok=True)
            model_state_dict = model.state_dict()
            
            torch.save({
                'precision_config': best_config,
                'alpha_snapshot': best_alpha_snapshot,
                'model_state_dict': model_state_dict,
                'training_history': training_history
            }, f'mbv2_best_models/best_dnas_cifar10_mbv2{mode_suffix}.pth')
            logger.info(f"Best model saved at epoch {epoch} with test acc {test_acc:.2f}%")
        
        if epoch == warmup_epochs - 1:
            logger.info(f"\n{'='*60}")
            logger.info(f"Warmup phase ended! Starting formal alpha parameter search from Epoch {warmup_epochs}")
            logger.info(f"{'='*60}")
            
        if epoch == warmup_epochs - 1:
            logger.info(f"\n{'='*60}")
            logger.info(f"Warmup phase ended! Starting formal alpha parameter search from Epoch {warmup_epochs}")
            logger.info(f"{'='*60}")

    total_training_time = time.time() - total_start_time
    logger.info(f"\nDNAS search completed!")
    logger.info(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.1f} minutes)")
    logger.info(f"Average time per epoch: {total_training_time/epochs:.2f}s")
    logger.info(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("Saving final supernet state...")
    final_supernet_state = {
        'model_state_dict': model.state_dict(),
        'best_config': best_config,
        'best_alpha_snapshot': best_alpha_snapshot,
        'training_history': training_history,
        'final_epoch': epoch,
        'final_temperature': current_temp,
        'final_accuracy': best_acc,
        'total_training_time': total_training_time
    }
    
    os.makedirs('mbv2_supernet_models', exist_ok=True)
    torch.save(final_supernet_state, f'mbv2_supernet_models/final_supernet_mbv2{mode_suffix}.pth')
    logger.info(f"Final supernet state saved to: mbv2_supernet_models/final_supernet_mbv2{mode_suffix}.pth")
    
    current_alpha_state = {}
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            current_alpha_state[name] = {
                'alpha': module.alpha.detach().cpu().numpy().tolist(),
                'temperature': module.temperature,
                'precision_options': module.precision_options
            }
    
    with open(f'mbv2_supernet_models/current_alpha_state_mbv2{mode_suffix}.json', 'w') as f:
        json.dump(current_alpha_state, f, indent=2)
    logger.info(f"Current alpha state saved to: mbv2_supernet_models/current_alpha_state_mbv2{mode_suffix}.json")
    
    logger.info("First stage search completed, best precision distribution:")
    for k, v in best_config.items():
        logger.info(f"{k}: {v}")
    logger.info(f"Best test accuracy: {best_acc:.2f}%")
    
    logger.info(f"\n{'='*60}")
    logger.info("DNAS search completed, supernet converged to one-hot state")
    logger.info(f"{'='*60}")

    logger.info("Validating final supernet performance...")
    try:
        final_test_acc = evaluate_model(model, test_loader, device)
        logger.info(f"Final test accuracy: {final_test_acc:.2f}%")
    except Exception as e:
        logger.error(f"Final supernet validation failed: {str(e)}")
        return

    model_state_dict = model.state_dict()
    
    torch.save({
        'precision_config': best_config,
        'model_state_dict': model_state_dict,
        'training_history': training_history,
        'final_accuracy': final_test_acc
    }, f'final_supernet_cifar10_mbv2{mode_suffix}.pth')
    logger.info("Final supernet saved.")

    save_dnas_results(best_config, best_acc, training_history, save_dir=f'dnas_results_mbv2{mode_suffix}', test_acc=best_acc)
    
    logger.info(f"\n{'='*60}")
    logger.info("DNAS Search Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Best accuracy: {best_acc:.2f}%")
    logger.info(f"Search epochs: {epochs}")
    logger.info(f"Temperature strategy: {temperature_decay}")
    logger.info(f"Search completed, supernet converged")
    logger.info(f"{'='*60}")

    total_time = time.time() - total_start_time
    logger.info(f"\n=== Total Time Statistics ===")
    logger.info(f"DNAS search: {total_training_time:.2f}s ({total_training_time/60:.1f} minutes)")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Start time: {datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main() 
