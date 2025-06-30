import torch
import yaml
import os
import json
from train_comparative import RoadDistressTrainer
from dual_input_model import create_model_variant, get_model_summary


def test_single_epoch_training(config, variant_name, max_batches=5):
    """Test training for a single epoch with limited batches."""
    print(f"\n{'='*60}")
    print(f"Testing {variant_name.upper()} - 1 epoch, max {max_batches} batches")
    print(f"{'='*60}")
    
    try:
        # Create a test config with smaller batch size but keep EfficientNet-B3
        test_config = yaml.safe_load(yaml.dump(config))  # Deep copy
        test_config['training']['batch_size'] = 2  # Very small batch size for testing
        # Keep EfficientNet-B3 as requested
        
        # Create trainer
        trainer = RoadDistressTrainer(test_config, variant_name)
        
        # Override data loaders to limit batches for testing
        original_train_loader = trainer.train_loader
        original_val_loader = trainer.val_loader
        
        # Create limited data loaders
        limited_train_dataset = torch.utils.data.Subset(
            original_train_loader.dataset, 
            range(min(len(original_train_loader.dataset), max_batches * test_config.get('training', {}).get('batch_size', 2)))
        )
        limited_val_dataset = torch.utils.data.Subset(
            original_val_loader.dataset, 
            range(min(len(original_val_loader.dataset), max_batches * test_config.get('training', {}).get('batch_size', 2)))
        )
        
        trainer.train_loader = torch.utils.data.DataLoader(
            limited_train_dataset,
            batch_size=test_config.get('training', {}).get('batch_size', 2),
            shuffle=True,
            num_workers=0,  # No multiprocessing for testing
            pin_memory=False  # Disable pin_memory for testing
        )
        
        trainer.val_loader = torch.utils.data.DataLoader(
            limited_val_dataset,
            batch_size=test_config.get('training', {}).get('batch_size', 2),
            shuffle=False,
            num_workers=0,  # No multiprocessing for testing
            pin_memory=False  # Disable pin_memory for testing
        )
        
        print(f"Model architecture: {trainer.model.__class__.__name__}")
        print(f"Backbone: {test_config.get('model', {}).get('architecture', {}).get('backbone', 'efficientnet_b3')}")
        print(f"Using masks: {trainer.variant_config['use_masks']}")
        print(f"Using augmentation: {trainer.variant_config['use_augmentation']}")
        print(f"Train samples: {len(trainer.train_loader.dataset)}")
        print(f"Val samples: {len(trainer.val_loader.dataset)}")
        print(f"Batch size: {test_config.get('training', {}).get('batch_size', 2)}")
        
        # Test model summary
        summary = get_model_summary(trainer.model)
        print(f"Model parameters: {summary['total_parameters']:,}")
        print(f"Model size: {summary['model_size_mb']:.2f} MB")
        
        # Test forward pass with sample data
        print("\nTesting forward pass...")
        sample_batch = next(iter(trainer.train_loader))
        if trainer.variant_config['use_masks']:
            images, masks, labels = sample_batch
            images, masks, labels = images.to(trainer.device), masks.to(trainer.device), labels.to(trainer.device)
            outputs = trainer.model(images, masks)
        else:
            images, labels = sample_batch
            images, labels = images.to(trainer.device), labels.to(trainer.device)
            outputs = trainer.model(images)
        
        print(f"Input shape: {images.shape}")
        if trainer.variant_config['use_masks']:
            print(f"Mask shape: {masks.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label distribution: {torch.bincount(labels, minlength=3).tolist()}")
        
        # Test one training epoch
        print("\nTesting training epoch...")
        train_metrics = trainer.train_epoch()
        print(f"Training metrics: {train_metrics}")
        
        # Test one validation epoch
        print("\nTesting validation epoch...")
        val_metrics = trainer.validate_epoch()
        print(f"Validation metrics: {val_metrics}")
        
        # Test checkpoint saving
        print("\nTesting checkpoint saving...")
        trainer.save_checkpoint(is_best=True)
        checkpoint_path = os.path.join(trainer.checkpoint_dir, 'best.pth')
        if os.path.exists(checkpoint_path):
            print(f"✓ Checkpoint saved successfully: {checkpoint_path}")
        else:
            print(f"✗ Checkpoint saving failed")
        
        # Test metrics saving
        print("\nTesting metrics saving...")
        trainer.epoch_metrics.append({
            'epoch': 0,
            'train': train_metrics,
            'val': val_metrics
        })
        trainer.save_epoch_metrics()
        metrics_path = os.path.join(trainer.output_dir, 'epoch_metrics.json')
        if os.path.exists(metrics_path):
            print(f"✓ Metrics saved successfully: {metrics_path}")
        else:
            print(f"✗ Metrics saving failed")
        
        print(f"\n✓ {variant_name.upper()} - ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ {variant_name.upper()} - TEST FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test training pipeline for all model variants."""
    print("Testing Training Pipeline - 1 Epoch, Limited Batches")
    print("=" * 60)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test variants
    variants = ['model_a', 'model_b', 'model_c', 'model_d']
    results = {}
    
    for variant in variants:
        success = test_single_epoch_training(config, variant, max_batches=3)
        results[variant] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("TESTING SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for variant, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{variant.upper()}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 ALL MODELS PASSED! Ready for full training.")
        print(f"Next step: Run 'python train_comparative.py' for full training")
    else:
        print(f"\n❌ Some models failed. Please fix issues before full training.")
    
    return all_passed


if __name__ == "__main__":
    main() 