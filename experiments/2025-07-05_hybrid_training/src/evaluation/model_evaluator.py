#!/usr/bin/env python3
"""
Model evaluator for loading and testing trained models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
from tqdm import tqdm
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.hybrid_model import create_model
from data.dataset import create_dataset
from evaluation.metrics_calculator import MetricsCalculator
from utils.platform_utils import PlatformManager

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Unified evaluator for all model variants with automatic CLAHE support.
    
    Automatically detects CLAHE models (E, F, G, H) and applies dynamic optimization
    during evaluation to match their training preprocessing pipeline.
    """
    
    def __init__(self, config_path: str, results_base_dir: str = "results"):
        """
        Initialize model evaluator with automatic CLAHE support.
        
        Args:
            config_path: Path to configuration file
            results_base_dir: Base directory containing model results
        """
        self.config = self._load_config(config_path)
        self.results_base_dir = Path(results_base_dir)
        self.platform_utils = PlatformManager(self.config)
        self.device = torch.device(self.platform_utils.get_device())
        self.metrics_calculator = MetricsCalculator()
        
        # Define CLAHE variants for automatic detection
        self.clahe_variants = ['model_e', 'model_f', 'model_g', 'model_h']
        
        logger.info(f"ModelEvaluator initialized with CLAHE support")
        logger.info(f"Device: {self.device}")
        logger.info(f"Results directory: {self.results_base_dir}")
        logger.info(f"CLAHE variants: {self.clahe_variants}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def evaluate_model(self, variant: str, checkpoint_name: str = "best_model.pth") -> Dict[str, Any]:
        """
        Evaluate a single model variant.
        
        Args:
            variant: Model variant ('model_a', 'model_b', 'model_c', 'model_d')
            checkpoint_name: Name of checkpoint file to load (default: best_model.pth)
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating {variant}")
        
        # Find model results directory
        model_dir = self.results_base_dir / variant
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load model
        model = self._load_model(variant, model_dir, checkpoint_name)
        
        # Create test dataset - CRITICAL: Force use_augmented=False for fair evaluation
        # Test evaluation should only use original images, never augmented ones
        logger.info(f"Creating test dataset for {variant} with use_augmented=False (evaluation best practice)")
        
        # Check if this variant uses CLAHE and enable dynamic optimization for evaluation
        clahe_variants = ['model_e', 'model_f', 'model_g', 'model_h']
        use_dynamic_clahe = variant in clahe_variants
        
        if use_dynamic_clahe:
            logger.info(f"Enabling dynamic CLAHE optimization for {variant} (CLAHE-trained model)")
        
        test_dataset = create_dataset('test', self.config, variant, use_augmented=False)
        
        # Enable dynamic CLAHE optimization for CLAHE-trained models during evaluation
        if use_dynamic_clahe:
            test_dataset.use_dynamic_clahe_optimization = True
            logger.info(f"Dynamic CLAHE optimization enabled for evaluation of {variant}")
        else:
            # Ensure the attribute exists and is False for non-CLAHE models
            test_dataset.use_dynamic_clahe_optimization = False
        
        # Use no workers for evaluation to eliminate multiprocessing issues and maximize GPU utilization
        num_workers = 0
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['evaluation']['test_batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda'
        )
        
        logger.info(f"Test dataset: {len(test_dataset)} samples, {len(test_loader)} batches")
        logger.info(f"Test dataset stats - use_augmented: {test_dataset.use_augmented}, use_masks: {test_dataset.use_masks}")
        
        if use_dynamic_clahe:
            dynamic_clahe_enabled = getattr(test_dataset, 'use_dynamic_clahe_optimization', False)
            logger.info(f"Dynamic CLAHE optimization: {'ENABLED' if dynamic_clahe_enabled else 'DISABLED'}")
        
        # Verify mask configuration for models that should use masks
        mask_variants = ['model_a', 'model_c', 'model_d', 'model_e', 'model_f', 'model_g', 'model_h']
        if variant in mask_variants:
            if not test_dataset.use_masks:
                logger.warning(f"WARNING: {variant} should use masks but use_masks=False!")
            else:
                logger.info(f"SUCCESS: Masks enabled for {variant}")
        
        # Run evaluation
        predictions, labels, probabilities = self._run_inference(model, test_loader, variant, use_dynamic_clahe)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            predictions, labels, probabilities
        )
        
        # Add model info
        model_info = self._get_model_info(model_dir)
        
        results = {
            'variant': variant,
            'model_info': model_info,
            'test_metrics': metrics,
            'predictions': predictions.cpu().numpy().tolist(),
            'labels': labels.cpu().numpy().tolist(),
            'probabilities': probabilities.cpu().numpy().tolist(),
            'summary_table': self.metrics_calculator.get_summary_table(metrics),
            'classification_report': self.metrics_calculator.get_classification_report(
                predictions.cpu().numpy(), labels.cpu().numpy()
            )
        }
        
        # Save evaluation results
        self._save_results(variant, results)
        
        logger.info(f"Evaluation completed for {variant}")
        return results
    
    def _get_preprocessing_pipeline_info(self, variant: str, uses_clahe: bool) -> Dict[str, Any]:
        """Get information about the preprocessing pipeline used."""
        
        pipeline_info = {
            'variant': variant,
            'uses_clahe': uses_clahe,
            'clahe_type': None,
            'other_preprocessing': []
        }
        
        if uses_clahe:
            # Determine CLAHE optimization type
            dynamic_optimization = True  # We're implementing dynamic optimization
            pipeline_info['clahe_type'] = 'dynamic_optimization' if dynamic_optimization else 'precomputed_parameters'
            pipeline_info['clahe_method'] = 'batch_clahe_optimization.py' if dynamic_optimization else 'clahe_params.json'
        
        # Add other preprocessing steps based on variant
        if variant in ['model_a', 'model_c', 'model_d', 'model_e', 'model_f', 'model_g', 'model_h']:
            pipeline_info['other_preprocessing'].append('road_masking')
        
        if variant in ['model_b', 'model_c', 'model_d', 'model_g', 'model_h']:
            pipeline_info['other_preprocessing'].append('data_augmentation_during_training')
        
        return pipeline_info
    
    def evaluate_all_models(self, checkpoint_name: str = "best_model.pth", variants: list = None) -> Dict[str, Any]:
        """
        Evaluate all available model variants with automatic CLAHE support.
        
        Args:
            checkpoint_name: Name of checkpoint file to load (default: best_model.pth)
            variants: List of variants to evaluate (default: all available)
            
        Returns:
            Dictionary containing results for all evaluated models
        """
        
        # Find available variants if not specified
        if variants is None:
            variants = []
            for variant_dir in self.results_base_dir.iterdir():
                if variant_dir.is_dir() and variant_dir.name.startswith('model_'):
                    variants.append(variant_dir.name)
            variants.sort()
        
        logger.info(f"🔬 Starting evaluation of {len(variants)} model variants")
        logger.info(f"Variants to evaluate: {variants}")
        
        results = {}
        clahe_count = 0
        
        for variant in variants:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Evaluating {variant.upper()}")
                logger.info(f"{'='*60}")
                
                variant_results = self.evaluate_model(variant, checkpoint_name)
                results[variant] = variant_results
                
                # Count CLAHE models
                if variant_results['evaluation_config']['uses_clahe']:
                    clahe_count += 1
                
                # Log key metrics
                metrics = variant_results['metrics']
                logger.info(f"✅ {variant} completed:")
                logger.info(f"   Overall Accuracy: {metrics.get('overall_accuracy', 0):.3f}")
                logger.info(f"   Macro F1: {metrics.get('f1_macro', 0):.3f}")
                logger.info(f"   Uses CLAHE: {variant_results['evaluation_config']['uses_clahe']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate {variant}: {str(e)}")
                results[variant] = {'error': str(e)}
        
        logger.info(f"\n🎯 Model Evaluation Summary:")
        logger.info(f"{'='*60}")
        
        successful = 0
        total = len(variants)
        
        for variant, result in results.items():
            if 'error' not in result:
                accuracy = result['metrics'].get('overall_accuracy', 0)
                uses_clahe = result['evaluation_config']['uses_clahe']
                clahe_status = "🎯 CLAHE" if uses_clahe else "📊 Standard"
                logger.info(f"✅ {variant}: {accuracy:.3f} accuracy ({clahe_status})")
                successful += 1
            else:
                logger.info(f"❌ {variant}: FAILED - {result['error']}")
        
        logger.info(f"\n📊 Evaluation Statistics:")
        logger.info(f"   Total models: {total}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {total - successful}")
        logger.info(f"   CLAHE models: {clahe_count}")
        logger.info(f"   Standard models: {successful - clahe_count}")
        
        return results
    
    def evaluate_clahe_models(self, checkpoint_name: str = "best_model.pth") -> Dict[str, Any]:
        """
        Evaluate only CLAHE-enhanced models (E, F, G, H) with dynamic optimization.
        
        Args:
            checkpoint_name: Name of checkpoint file to load (default: best_model.pth)
            
        Returns:
            Dictionary containing results for CLAHE models
        """
        logger.info("🔬 Starting evaluation of CLAHE-enhanced models")
        logger.info(f"CLAHE models to evaluate: {self.clahe_variants}")
        
        return self.evaluate_all_models(checkpoint_name, self.clahe_variants)
    
    def _load_model(self, variant: str, model_dir: Path, checkpoint_name: str) -> nn.Module:
        """Load trained model from checkpoint."""
        checkpoint_path = model_dir / "checkpoints" / checkpoint_name
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Create model
        model_config = self.config.get('model', {})
        model = create_model(
            variant=variant,
            num_classes=model_config.get('num_classes', 3),
            dropout_rate=model_config.get('classifier', {}).get('dropout_rate', 0.5),
            encoder_name=model_config.get('encoder_name', 'efficientnet_b3'),
            encoder_weights=model_config.get('encoder_weights', 'imagenet')
        )
        
        # Load checkpoint
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _run_inference(self, model: nn.Module, test_loader: DataLoader, variant: str = None, uses_clahe: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference on test data with enhanced logging for CLAHE models."""
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        model.eval()
        
        # Enhanced progress bar description
        desc = f"Evaluating {variant}" if variant else "Running inference"
        if uses_clahe:
            desc += " (CLAHE-optimized)"
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=desc)):
                # Handle different batch formats
                if len(batch) == 3:  # With masks
                    images, masks, labels = batch
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    labels = labels.to(self.device)
                else:  # Without masks
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    masks = None
                
                # Forward pass
                if masks is not None:
                    outputs = model(images, masks)
                else:
                    outputs = model(images)
                
                # Get probabilities and predictions
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_probabilities.append(probabilities.cpu())
                
                # Log progress for CLAHE models
                if uses_clahe and batch_idx > 0 and batch_idx % 50 == 0:
                    logger.debug(f"Processed {batch_idx * test_loader.batch_size} images with dynamic CLAHE optimization")
        
        # Concatenate all results
        predictions = torch.cat(all_predictions, dim=0)
        labels = torch.cat(all_labels, dim=0)
        probabilities = torch.cat(all_probabilities, dim=0)
        
        return predictions, labels, probabilities
    
    def _get_model_info(self, model_dir: Path) -> Dict[str, Any]:
        """Get model information from training summary."""
        summary_path = model_dir / "training_summary.json"
        
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                training_summary = json.load(f)
            
            model_info = {
                'training_time_hours': training_summary.get('training_results', {}).get('total_time_hours', 0),
                'best_epoch': training_summary.get('training_results', {}).get('best_epoch', 0),
                'best_val_metric': training_summary.get('training_results', {}).get('best_metric', 0),
                'final_train_loss': training_summary.get('training_results', {}).get('final_train_loss', 0),
                'final_val_loss': training_summary.get('training_results', {}).get('final_val_loss', 0),
                'config': training_summary.get('config', {}),
                'platform': training_summary.get('platform_info', {})
            }
        else:
            logger.warning(f"Training summary not found: {summary_path}")
            model_info = {}
        
        return model_info
    
    def _save_results(self, variant: str, results: Dict[str, Any]):
        """Save evaluation results to file."""
        # Create evaluation results directory
        eval_dir = self.results_base_dir / variant / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        # Save detailed results (without predictions/labels to reduce size)
        summary_results = {k: v for k, v in results.items() 
                          if k not in ['predictions', 'labels', 'probabilities']}
        
        summary_path = eval_dir / "test_evaluation.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_results, f, indent=2, default=str)
        
        # Save full results with predictions
        full_path = eval_dir / "test_evaluation_full.json"
        with open(full_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save text summary
        summary_text_path = eval_dir / "test_summary.txt"
        with open(summary_text_path, 'w') as f:
            f.write(results['summary_table'])
            f.write("\n\n")
            f.write(results['classification_report'])
        
        logger.info(f"Evaluation results saved to {eval_dir}")
    
    def evaluate_all_models(self, variants: list = None) -> Dict[str, Any]:
        """
        Evaluate all model variants.
        
        Args:
            variants: List of variants to evaluate (default: all variants)
            
        Returns:
            Dictionary with results for all models
        """
        if variants is None:
            variants = ['model_a', 'model_b', 'model_c', 'model_d']
        
        all_results = {}
        failed_models = []
        
        for variant in variants:
            try:
                results = self.evaluate_model(variant)
                all_results[variant] = results
                logger.info(f"✅ {variant} evaluation completed")
            except Exception as e:
                logger.error(f"❌ {variant} evaluation failed: {e}")
                failed_models.append(variant)
                all_results[variant] = {'error': str(e)}
        
        # Add summary
        summary = {
            'total_models': len(variants),
            'successful_evaluations': len(variants) - len(failed_models),
            'failed_evaluations': len(failed_models),
            'failed_models': failed_models
        }
        
        all_results['evaluation_summary'] = summary
        
        return all_results 