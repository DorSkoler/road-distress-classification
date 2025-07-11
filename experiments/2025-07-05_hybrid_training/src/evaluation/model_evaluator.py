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
    """Evaluates trained models on test data."""
    
    def __init__(self, config_path: str, results_base_dir: str = "results"):
        """
        Initialize model evaluator.
        
        Args:
            config_path: Path to configuration file
            results_base_dir: Base directory containing model results
        """
        self.config = self._load_config(config_path)
        self.results_base_dir = Path(results_base_dir)
        self.platform_utils = PlatformManager(self.config)
        self.device = torch.device(self.platform_utils.get_device())
        self.metrics_calculator = MetricsCalculator()
        
        logger.info(f"ModelEvaluator initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Results directory: {self.results_base_dir}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def evaluate_model(self, variant: str, checkpoint_name: str = "best_checkpoint.pth") -> Dict[str, Any]:
        """
        Evaluate a single model variant.
        
        Args:
            variant: Model variant ('model_a', 'model_b', 'model_c', 'model_d')
            checkpoint_name: Name of checkpoint file to load
            
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
        
        # Create test dataset
        test_dataset = create_dataset('test', self.config, variant)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['evaluation']['test_batch_size'],
            shuffle=False,
            num_workers=self.platform_utils.get_num_workers(),
            pin_memory=self.device.type == 'cuda'
        )
        
        logger.info(f"Test dataset: {len(test_dataset)} samples, {len(test_loader)} batches")
        
        # Run evaluation
        predictions, labels, probabilities = self._run_inference(model, test_loader)
        
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _run_inference(self, model: nn.Module, test_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference on test data."""
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Running inference"):
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