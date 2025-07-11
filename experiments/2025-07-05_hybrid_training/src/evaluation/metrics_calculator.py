#!/usr/bin/env python3
"""
Comprehensive metrics calculator for road distress classification.
"""

import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_auc_score, average_precision_score,
    matthews_corrcoef
)
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Comprehensive metrics calculation for multi-label classification."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: Names of classes (default: ['damage', 'occlusion', 'crop'])
        """
        self.class_names = class_names or ['damage', 'occlusion', 'crop']
        self.num_classes = len(self.class_names)
    
    def calculate_all_metrics(self, 
                            predictions: torch.Tensor, 
                            labels: torch.Tensor,
                            probabilities: torch.Tensor = None) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for model evaluation.
        
        Args:
            predictions: Binary predictions [N, num_classes]
            labels: Ground truth labels [N, num_classes] 
            probabilities: Prediction probabilities [N, num_classes]
            
        Returns:
            Dictionary with all metrics
        """
        # Convert to numpy for sklearn compatibility
        predictions_np = predictions.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        if probabilities is not None:
            probabilities_np = probabilities.detach().cpu().numpy()
        else:
            probabilities_np = predictions_np
            
        metrics = {}
        
        # Basic per-class metrics
        metrics.update(self._calculate_per_class_metrics(predictions_np, labels_np))
        
        # Overall metrics
        metrics.update(self._calculate_overall_metrics(predictions_np, labels_np))
        
        # Advanced metrics (if probabilities available)
        if probabilities is not None:
            metrics.update(self._calculate_advanced_metrics(predictions_np, labels_np, probabilities_np))
        
        # Confusion matrices
        metrics['confusion_matrices'] = self._calculate_confusion_matrices(predictions_np, labels_np)
        
        # Summary statistics
        metrics.update(self._calculate_summary_stats(predictions_np, labels_np))
        
        return metrics
    
    def _calculate_per_class_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate precision, recall, F1 for each class."""
        metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            pred_class = predictions[:, i]
            label_class = labels[:, i]
            
            # Basic counts
            tp = np.sum(pred_class * label_class)
            fp = np.sum(pred_class * (1 - label_class))
            fn = np.sum((1 - pred_class) * label_class)
            tn = np.sum((1 - pred_class) * (1 - label_class))
            
            # Calculate metrics with division by zero protection
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            # Specificity (true negative rate)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Store metrics
            metrics[f'{class_name}_precision'] = precision
            metrics[f'{class_name}_recall'] = recall
            metrics[f'{class_name}_f1'] = f1
            metrics[f'{class_name}_accuracy'] = accuracy
            metrics[f'{class_name}_specificity'] = specificity
            
            # Store counts for detailed analysis
            metrics[f'{class_name}_tp'] = int(tp)
            metrics[f'{class_name}_fp'] = int(fp)
            metrics[f'{class_name}_fn'] = int(fn)
            metrics[f'{class_name}_tn'] = int(tn)
            
            # Support (number of true instances)
            metrics[f'{class_name}_support'] = int(np.sum(label_class))
        
        return metrics
    
    def _calculate_overall_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate overall metrics across all classes."""
        metrics = {}
        
        # Overall accuracy (exact match)
        exact_match = np.all(predictions == labels, axis=1)
        metrics['exact_match_accuracy'] = np.mean(exact_match)
        
        # Hamming accuracy (per-label accuracy)
        metrics['hamming_accuracy'] = np.mean(predictions == labels)
        
        # Macro averages (unweighted)
        precisions = []
        recalls = []
        f1s = []
        
        for i in range(self.num_classes):
            pred_class = predictions[:, i]
            label_class = labels[:, i]
            
            tp = np.sum(pred_class * label_class)
            fp = np.sum(pred_class * (1 - label_class))
            fn = np.sum((1 - pred_class) * label_class)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        metrics['macro_precision'] = np.mean(precisions)
        metrics['macro_recall'] = np.mean(recalls)
        metrics['macro_f1'] = np.mean(f1s)
        
        # Micro averages (weighted by support)
        total_tp = np.sum([np.sum(predictions[:, i] * labels[:, i]) for i in range(self.num_classes)])
        total_fp = np.sum([np.sum(predictions[:, i] * (1 - labels[:, i])) for i in range(self.num_classes)])
        total_fn = np.sum([np.sum((1 - predictions[:, i]) * labels[:, i]) for i in range(self.num_classes)])
        
        metrics['micro_precision'] = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        metrics['micro_recall'] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        metrics['micro_f1'] = (2 * metrics['micro_precision'] * metrics['micro_recall'] / 
                              (metrics['micro_precision'] + metrics['micro_recall'])) if (metrics['micro_precision'] + metrics['micro_recall']) > 0 else 0.0
        
        # Weighted averages (by class support)
        class_supports = [np.sum(labels[:, i]) for i in range(self.num_classes)]
        total_support = sum(class_supports)
        
        if total_support > 0:
            weights = [support / total_support for support in class_supports]
            metrics['weighted_precision'] = sum(w * p for w, p in zip(weights, precisions))
            metrics['weighted_recall'] = sum(w * r for w, r in zip(weights, recalls))
            metrics['weighted_f1'] = sum(w * f for w, f in zip(weights, f1s))
        else:
            metrics['weighted_precision'] = 0.0
            metrics['weighted_recall'] = 0.0
            metrics['weighted_f1'] = 0.0
        
        return metrics
    
    def _calculate_advanced_metrics(self, predictions: np.ndarray, labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
        """Calculate advanced metrics using probabilities."""
        metrics = {}
        
        try:
            # ROC AUC for each class
            for i, class_name in enumerate(self.class_names):
                if len(np.unique(labels[:, i])) > 1:  # Only if both classes present
                    auc = roc_auc_score(labels[:, i], probabilities[:, i])
                    metrics[f'{class_name}_auc'] = auc
                else:
                    metrics[f'{class_name}_auc'] = 0.0
            
            # Average Precision (AP) for each class
            for i, class_name in enumerate(self.class_names):
                if len(np.unique(labels[:, i])) > 1:
                    ap = average_precision_score(labels[:, i], probabilities[:, i])
                    metrics[f'{class_name}_ap'] = ap
                else:
                    metrics[f'{class_name}_ap'] = 0.0
            
            # Overall AUC metrics
            aucs = [metrics[f'{class_name}_auc'] for class_name in self.class_names]
            aps = [metrics[f'{class_name}_ap'] for class_name in self.class_names]
            
            metrics['macro_auc'] = np.mean(aucs)
            metrics['macro_ap'] = np.mean(aps)
            
        except Exception as e:
            logger.warning(f"Could not calculate advanced metrics: {e}")
            for class_name in self.class_names:
                metrics[f'{class_name}_auc'] = 0.0
                metrics[f'{class_name}_ap'] = 0.0
            metrics['macro_auc'] = 0.0
            metrics['macro_ap'] = 0.0
        
        return metrics
    
    def _calculate_confusion_matrices(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate confusion matrix for each class."""
        confusion_matrices = {}
        
        for i, class_name in enumerate(self.class_names):
            cm = confusion_matrix(labels[:, i], predictions[:, i])
            confusion_matrices[class_name] = cm
        
        return confusion_matrices
    
    def _calculate_summary_stats(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate summary statistics."""
        metrics = {}
        
        # Class distribution in predictions and labels
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_pred_positive_rate'] = np.mean(predictions[:, i])
            metrics[f'{class_name}_true_positive_rate'] = np.mean(labels[:, i])
        
        # Overall statistics
        metrics['total_samples'] = len(predictions)
        metrics['avg_labels_per_sample'] = np.mean(np.sum(labels, axis=1))
        metrics['avg_predictions_per_sample'] = np.mean(np.sum(predictions, axis=1))
        
        # Label correlation (how often classes co-occur)
        label_correlations = {}
        for i, class1 in enumerate(self.class_names):
            for j, class2 in enumerate(self.class_names):
                if i < j:  # Only calculate upper triangle
                    correlation = np.corrcoef(labels[:, i], labels[:, j])[0, 1]
                    label_correlations[f'{class1}_{class2}_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        metrics['label_correlations'] = label_correlations
        
        return metrics
    
    def get_classification_report(self, predictions: np.ndarray, labels: np.ndarray) -> str:
        """Get detailed classification report."""
        reports = []
        
        for i, class_name in enumerate(self.class_names):
            report = classification_report(
                labels[:, i], 
                predictions[:, i], 
                target_names=[f'not_{class_name}', class_name],
                zero_division=0
            )
            reports.append(f"\n=== {class_name.upper()} ===\n{report}")
        
        return "\n".join(reports)
    
    def get_summary_table(self, metrics: Dict[str, Any]) -> str:
        """Get formatted summary table of key metrics."""
        lines = [
            "=" * 80,
            "MODEL EVALUATION SUMMARY",
            "=" * 80,
            f"{'Metric':<25} {'Damage':<12} {'Occlusion':<12} {'Crop':<12} {'Overall':<12}",
            "-" * 80
        ]
        
        # Per-class metrics
        for metric in ['precision', 'recall', 'f1', 'accuracy']:
            line = f"{metric.capitalize():<25}"
            for class_name in self.class_names:
                value = metrics.get(f'{class_name}_{metric}', 0.0)
                line += f"{value:<12.4f}"
            
            # Overall metric
            overall_value = metrics.get(f'macro_{metric}', metrics.get(f'weighted_{metric}', 0.0))
            line += f"{overall_value:<12.4f}"
            lines.append(line)
        
        # Add AUC if available
        if any(f'{class_name}_auc' in metrics for class_name in self.class_names):
            line = f"{'AUC':<25}"
            for class_name in self.class_names:
                value = metrics.get(f'{class_name}_auc', 0.0)
                line += f"{value:<12.4f}"
            overall_value = metrics.get('macro_auc', 0.0)
            line += f"{overall_value:<12.4f}"
            lines.append(line)
        
        lines.extend([
            "-" * 80,
            f"Exact Match Accuracy: {metrics.get('exact_match_accuracy', 0.0):.4f}",
            f"Hamming Accuracy: {metrics.get('hamming_accuracy', 0.0):.4f}",
            f"Total Samples: {metrics.get('total_samples', 0)}",
            "=" * 80
        ])
        
        return "\n".join(lines) 