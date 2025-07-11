#!/usr/bin/env python3
"""
Comparison runner for evaluating and comparing all trained models.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import pandas as pd
import sys
import numpy as np

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.model_evaluator import ModelEvaluator
from evaluation.visualization import create_comparison_plots, create_summary_table_plot

logger = logging.getLogger(__name__)

class ComparisonRunner:
    """Runs comprehensive comparison of all trained models."""
    
    def __init__(self, config_path: str, results_base_dir: str = "results"):
        """
        Initialize comparison runner.
        
        Args:
            config_path: Path to configuration file
            results_base_dir: Base directory containing model results
        """
        self.config_path = config_path
        self.results_base_dir = Path(results_base_dir)
        self.evaluator = ModelEvaluator(config_path, results_base_dir)
        
        # Create comparison results directory
        self.comparison_dir = self.results_base_dir / "comparison"
        self.comparison_dir.mkdir(exist_ok=True)
        
        logger.info(f"ComparisonRunner initialized")
        logger.info(f"Comparison results will be saved to: {self.comparison_dir}")
    
    def run_full_comparison(self, variants: List[str] = None) -> Dict[str, Any]:
        """
        Run complete evaluation and comparison of all models.
        
        Args:
            variants: List of model variants to compare (default: all)
            
        Returns:
            Dictionary with complete comparison results
        """
        if variants is None:
            variants = ['model_a', 'model_b', 'model_c', 'model_d']
        
        logger.info(f"Starting full comparison for variants: {variants}")
        start_time = time.time()
        
        # Run evaluations
        evaluation_results = self._run_evaluations(variants)
        
        # Create comparison analysis
        comparison_analysis = self._create_comparison_analysis(evaluation_results)
        
        # Generate visualizations
        self._generate_visualizations(evaluation_results)
        
        # Create summary report
        summary_report = self._create_summary_report(evaluation_results, comparison_analysis)
        
        # Save all results
        self._save_comparison_results(evaluation_results, comparison_analysis, summary_report)
        
        total_time = time.time() - start_time
        logger.info(f"Full comparison completed in {total_time:.2f} seconds")
        
        return {
            'evaluation_results': evaluation_results,
            'comparison_analysis': comparison_analysis,
            'summary_report': summary_report,
            'total_time_seconds': total_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def _run_evaluations(self, variants: List[str]) -> Dict[str, Any]:
        """Run evaluations for all specified variants."""
        logger.info("Running model evaluations...")
        
        evaluation_results = {}
        successful_evaluations = 0
        
        for variant in variants:
            try:
                logger.info(f"Evaluating {variant}...")
                result = self.evaluator.evaluate_model(variant)
                evaluation_results[variant] = result
                successful_evaluations += 1
                logger.info(f"✅ {variant} evaluation completed")
            except Exception as e:
                logger.error(f"❌ {variant} evaluation failed: {e}")
                evaluation_results[variant] = {'error': str(e)}
        
        # Add evaluation summary
        evaluation_results['evaluation_summary'] = {
            'total_models': len(variants),
            'successful_evaluations': successful_evaluations,
            'failed_evaluations': len(variants) - successful_evaluations,
            'success_rate': successful_evaluations / len(variants) if variants else 0
        }
        
        logger.info(f"Evaluations completed: {successful_evaluations}/{len(variants)} successful")
        return evaluation_results
    
    def _create_comparison_analysis(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed comparison analysis of all models."""
        logger.info("Creating comparison analysis...")
        
        # Extract valid results
        valid_results = {k: v for k, v in evaluation_results.items() 
                        if k != 'evaluation_summary' and 'error' not in v}
        
        if not valid_results:
            logger.warning("No valid results for comparison analysis")
            return {'error': 'No valid results available'}
        
        analysis = {}
        
        # Best performing models by metric
        analysis['best_performers'] = self._find_best_performers(valid_results)
        
        # Ranking table
        analysis['ranking_table'] = self._create_ranking_table(valid_results)
        
        # Statistical analysis
        analysis['statistical_summary'] = self._calculate_statistical_summary(valid_results)
        
        # Training efficiency analysis
        analysis['efficiency_analysis'] = self._analyze_training_efficiency(valid_results)
        
        # Class-specific analysis
        analysis['class_analysis'] = self._analyze_class_performance(valid_results)
        
        # Model recommendations
        analysis['recommendations'] = self._generate_recommendations(valid_results)
        
        return analysis
    
    def _find_best_performers(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Find best performing model for each metric."""
        metrics = ['macro_f1', 'macro_precision', 'macro_recall', 'hamming_accuracy', 
                  'exact_match_accuracy']
        
        best_performers = {}
        
        for metric in metrics:
            best_model = None
            best_value = -1
            
            for model_name, model_results in results.items():
                value = model_results['test_metrics'].get(metric, 0)
                if value > best_value:
                    best_value = value
                    best_model = model_name
            
            best_performers[metric] = {
                'model': best_model,
                'value': best_value
            }
        
        return best_performers
    
    def _create_ranking_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create comprehensive ranking table."""
        ranking_data = []
        
        for model_name, model_results in results.items():
            metrics = model_results['test_metrics']
            model_info = model_results.get('model_info', {})
            
            row = {
                'Model': model_name,
                'Macro F1': metrics.get('macro_f1', 0),
                'Macro Precision': metrics.get('macro_precision', 0),
                'Macro Recall': metrics.get('macro_recall', 0),
                'Hamming Accuracy': metrics.get('hamming_accuracy', 0),
                'Exact Match': metrics.get('exact_match_accuracy', 0),
                'Damage F1': metrics.get('damage_f1', 0),
                'Occlusion F1': metrics.get('occlusion_f1', 0),
                'Crop F1': metrics.get('crop_f1', 0),
                'Training Time (h)': model_info.get('training_time_hours', 0),
                'Best Epoch': model_info.get('best_epoch', 0),
                'Val Loss Ratio': (model_info.get('final_val_loss', 0) / 
                                 max(model_info.get('final_train_loss', 1), 1e-6))
            }
            ranking_data.append(row)
        
        df = pd.DataFrame(ranking_data)
        
        # Sort by Macro F1 (primary metric)
        df = df.sort_values('Macro F1', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        
        # Reorder columns
        cols = ['Rank', 'Model'] + [col for col in df.columns if col not in ['Rank', 'Model']]
        df = df[cols]
        
        return df
    
    def _calculate_statistical_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical summary of model performance."""
        metrics_data = {}
        
        # Collect metrics from all models
        for model_name, model_results in results.items():
            for metric_name, value in model_results['test_metrics'].items():
                if isinstance(value, (int, float)):
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = []
                    metrics_data[metric_name].append(value)
        
        # Calculate statistics
        summary = {}
        for metric_name, values in metrics_data.items():
            if len(values) > 1:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values))
                }
        
        return summary
    
    def _analyze_training_efficiency(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training time vs performance efficiency."""
        efficiency_data = []
        
        for model_name, model_results in results.items():
            model_info = model_results.get('model_info', {})
            metrics = model_results['test_metrics']
            
            training_time = model_info.get('training_time_hours', 0)
            f1_score = metrics.get('macro_f1', 0)
            
            # Calculate efficiency score (F1 per hour)
            efficiency = f1_score / max(training_time, 0.01)  # Avoid division by zero
            
            efficiency_data.append({
                'model': model_name,
                'training_time': training_time,
                'f1_score': f1_score,
                'efficiency_score': efficiency
            })
        
        # Sort by efficiency
        efficiency_data.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        return {
            'efficiency_ranking': efficiency_data,
            'most_efficient': efficiency_data[0]['model'] if efficiency_data else None,
            'least_efficient': efficiency_data[-1]['model'] if efficiency_data else None
        }
    
    def _analyze_class_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze per-class performance across models."""
        class_names = ['damage', 'occlusion', 'crop']
        analysis = {}
        
        for class_name in class_names:
            class_data = []
            
            for model_name, model_results in results.items():
                metrics = model_results['test_metrics']
                
                class_data.append({
                    'model': model_name,
                    'precision': metrics.get(f'{class_name}_precision', 0),
                    'recall': metrics.get(f'{class_name}_recall', 0),
                    'f1': metrics.get(f'{class_name}_f1', 0),
                    'accuracy': metrics.get(f'{class_name}_accuracy', 0)
                })
            
            # Find best model for this class
            best_model = max(class_data, key=lambda x: x['f1'])
            
            analysis[class_name] = {
                'performance_data': class_data,
                'best_model': best_model['model'],
                'best_f1': best_model['f1']
            }
        
        return analysis
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate recommendations based on analysis."""
        recommendations = {}
        
        # Overall best model
        best_overall = max(results.items(), 
                          key=lambda x: x[1]['test_metrics'].get('macro_f1', 0))
        recommendations['best_overall'] = f"{best_overall[0]} (Macro F1: {best_overall[1]['test_metrics']['macro_f1']:.3f})"
        
        # Most efficient model
        efficiency_scores = []
        for model_name, model_results in results.items():
            model_info = model_results.get('model_info', {})
            training_time = model_info.get('training_time_hours', 0.01)
            f1_score = model_results['test_metrics'].get('macro_f1', 0)
            efficiency = f1_score / training_time
            efficiency_scores.append((model_name, efficiency))
        
        most_efficient = max(efficiency_scores, key=lambda x: x[1])
        recommendations['most_efficient'] = f"{most_efficient[0]} (F1/hour: {most_efficient[1]:.3f})"
        
        # Best for specific use cases
        recommendations['best_for_damage'] = max(results.items(), 
                                               key=lambda x: x[1]['test_metrics'].get('damage_f1', 0))[0]
        recommendations['best_for_occlusion'] = max(results.items(), 
                                                  key=lambda x: x[1]['test_metrics'].get('occlusion_f1', 0))[0]
        recommendations['best_for_crop'] = max(results.items(), 
                                             key=lambda x: x[1]['test_metrics'].get('crop_f1', 0))[0]
        
        return recommendations
    
    def _generate_visualizations(self, evaluation_results: Dict[str, Any]) -> None:
        """Generate all comparison visualizations."""
        logger.info("Generating comparison visualizations...")
        
        # Create plots directory
        plots_dir = self.comparison_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        try:
            # Main comparison plots
            create_comparison_plots(evaluation_results, plots_dir)
            
            # Summary table plot
            create_summary_table_plot(evaluation_results, plots_dir)
            
            logger.info(f"Visualizations saved to {plots_dir}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _create_summary_report(self, evaluation_results: Dict[str, Any], 
                             comparison_analysis: Dict[str, Any]) -> str:
        """Create comprehensive text summary report."""
        lines = [
            "=" * 80,
            "COMPREHENSIVE MODEL COMPARISON REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Evaluation summary
        eval_summary = evaluation_results.get('evaluation_summary', {})
        lines.extend([
            "EVALUATION SUMMARY",
            "-" * 40,
            f"Total models evaluated: {eval_summary.get('total_models', 0)}",
            f"Successful evaluations: {eval_summary.get('successful_evaluations', 0)}",
            f"Failed evaluations: {eval_summary.get('failed_evaluations', 0)}",
            f"Success rate: {eval_summary.get('success_rate', 0):.1%}",
            ""
        ])
        
        # Best performers
        if 'best_performers' in comparison_analysis:
            lines.extend([
                "BEST PERFORMING MODELS BY METRIC",
                "-" * 40
            ])
            for metric, info in comparison_analysis['best_performers'].items():
                lines.append(f"{metric.replace('_', ' ').title()}: {info['model']} ({info['value']:.3f})")
            lines.append("")
        
        # Ranking table
        if 'ranking_table' in comparison_analysis:
            lines.extend([
                "MODEL RANKING (by Macro F1)",
                "-" * 40
            ])
            df = comparison_analysis['ranking_table']
            lines.append(df.to_string(index=False, float_format='%.3f'))
            lines.append("")
        
        # Recommendations
        if 'recommendations' in comparison_analysis:
            lines.extend([
                "RECOMMENDATIONS",
                "-" * 40
            ])
            for use_case, recommendation in comparison_analysis['recommendations'].items():
                lines.append(f"{use_case.replace('_', ' ').title()}: {recommendation}")
            lines.append("")
        
        # Class-specific analysis
        if 'class_analysis' in comparison_analysis:
            lines.extend([
                "CLASS-SPECIFIC PERFORMANCE",
                "-" * 40
            ])
            for class_name, analysis in comparison_analysis['class_analysis'].items():
                lines.append(f"{class_name.capitalize()}: Best model is {analysis['best_model']} (F1: {analysis['best_f1']:.3f})")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _save_comparison_results(self, evaluation_results: Dict[str, Any], 
                               comparison_analysis: Dict[str, Any], 
                               summary_report: str) -> None:
        """Save all comparison results to files."""
        logger.info("Saving comparison results...")
        
        # Save detailed results
        with open(self.comparison_dir / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Save comparison analysis
        with open(self.comparison_dir / "comparison_analysis.json", 'w') as f:
            json.dump(comparison_analysis, f, indent=2, default=str)
        
        # Save summary report
        with open(self.comparison_dir / "summary_report.txt", 'w') as f:
            f.write(summary_report)
        
        # Save ranking table as CSV
        if 'ranking_table' in comparison_analysis:
            df = comparison_analysis['ranking_table']
            df.to_csv(self.comparison_dir / "model_rankings.csv", index=False)
        
        logger.info(f"Comparison results saved to {self.comparison_dir}")


def main():
    """Main function for running comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive model comparison')
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Base directory containing model results')
    parser.add_argument('--variants', nargs='+', 
                       default=['model_a', 'model_b', 'model_c', 'model_d'],
                       help='Model variants to compare')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        runner = ComparisonRunner(args.config, args.results_dir)
        results = runner.run_full_comparison(args.variants)
        
        print(f"\n✅ Comparison completed successfully!")
        print(f"📊 Results saved to: {runner.comparison_dir}")
        print(f"🕒 Total time: {results['total_time_seconds']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise

if __name__ == "__main__":
    main() 