#!/usr/bin/env python3
"""
Comparative Training Script
Train all 4 model variants sequentially and compare results.
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from training.trainer import HybridTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComparativeTrainer:
    """Train all model variants and compare results."""
    
    def __init__(self, config_path: str = "config/base_config.yaml"):
        self.config_path = config_path
        self.variants = ['model_a', 'model_b', 'model_c', 'model_d']
        self.results = {}
        
    def train_all_variants(self) -> Dict:
        """Train all model variants sequentially."""
        logger.info("Starting comparative training for all variants")
        
        total_start_time = time.time()
        
        for i, variant in enumerate(self.variants, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training variant {i}/4: {variant}")
            logger.info(f"{'='*60}")
            
            try:
                trainer = HybridTrainer(self.config_path, variant)
                results = trainer.train()
                
                self.results[variant] = results
                
                print(f"\n✅ Completed {variant}:")
                print(f"  Best F1 score: {results['best_metric']:.4f}")
                print(f"  Best epoch: {results['best_epoch']+1}")
                print(f"  Total time: {results['total_time']/3600:.2f} hours")
                
                # Save intermediate results
                self._save_intermediate_results()
                
            except KeyboardInterrupt:
                logger.info(f"Training interrupted at {variant}")
                break
            except Exception as e:
                logger.error(f"Error training {variant}: {e}")
                self.results[variant] = {'error': str(e)}
                continue
        
        total_time = time.time() - total_start_time
        logger.info(f"\nTotal comparative training time: {total_time/3600:.2f} hours")
        
        return self.results
    
    def _save_intermediate_results(self):
        """Save intermediate results after each variant."""
        results_dir = Path("results/comparison")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / "intermediate_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def create_comparison_report(self):
        """Create comprehensive comparison report."""
        if not self.results:
            logger.warning("No results to compare")
            return
        
        results_dir = Path("results/comparison")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comparison table
        comparison_data = []
        for variant, results in self.results.items():
            if 'error' in results:
                comparison_data.append({
                    'Variant': variant,
                    'Status': 'Failed',
                    'Best F1': 0.0,
                    'Best Epoch': 0,
                    'Training Time (h)': 0.0,
                    'Error': results['error']
                })
            else:
                comparison_data.append({
                    'Variant': variant,
                    'Status': 'Success',
                    'Best F1': results['best_metric'],
                    'Best Epoch': results['best_epoch'] + 1,
                    'Training Time (h)': results['total_time'] / 3600,
                    'Error': None
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        df.to_csv(results_dir / "comparison_table.csv", index=False)
        
        # Create visualization
        self._create_comparison_plots(df, results_dir)
        
        # Print summary
        self._print_summary(df)
        
        # Save detailed results
        with open(results_dir / "detailed_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Comparison report saved to {results_dir}")
    
    def _create_comparison_plots(self, df: pd.DataFrame, results_dir: Path):
        """Create comparison visualizations."""
        plt.style.use('default')
        
        # Filter successful results
        success_df = df[df['Status'] == 'Success']
        
        if len(success_df) == 0:
            logger.warning("No successful results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # F1 Score comparison
        axes[0, 0].bar(success_df['Variant'], success_df['Best F1'], 
                      color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
        axes[0, 0].set_title('Best F1 Score by Variant')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_ylim(0, 1)
        
        # Training time comparison
        axes[0, 1].bar(success_df['Variant'], success_df['Training Time (h)'], 
                      color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
        axes[0, 1].set_title('Training Time by Variant')
        axes[0, 1].set_ylabel('Hours')
        
        # Epochs to convergence
        axes[1, 0].bar(success_df['Variant'], success_df['Best Epoch'], 
                      color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
        axes[1, 0].set_title('Epochs to Best Performance')
        axes[1, 0].set_ylabel('Epochs')
        
        # Efficiency (F1 per hour)
        efficiency = success_df['Best F1'] / success_df['Training Time (h)']
        axes[1, 1].bar(success_df['Variant'], efficiency, 
                      color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
        axes[1, 1].set_title('Training Efficiency (F1/hour)')
        axes[1, 1].set_ylabel('F1 Score per Hour')
        
        plt.tight_layout()
        plt.savefig(results_dir / "comparison_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Comparison plots saved")
    
    def _print_summary(self, df: pd.DataFrame):
        """Print training summary."""
        print(f"\n{'='*80}")
        print("COMPARATIVE TRAINING SUMMARY")
        print(f"{'='*80}")
        
        print("\nResults Table:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Find best performer
        success_df = df[df['Status'] == 'Success']
        if len(success_df) > 0:
            best_variant = success_df.loc[success_df['Best F1'].idxmax()]
            print(f"\n🏆 Best Performer: {best_variant['Variant']}")
            print(f"   F1 Score: {best_variant['Best F1']:.4f}")
            print(f"   Training Time: {best_variant['Training Time (h)']:.2f} hours")
            print(f"   Epochs: {best_variant['Best Epoch']}")
        
        # Training statistics
        if len(success_df) > 0:
            total_time = success_df['Training Time (h)'].sum()
            avg_f1 = success_df['Best F1'].mean()
            print(f"\n📊 Training Statistics:")
            print(f"   Successful variants: {len(success_df)}/4")
            print(f"   Total training time: {total_time:.2f} hours")
            print(f"   Average F1 score: {avg_f1:.4f}")
        
        print(f"\n{'='*80}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all hybrid model variants comparatively')
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        trainer = ComparativeTrainer(args.config)
        results = trainer.train_all_variants()
        trainer.create_comparison_report()
        
        print("\n✅ Comparative training completed!")
        print("📊 Check results/comparison/ for detailed analysis")
        
    except KeyboardInterrupt:
        print("\n🛑 Comparative training interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in comparative training: {e}")
        raise

if __name__ == "__main__":
    main() 