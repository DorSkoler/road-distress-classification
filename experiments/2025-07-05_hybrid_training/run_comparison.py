#!/usr/bin/env python3
"""
Standalone script to run comprehensive model comparison and evaluation.

This script evaluates all trained models and creates detailed comparison reports
with visualizations, metrics, and recommendations.
"""

import sys
import logging
from pathlib import Path
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from evaluation.comparison_runner import ComparisonRunner

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('comparison.log')
        ]
    )

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive comparison of all trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_comparison.py                           # Compare all models
  python run_comparison.py --variants model_a model_b  # Compare specific models
  python run_comparison.py --verbose                 # Detailed logging
  
The script will:
  1. Load all trained models from results/ directory
  2. Evaluate each model on the test dataset
  3. Calculate comprehensive metrics (accuracy, F1, precision, recall, etc.)
  4. Generate comparison visualizations
  5. Create detailed reports and recommendations
  6. Save everything to results/comparison/
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/base_config.yaml',
        help='Path to configuration file (default: config/base_config.yaml)'
    )
    
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='results',
        help='Directory containing trained model results (default: results)'
    )
    
    parser.add_argument(
        '--variants', 
        nargs='+', 
        default=['model_a', 'model_b', 'model_c', 'model_d'],
        help='Model variants to compare (default: all models)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    print("Starting Comprehensive Model Comparison")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Results directory: {args.results_dir}")
    print(f"Models to compare: {', '.join(args.variants)}")
    print(f"Verbose logging: {args.verbose}")
    print("=" * 60)
    
    try:
        # Initialize comparison runner
        logger.info("Initializing comparison runner...")
        runner = ComparisonRunner(args.config, args.results_dir)
        
        # Run full comparison
        logger.info("Starting comprehensive comparison...")
        results = runner.run_full_comparison(args.variants)
        
        # Print summary
        print("\n" + "=" * 60)
        print("✅ COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        eval_summary = results['evaluation_results'].get('evaluation_summary', {})
        print(f"📊 Models evaluated: {eval_summary.get('successful_evaluations', 0)}/{eval_summary.get('total_models', 0)}")
        print(f"🕒 Total time: {results['total_time_seconds']:.2f} seconds")
        print(f"📁 Results saved to: {runner.comparison_dir}")
        
        print("\n📋 Generated Files:")
        print(f"  📄 Summary report: {runner.comparison_dir}/summary_report.txt")
        print(f"  📊 Model rankings: {runner.comparison_dir}/model_rankings.csv")
        print(f"  📈 Visualizations: {runner.comparison_dir}/plots/")
        print(f"  📋 Detailed results: {runner.comparison_dir}/evaluation_results.json")
        
        # Show best performers if available
        if 'comparison_analysis' in results and 'best_performers' in results['comparison_analysis']:
            best_performers = results['comparison_analysis']['best_performers']
            print(f"\n🏆 Best Overall Model: {best_performers.get('macro_f1', {}).get('model', 'N/A')}")
            print(f"   Macro F1 Score: {best_performers.get('macro_f1', {}).get('value', 0):.3f}")
        
        # Show recommendations if available
        if 'comparison_analysis' in results and 'recommendations' in results['comparison_analysis']:
            recommendations = results['comparison_analysis']['recommendations']
            print(f"\n💡 Recommendations:")
            print(f"   Best Overall: {recommendations.get('best_overall', 'N/A')}")
            print(f"   Most Efficient: {recommendations.get('most_efficient', 'N/A')}")
        
        print("\n🎯 Next Steps:")
        print("  1. Review the summary report for detailed analysis")
        print("  2. Check visualizations in the plots directory")  
        print("  3. Use the rankings CSV for further analysis")
        print("  4. Deploy the best performing model for your use case")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you have:")
        print("  1. Trained models in the results directory")
        print("  2. Correct configuration file path")
        print("  3. Proper checkpoint files for each model")
        return 1
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        print(f"\n❌ Comparison failed: {e}")
        print(f"\n📋 Check comparison.log for detailed error information")
        return 1

if __name__ == "__main__":
    exit(main()) 