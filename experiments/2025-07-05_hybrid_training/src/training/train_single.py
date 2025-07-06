#!/usr/bin/env python3
"""
Single Model Training Script
Train individual model variants for hybrid training experiment.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from training.trainer import HybridTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train single hybrid model variant')
    parser.add_argument('--variant', type=str, required=True,
                       choices=['model_a', 'model_b', 'model_c', 'model_d'],
                       help='Model variant to train')
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    logger.info(f"Starting training for {args.variant}")
    
    try:
        trainer = HybridTrainer(args.config, args.variant)
        results = trainer.train()
        
        print(f"\n✅ Training completed for {args.variant}:")
        print(f"  Best F1 score: {results['best_metric']:.4f}")
        print(f"  Best epoch: {results['best_epoch']+1}")
        print(f"  Total time: {results['total_time']/3600:.2f} hours")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted for {args.variant}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in training {args.variant}: {e}")
        raise

if __name__ == "__main__":
    main() 