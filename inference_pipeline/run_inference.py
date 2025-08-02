#!/usr/bin/env python3
"""
Main Inference Pipeline Script
Date: 2025-08-01

Run road distress classification inference on arbitrary images with
confidence heatmap generation and detailed analysis.

Usage:
    python run_inference.py --image path/to/image.jpg
    python run_inference.py --image path/to/image.jpg --output results/
    python run_inference.py --batch path/to/images/ --output results/
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import time
from typing import List, Dict, Union

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.inference_engine import create_inference_engine
from src.heatmap_generator import HeatmapGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_output_directory(output_path: str) -> Path:
    """Setup output directory structure."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "heatmaps").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    
    return output_dir


def process_single_image(image_path: str, 
                        inference_engine,
                        heatmap_generator: HeatmapGenerator,
                        output_dir: Path,
                        save_visualizations: bool = True) -> Dict:
    """Process a single image and generate all outputs."""
    logger.info(f"Processing image: {image_path}")
    
    # Get image name for output files
    image_name = Path(image_path).stem
    
    # Run inference
    start_time = time.time()
    results = inference_engine.predict_single(image_path)
    inference_time = time.time() - start_time
    
    # Generate confidence map
    confidence_map, _ = inference_engine.get_damage_confidence_map(image_path)
    
    # Add timing information
    results['inference_time'] = inference_time
    results['image_path'] = str(image_path)
    results['image_name'] = image_name
    
    if save_visualizations:
        # Load original image
        original_image = results['resized_image']
        
        # Create visualizations
        logger.info("Generating visualizations...")
        
        # 1. Damage confidence heatmap
        damage_heatmap = heatmap_generator.create_damage_confidence_heatmap(
            original_image, confidence_map, results, 
            title=f"Damage Analysis: {image_name}"
        )
        
        # 2. Multi-class visualization
        multi_class_viz = heatmap_generator.create_multi_class_visualization(
            original_image, results
        )
        
        # 3. Comparison grid
        comparison_grid = heatmap_generator.create_comparison_grid(
            original_image, results, confidence_map
        )
        
        # Save visualizations
        heatmap_path = output_dir / "heatmaps" / f"{image_name}_heatmap.jpg"
        multiclass_path = output_dir / "visualizations" / f"{image_name}_multiclass.jpg"
        grid_path = output_dir / "visualizations" / f"{image_name}_comparison.jpg"
        
        heatmap_generator.save_visualization(damage_heatmap, str(heatmap_path))
        heatmap_generator.save_visualization(multi_class_viz, str(multiclass_path))
        heatmap_generator.save_visualization(comparison_grid, str(grid_path))
        
        # Add visualization paths to results
        results['visualizations'] = {
            'damage_heatmap': str(heatmap_path),
            'multi_class': str(multiclass_path),
            'comparison_grid': str(grid_path)
        }
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    # Save results JSON
    results_path = output_dir / "results" / f"{image_name}_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'image_path': results['image_path'],
        'image_name': results['image_name'],
        'inference_time': results['inference_time'],
        'probabilities': results['probabilities'].tolist(),
        'predictions': results['predictions'].tolist(),
        'confidence': results['confidence'].tolist(),
        'overall_confidence': results['overall_confidence'],
        'original_size': results['original_size'],
        'processed_size': results['processed_size'],
        'class_results': results['class_results']
    }
    
    if save_visualizations:
        json_results['visualizations'] = results['visualizations']
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    return results


def process_batch_images(image_dir: str,
                        inference_engine,
                        heatmap_generator: HeatmapGenerator,
                        output_dir: Path,
                        save_visualizations: bool = True) -> List[Dict]:
    """Process a batch of images from a directory."""
    image_dir = Path(image_dir)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in image_dir.rglob('*') if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning(f"No image files found in {image_dir}")
        return []
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    all_results = []
    for i, image_path in enumerate(image_files):
        logger.info(f"Processing image {i+1}/{len(image_files)}: {image_path.name}")
        
        try:
            results = process_single_image(
                str(image_path), inference_engine, heatmap_generator, 
                output_dir, save_visualizations
            )
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            continue
    
    # Create batch summary
    summary = create_batch_summary(all_results)
    summary_path = output_dir / "batch_summary.json"
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Batch summary saved to {summary_path}")
    
    return all_results


def create_batch_summary(results: List[Dict]) -> Dict:
    """Create a summary of batch processing results."""
    if not results:
        return {"total_images": 0, "successful_predictions": 0}
    
    # Calculate statistics
    total_images = len(results)
    damage_predictions = sum(1 for r in results if r['class_results']['damage']['prediction'])
    occlusion_predictions = sum(1 for r in results if r['class_results']['occlusion']['prediction'])
    crop_predictions = sum(1 for r in results if r['class_results']['crop']['prediction'])
    
    avg_inference_time = sum(r['inference_time'] for r in results) / total_images
    avg_damage_confidence = sum(r['class_results']['damage']['probability'] for r in results) / total_images
    avg_overall_confidence = sum(r['overall_confidence'] for r in results) / total_images
    
    summary = {
        "total_images": total_images,
        "successful_predictions": total_images,
        "statistics": {
            "damage_predictions": damage_predictions,
            "occlusion_predictions": occlusion_predictions,
            "crop_predictions": crop_predictions,
            "damage_percentage": (damage_predictions / total_images) * 100,
            "occlusion_percentage": (occlusion_predictions / total_images) * 100,
            "crop_percentage": (crop_predictions / total_images) * 100
        },
        "performance": {
            "average_inference_time": avg_inference_time,
            "average_damage_confidence": avg_damage_confidence,
            "average_overall_confidence": avg_overall_confidence
        },
        "top_damage_images": sorted(
            [{"image": r['image_name'], "confidence": r['class_results']['damage']['probability']} 
             for r in results if r['class_results']['damage']['prediction']],
            key=lambda x: x['confidence'], reverse=True
        )[:10]
    }
    
    return summary


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Road Distress Classification Inference Pipeline")
    
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--batch', type=str, help='Path to directory containing images')
    parser.add_argument('--output', type=str, default='output', help='Output directory path')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    parser.add_argument('--experiments-path', type=str, 
                       default='../experiments/2025-07-05_hybrid_training',
                       help='Path to experiments directory')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.batch:
        parser.error("Must specify either --image or --batch")
    
    if args.image and args.batch:
        parser.error("Cannot specify both --image and --batch")
    
    # Setup output directory
    output_dir = setup_output_directory(args.output)
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize inference components
    logger.info("Initializing inference engine...")
    try:
        inference_engine = create_inference_engine(args.experiments_path)
        heatmap_generator = HeatmapGenerator()
        logger.info("Inference engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        sys.exit(1)
    
    # Process images
    save_visualizations = not args.no_viz
    
    try:
        if args.image:
            # Single image processing
            results = process_single_image(
                args.image, inference_engine, heatmap_generator, 
                output_dir, save_visualizations
            )
            
            # Print summary
            print("\n" + "="*50)
            print("INFERENCE RESULTS")
            print("="*50)
            print(f"Image: {results['image_name']}")
            print(f"Inference Time: {results['inference_time']:.3f}s")
            print(f"Overall Confidence: {results['overall_confidence']:.3f}")
            print("\nClass Predictions:")
            for class_name, class_result in results['class_results'].items():
                status = "✓" if class_result['prediction'] else "✗"
                print(f"  {status} {class_name.capitalize()}: {class_result['probability']:.3f}")
            
        else:
            # Batch processing
            results = process_batch_images(
                args.batch, inference_engine, heatmap_generator,
                output_dir, save_visualizations
            )
            
            # Print batch summary
            if results:
                summary = create_batch_summary(results)
                print("\n" + "="*50)
                print("BATCH PROCESSING RESULTS")
                print("="*50)
                print(f"Total Images: {summary['total_images']}")
                print(f"Damage Detected: {summary['statistics']['damage_predictions']} ({summary['statistics']['damage_percentage']:.1f}%)")
                print(f"Occlusion Detected: {summary['statistics']['occlusion_predictions']} ({summary['statistics']['occlusion_percentage']:.1f}%)")
                print(f"Crop Issues: {summary['statistics']['crop_predictions']} ({summary['statistics']['crop_percentage']:.1f}%)")
                print(f"Average Inference Time: {summary['performance']['average_inference_time']:.3f}s")
                print(f"Average Damage Confidence: {summary['performance']['average_damage_confidence']:.3f}")
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)
    
    print(f"\nAll results saved to: {output_dir}")
    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()