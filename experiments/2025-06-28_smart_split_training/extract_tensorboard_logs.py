import os
import json
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
from typing import Dict, List, Any


def extract_tensorboard_metrics(event_file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract metrics from TensorBoard event file.
    
    Args:
        event_file_path: Path to the TensorBoard event file
        
    Returns:
        Dictionary containing extracted metrics organized by tag
    """
    try:
        # Load the event file
        ea = EventAccumulator(event_file_path)
        ea.Reload()
        
        # Get all scalar tags
        scalar_tags = ea.Tags()['scalars']
        
        metrics = {}
        
        for tag in scalar_tags:
            # Get all events for this tag
            events = ea.Scalars(tag)
            
            # Convert to list of dictionaries
            tag_metrics = []
            for event in events:
                tag_metrics.append({
                    'step': event.step,
                    'value': event.value,
                    'wall_time': event.wall_time
                })
            
            metrics[tag] = tag_metrics
        
        return metrics
        
    except Exception as e:
        print(f"Error reading TensorBoard file {event_file_path}: {str(e)}")
        return {}


def convert_to_epoch_metrics(tensorboard_metrics: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Convert TensorBoard metrics to epoch-based format.
    
    Args:
        tensorboard_metrics: Metrics extracted from TensorBoard
        
    Returns:
        List of epoch metrics
    """
    # Find the maximum number of steps
    max_steps = 0
    for tag, events in tensorboard_metrics.items():
        if events:
            max_steps = max(max_steps, max(event['step'] for event in events))
    
    epoch_metrics = []
    
    # For each step (epoch), collect all metrics
    for step in range(max_steps + 1):
        epoch_data = {'epoch': step + 1}
        
        for tag, events in tensorboard_metrics.items():
            # Find the event for this step
            step_events = [e for e in events if e['step'] == step]
            if step_events:
                epoch_data[tag] = step_events[0]['value']
        
        if len(epoch_data) > 1:  # Only add if we have metrics for this epoch
            epoch_metrics.append(epoch_data)
    
    return epoch_metrics


def main():
    parser = argparse.ArgumentParser(description='Extract metrics from TensorBoard logs')
    parser.add_argument('--log_dir', type=str, default='results/model_a/logs', 
                       help='Directory containing TensorBoard logs')
    parser.add_argument('--output_dir', type=str, default='extracted_metrics', 
                       help='Output directory for extracted metrics')
    parser.add_argument('--format', type=str, choices=['json', 'csv', 'both'], default='both',
                       help='Output format')
    
    args = parser.parse_args()
    
    # Find TensorBoard event files
    event_files = []
    for root, dirs, files in os.walk(args.log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print(f"No TensorBoard event files found in {args.log_dir}")
        return
    
    print(f"Found {len(event_files)} TensorBoard event files:")
    for f in event_files:
        print(f"  {f}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each event file
    for event_file in event_files:
        print(f"\nProcessing {event_file}...")
        
        # Extract metrics
        tensorboard_metrics = extract_tensorboard_metrics(event_file)
        
        if not tensorboard_metrics:
            print(f"No metrics found in {event_file}")
            continue
        
        # Convert to epoch format
        epoch_metrics = convert_to_epoch_metrics(tensorboard_metrics)
        
        if not epoch_metrics:
            print(f"No epoch metrics found in {event_file}")
            continue
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(event_file))[0]
        
        # Save in requested format
        if args.format in ['json', 'both']:
            json_file = os.path.join(args.output_dir, f"{base_name}_metrics.json")
            with open(json_file, 'w') as f:
                json.dump(epoch_metrics, f, indent=2)
            print(f"Saved JSON metrics to {json_file}")
        
        if args.format in ['csv', 'both']:
            csv_file = os.path.join(args.output_dir, f"{base_name}_metrics.csv")
            df = pd.DataFrame(epoch_metrics)
            df.to_csv(csv_file, index=False)
            print(f"Saved CSV metrics to {csv_file}")
        
        # Print summary
        print(f"Extracted {len(epoch_metrics)} epochs with metrics:")
        for tag in tensorboard_metrics.keys():
            print(f"  - {tag}")
        
        # Print first few epochs as example
        print("\nFirst 3 epochs:")
        for i, epoch in enumerate(epoch_metrics[:3]):
            print(f"  Epoch {epoch['epoch']}: {epoch}")


if __name__ == "__main__":
    main() 