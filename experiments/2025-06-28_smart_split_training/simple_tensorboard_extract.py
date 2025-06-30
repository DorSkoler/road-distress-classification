import os
import json
from tensorflow.python.lib.io import tf_record
from tensorflow.core.util import event_pb2
import struct

def extract_tensorboard_events(event_file_path):
    """Extract events from TensorBoard event file."""
    events = []
    
    try:
        for event in tf_record.tf_record_iterator(event_file_path):
            event_pb = event_pb2.Event.FromString(event)
            
            if event_pb.HasField('summary'):
                for value in event_pb.summary.value:
                    if value.HasField('simple_value'):
                        events.append({
                            'step': event_pb.step,
                            'tag': value.tag,
                            'value': value.simple_value,
                            'wall_time': event_pb.wall_time
                        })
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return events

def main():
    event_file = "results/model_a/logs/events.out.tfevents.1751115418.DESKTOP-FS4JJ12.7248.0"
    
    if not os.path.exists(event_file):
        print(f"File not found: {event_file}")
        return
    
    print(f"Reading TensorBoard events from: {event_file}")
    
    events = extract_tensorboard_events(event_file)
    
    if not events:
        print("No events found")
        return
    
    print(f"Found {len(events)} events")
    
    # Group by tag
    metrics_by_tag = {}
    for event in events:
        tag = event['tag']
        if tag not in metrics_by_tag:
            metrics_by_tag[tag] = []
        metrics_by_tag[tag].append(event)
    
    print("\nMetrics by tag:")
    for tag, tag_events in metrics_by_tag.items():
        print(f"  {tag}: {len(tag_events)} events")
        # Show first few values
        for i, event in enumerate(tag_events[:3]):
            print(f"    Step {event['step']}: {event['value']:.4f}")
    
    # Convert to epoch format
    max_step = max(event['step'] for event in events)
    epoch_metrics = []
    
    for step in range(max_step + 1):
        epoch_data = {'epoch': step + 1}
        for tag, tag_events in metrics_by_tag.items():
            step_events = [e for e in tag_events if e['step'] == step]
            if step_events:
                epoch_data[tag] = step_events[0]['value']
        
        if len(epoch_data) > 1:
            epoch_metrics.append(epoch_data)
    
    # Save to JSON
    output_file = "extracted_metrics/model_a_tensorboard_metrics.json"
    os.makedirs("extracted_metrics", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(epoch_metrics, f, indent=2)
    
    print(f"\nSaved {len(epoch_metrics)} epochs to {output_file}")
    
    # Show first few epochs
    print("\nFirst 3 epochs:")
    for epoch in epoch_metrics[:3]:
        print(f"  Epoch {epoch['epoch']}: {epoch}")

if __name__ == "__main__":
    main() 