import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import torch
import yaml
from dual_input_model import create_model_variant, get_model_summary


def create_architecture_diagram():
    """Create a comprehensive architecture diagram for all model variants."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define colors
    colors = {
        'input': '#E8F4FD',
        'backbone': '#FFE6E6', 
        'encoder': '#E6F3FF',
        'fusion': '#FFF0E6',
        'classifier': '#E6FFE6',
        'output': '#F0E6FF',
        'flow': '#666666'
    }
    
    # Model A: Images Only
    ax1 = plt.subplot(2, 2, 1)
    draw_single_input_model(ax1, "Model A: Images Only", colors)
    
    # Model B: Images + Masks  
    ax2 = plt.subplot(2, 2, 2)
    draw_dual_input_model(ax2, "Model B: Images + Masks", colors)
    
    # Model C: Augmented Images Only
    ax3 = plt.subplot(2, 2, 3)
    draw_single_input_model(ax3, "Model C: Augmented Images Only", colors, augmented=True)
    
    # Model D: Augmented Images + Masks
    ax4 = plt.subplot(2, 2, 4)
    draw_dual_input_model(ax4, "Model D: Augmented Images + Masks", colors, augmented=True)
    
    plt.tight_layout()
    plt.savefig('model_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def draw_single_input_model(ax, title, colors, augmented=False):
    """Draw single-input model architecture."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Input
    input_label = "Augmented Images" if augmented else "Images"
    input_box = FancyBboxPatch((1, 10), 8, 1, boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], edgecolor='black', linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(5, 10.5, f"{input_label}\n(B, 3, 512, 512)", ha='center', va='center', fontsize=10, fontweight='bold')
    
    # EfficientNet-B3 Backbone
    backbone_box = FancyBboxPatch((1, 7.5), 8, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=colors['backbone'], edgecolor='black', linewidth=1.5)
    ax.add_patch(backbone_box)
    ax.text(5, 8.25, "EfficientNet-B3 Backbone\n(Feature Extraction)\n→ (B, 1536)", ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Classification Head
    classifier_box = FancyBboxPatch((1, 4.5), 8, 2, boxstyle="round,pad=0.1",
                                   facecolor=colors['classifier'], edgecolor='black', linewidth=1.5)
    ax.add_patch(classifier_box)
    ax.text(5, 5.5, "Classification Head\nLinear(1536 → 512) + ReLU + Dropout\nLinear(512 → 3)", 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output
    output_box = FancyBboxPatch((1, 2), 8, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['output'], edgecolor='black', linewidth=1.5)
    ax.add_patch(output_box)
    ax.text(5, 2.5, "Output Logits\n(B, 3) → [Damaged, Occlusion, Cropped]", 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Flow arrows
    draw_arrow(ax, (5, 10), (5, 9), colors['flow'])
    draw_arrow(ax, (5, 7.5), (5, 6.5), colors['flow'])
    draw_arrow(ax, (5, 4.5), (5, 3), colors['flow'])
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def draw_dual_input_model(ax, title, colors, augmented=False):
    """Draw dual-input model architecture."""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Input Images
    input_label = "Augmented Images" if augmented else "Images"
    image_box = FancyBboxPatch((0.5, 12), 4, 1, boxstyle="round,pad=0.1",
                              facecolor=colors['input'], edgecolor='black', linewidth=1.5)
    ax.add_patch(image_box)
    ax.text(2.5, 12.5, f"{input_label}\n(B, 3, 512, 512)", ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Input Masks
    mask_label = "Augmented Masks" if augmented else "Road Masks"
    mask_box = FancyBboxPatch((7.5, 12), 4, 1, boxstyle="round,pad=0.1",
                             facecolor=colors['input'], edgecolor='black', linewidth=1.5)
    ax.add_patch(mask_box)
    ax.text(9.5, 12.5, f"{mask_label}\n(B, 1, 512, 512)", ha='center', va='center', fontsize=9, fontweight='bold')
    
    # EfficientNet-B3 Backbone
    backbone_box = FancyBboxPatch((0.5, 9.5), 4, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=colors['backbone'], edgecolor='black', linewidth=1.5)
    ax.add_patch(backbone_box)
    ax.text(2.5, 10.25, "EfficientNet-B3\nBackbone\n→ (B, 1536)", ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Simple Mask Encoder
    encoder_box = FancyBboxPatch((7.5, 9.5), 4, 1.5, boxstyle="round,pad=0.1",
                                facecolor=colors['encoder'], edgecolor='black', linewidth=1.5)
    ax.add_patch(encoder_box)
    ax.text(9.5, 10.25, "Simple CNN\nMask Encoder\n→ (B, 256)", ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Attention Fusion
    fusion_box = FancyBboxPatch((3, 6.5), 6, 2, boxstyle="round,pad=0.1",
                               facecolor=colors['fusion'], edgecolor='black', linewidth=1.5)
    ax.add_patch(fusion_box)
    ax.text(6, 7.5, "Attention Fusion Module\nImage Attention + Mask Attention\nWeighted Combination\n→ (B, 512)", 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Classification Head
    classifier_box = FancyBboxPatch((3, 3.5), 6, 2, boxstyle="round,pad=0.1",
                                   facecolor=colors['classifier'], edgecolor='black', linewidth=1.5)
    ax.add_patch(classifier_box)
    ax.text(6, 4.5, "Classification Head\nLinear(512 → 512) + ReLU + Dropout\nLinear(512 → 3)", 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Output
    output_box = FancyBboxPatch((3, 1), 6, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['output'], edgecolor='black', linewidth=1.5)
    ax.add_patch(output_box)
    ax.text(6, 1.5, "Output Logits\n(B, 3) → [Damaged, Occlusion, Cropped]", 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Flow arrows
    draw_arrow(ax, (2.5, 12), (2.5, 11), colors['flow'])  # Image to backbone
    draw_arrow(ax, (9.5, 12), (9.5, 11), colors['flow'])  # Mask to encoder
    draw_arrow(ax, (2.5, 9.5), (4.5, 8.5), colors['flow'])  # Backbone to fusion
    draw_arrow(ax, (9.5, 9.5), (7.5, 8.5), colors['flow'])  # Encoder to fusion
    draw_arrow(ax, (6, 6.5), (6, 5.5), colors['flow'])  # Fusion to classifier
    draw_arrow(ax, (6, 3.5), (6, 2), colors['flow'])  # Classifier to output
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def draw_arrow(ax, start, end, color):
    """Draw an arrow between two points."""
    arrow = patches.FancyArrowPatch(start, end, connectionstyle="arc3", 
                                   arrowstyle='->', mutation_scale=20, 
                                   color=color, linewidth=2)
    ax.add_patch(arrow)


def create_detailed_architecture_diagram():
    """Create a detailed architecture diagram showing internal components."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14)
    ax.set_title("Detailed Dual-Input Model Architecture", fontsize=16, fontweight='bold', pad=20)
    
    colors = {
        'input': '#E8F4FD',
        'backbone': '#FFE6E6', 
        'encoder': '#E6F3FF',
        'attention': '#FFF0E6',
        'fusion': '#FFEEEE',
        'classifier': '#E6FFE6',
        'output': '#F0E6FF',
        'flow': '#666666'
    }
    
    # Input layer
    img_input = FancyBboxPatch((1, 12), 3, 1, boxstyle="round,pad=0.1",
                              facecolor=colors['input'], edgecolor='black', linewidth=1.5)
    ax.add_patch(img_input)
    ax.text(2.5, 12.5, "Images\n(B, 3, 512, 512)", ha='center', va='center', fontweight='bold')
    
    mask_input = FancyBboxPatch((12, 12), 3, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['input'], edgecolor='black', linewidth=1.5)
    ax.add_patch(mask_input)
    ax.text(13.5, 12.5, "Masks\n(B, 1, 512, 512)", ha='center', va='center', fontweight='bold')
    
    # EfficientNet-B3 details
    efficientnet = FancyBboxPatch((0.5, 9.5), 4, 2, boxstyle="round,pad=0.1",
                                 facecolor=colors['backbone'], edgecolor='black', linewidth=1.5)
    ax.add_patch(efficientnet)
    ax.text(2.5, 10.5, "EfficientNet-B3\n• MBConv Blocks\n• Squeeze-Excitation\n• Global AvgPool\n→ (B, 1536)", 
            ha='center', va='center', fontweight='bold')
    
    # Mask encoder details
    mask_encoder = FancyBboxPatch((11.5, 9.5), 4, 2, boxstyle="round,pad=0.1",
                                 facecolor=colors['encoder'], edgecolor='black', linewidth=1.5)
    ax.add_patch(mask_encoder)
    ax.text(13.5, 10.5, "Simple CNN Encoder\n• 4 Conv2D + BatchNorm\n• Global AvgPool\n• Linear + Dropout\n→ (B, 256)", 
            ha='center', va='center', fontweight='bold')
    
    # Attention mechanism
    img_attention = FancyBboxPatch((1, 6.5), 3, 1.5, boxstyle="round,pad=0.1",
                                  facecolor=colors['attention'], edgecolor='black', linewidth=1.5)
    ax.add_patch(img_attention)
    ax.text(2.5, 7.25, "Image Attention\nLinear→ReLU→Linear\n→Sigmoid", ha='center', va='center', fontweight='bold')
    
    mask_attention = FancyBboxPatch((12, 6.5), 3, 1.5, boxstyle="round,pad=0.1",
                                   facecolor=colors['attention'], edgecolor='black', linewidth=1.5)
    ax.add_patch(mask_attention)
    ax.text(13.5, 7.25, "Mask Attention\nLinear→ReLU→Linear\n→Sigmoid", ha='center', va='center', fontweight='bold')
    
    # Fusion layer
    fusion = FancyBboxPatch((6, 4.5), 4, 2, boxstyle="round,pad=0.1",
                           facecolor=colors['fusion'], edgecolor='black', linewidth=1.5)
    ax.add_patch(fusion)
    ax.text(8, 5.5, "Attention Fusion\n• Weighted Combination\n• Concatenation\n• Linear + ReLU + Dropout\n→ (B, 512)", 
            ha='center', va='center', fontweight='bold')
    
    # Classification head
    classifier = FancyBboxPatch((6, 2), 4, 1.5, boxstyle="round,pad=0.1",
                               facecolor=colors['classifier'], edgecolor='black', linewidth=1.5)
    ax.add_patch(classifier)
    ax.text(8, 2.75, "Classification Head\nLinear(512→512) + ReLU\nDropout + Linear(512→3)", 
            ha='center', va='center', fontweight='bold')
    
    # Output
    output = FancyBboxPatch((6.5, 0.2), 3, 0.8, boxstyle="round,pad=0.1",
                           facecolor=colors['output'], edgecolor='black', linewidth=1.5)
    ax.add_patch(output)
    ax.text(8, 0.6, "3 Classes\n[Damaged, Occlusion, Cropped]", ha='center', va='center', fontweight='bold')
    
    # Flow arrows with labels
    draw_arrow(ax, (2.5, 12), (2.5, 11.5), colors['flow'])
    draw_arrow(ax, (13.5, 12), (13.5, 11.5), colors['flow'])
    draw_arrow(ax, (2.5, 9.5), (2.5, 8), colors['flow'])
    draw_arrow(ax, (13.5, 9.5), (13.5, 8), colors['flow'])
    draw_arrow(ax, (4, 7.25), (6, 5.5), colors['flow'])
    draw_arrow(ax, (12, 7.25), (10, 5.5), colors['flow'])
    draw_arrow(ax, (8, 4.5), (8, 3.5), colors['flow'])
    draw_arrow(ax, (8, 2), (8, 1), colors['flow'])
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('detailed_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_model_comparison_table():
    """Create a comparison table of all model variants."""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model summaries
    variants = ['model_a', 'model_b', 'model_c', 'model_d']
    model_info = {}
    
    for variant in variants:
        model = create_model_variant(config, variant)
        summary = get_model_summary(model)
        variant_config = config.get('comparative_training', {}).get('variants', {}).get(variant, {})
        
        model_info[variant] = {
            'name': variant_config.get('name', variant),
            'description': variant_config.get('description', ''),
            'use_masks': variant_config.get('use_masks', False),
            'use_augmentation': variant_config.get('use_augmentation', False),
            'parameters': summary['total_parameters'],
            'size_mb': summary['model_size_mb']
        }
    
    # Create comparison figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Table data
    headers = ['Model', 'Description', 'Uses Masks', 'Uses Augmentation', 'Parameters', 'Size (MB)']
    data = []
    
    for variant, info in model_info.items():
        data.append([
            variant.upper(),
            info['description'],
            '✓' if info['use_masks'] else '✗',
            '✓' if info['use_augmentation'] else '✗',
            f"{info['parameters']:,}",
            f"{info['size_mb']:.1f}"
        ])
    
    # Create table
    table = ax.table(cellText=data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title('Model Variants Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('model_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Generate all architecture visualizations."""
    print("Creating architecture visualizations...")
    
    # Create comparison of all 4 model variants
    create_architecture_diagram()
    print("✓ Created model_architecture_comparison.png")
    
    # Create detailed architecture diagram
    create_detailed_architecture_diagram()
    print("✓ Created detailed_architecture.png")
    
    # Create model comparison table
    create_model_comparison_table()
    print("✓ Created model_comparison_table.png")
    
    print("\nAll visualizations saved successfully!")


if __name__ == "__main__":
    main() 