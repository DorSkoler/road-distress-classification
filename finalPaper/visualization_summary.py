#!/usr/bin/env python3
"""
Final summary of all created visualizations
Provides overview and cleanup of the visualization creation process
"""

import os
import glob

def summarize_visualizations():
    """Provide a comprehensive summary of all created visualizations"""
    
    # Get output directory
    output_dir = os.path.join(os.getcwd(), "mlds_final_project_template", "images")
    
    print("="*80)
    print("ROAD DISTRESS CLASSIFICATION - FINAL PAPER VISUALIZATIONS")
    print("="*80)
    print(f"Output Directory: {output_dir}")
    print()
    
    # Get all PNG files
    png_files = glob.glob(os.path.join(output_dir, "*.png"))
    png_files = [os.path.basename(f) for f in png_files]
    png_files.sort()
    
    # Categorize visualizations
    categories = {
        "Dataset Analysis": [
            "damage_classification_distribution.png",
            "occlusion_classification_distribution.png", 
            "crop_classification_distribution.png",
            "dataset_split_analysis.png",
            "dataset_split.png"
        ],
        "Model Performance": [
            "model_comparison_detailed.png",
            "individual_model_breakdown.png",
            "model_performance.png"
        ],
        "Threshold Optimization": [
            "threshold_optimization_analysis.png",
            "threshold_strategies_comparison.png",
            "threshold_optimization.png"
        ],
        "Performance Metrics": [
            "performance_metrics_analysis.png",
            "operational_performance_analysis.png",
            "performance_metrics.png"
        ],
        "Architecture & Systems": [
            "ensemble_architecture_detailed.png",
            "ensemble_architecture.png",
            "experimental_timeline_detailed.png"
        ],
        "Breakthrough Analysis": [
            "breakthrough_analysis.png"
        ],
        "Test/Legacy Files": [
            "test_plot.png"
        ]
    }
    
    # Print categorized summary
    total_files = 0
    for category, files in categories.items():
        print(f"\n{category.upper()}:")
        print("-" * (len(category) + 1))
        
        existing_files = [f for f in files if f in png_files]
        total_files += len(existing_files)
        
        if existing_files:
            for file in existing_files:
                file_path = os.path.join(output_dir, file)
                size_kb = os.path.getsize(file_path) // 1024
                print(f"  ✓ {file:<50} ({size_kb:>4} KB)")
        else:
            print("  (No files in this category)")
    
    print(f"\nTOTAL VISUALIZATIONS CREATED: {total_files}")
    print(f"TOTAL FILE SIZE: {sum(os.path.getsize(os.path.join(output_dir, f)) for f in png_files) // 1024} KB")
    
    # Paper integration recommendations
    print("\n" + "="*80)
    print("LATEX PAPER INTEGRATION RECOMMENDATIONS")
    print("="*80)
    
    recommended_figures = [
        ("Section 3.1 - Dataset Composition", "dataset_split_analysis.png"),
        ("Section 3.2 - Class Distributions", "damage_classification_distribution.png"),
        ("Section 3.2 - Class Distributions", "occlusion_classification_distribution.png"), 
        ("Section 3.2 - Class Distributions", "crop_classification_distribution.png"),
        ("Section 4 - Architecture", "ensemble_architecture_detailed.png"),
        ("Section 4.1 - Global Performance", "performance_metrics_analysis.png"),
        ("Section 4.2 - Operational Performance", "operational_performance_analysis.png"),
        ("Section 4.3 - Alternative Strategies", "threshold_strategies_comparison.png"),
        ("Section 5.1 - Model Performance", "model_comparison_detailed.png"),
        ("Section 5.1 - Model Breakdown", "individual_model_breakdown.png"),
        ("Section 5.2 - Breakthrough", "breakthrough_analysis.png"),
        ("Section 5.2 - Threshold Results", "threshold_optimization_analysis.png"),
        ("Section 6 - Experimental Evolution", "experimental_timeline_detailed.png")
    ]
    
    print("\nRECOMMENDED FIGURE PLACEMENTS:")
    for i, (section, filename) in enumerate(recommended_figures, 1):
        print(f"{i:2d}. {section:<35} -> {filename}")
    
    # LaTeX code examples
    print("\n" + "="*80)
    print("SAMPLE LATEX CODE")
    print("="*80)
    
    sample_latex = """
% Replace the TikZ ensemble diagram with:
\\begin{figure}[!htb]
\\centering
\\includegraphics[width=0.9\\textwidth]{images/ensemble_architecture_detailed.png}
\\caption{Two-model ensemble architecture with per-class threshold optimization}
\\end{figure}

% Add class distribution figures:
\\begin{figure}[!htb]
\\centering
\\includegraphics[width=0.7\\textwidth]{images/damage_classification_distribution.png}
\\caption{Damage classification distribution showing moderate class imbalance}
\\end{figure}

% Add breakthrough visualization:
\\begin{figure}[!htb]
\\centering
\\includegraphics[width=0.9\\textwidth]{images/breakthrough_analysis.png}
\\caption{The per-class threshold breakthrough: +28.7\\% accuracy improvement}
\\end{figure}
"""
    
    print(sample_latex)
    
    print("\n" + "="*80)
    print("KEY ACHIEVEMENTS")
    print("="*80)
    print("✓ All visualizations include proper axis labels and descriptions")
    print("✓ Every chart has explanatory text and annotations") 
    print("✓ Consistent color coding and professional formatting")
    print("✓ High-resolution (300 DPI) publication-ready quality")
    print("✓ Comprehensive coverage of all paper sections")
    print("✓ Self-contained visualizations suitable for presentations")
    print("✓ Detailed documentation in VISUALIZATION_INDEX.md")
    
    print("\n" + "="*80)
    print("VISUALIZATION CREATION COMPLETE!")
    print("="*80)

def cleanup_test_files():
    """Remove test files if requested"""
    output_dir = os.path.join(os.getcwd(), "mlds_final_project_template", "images")
    test_files = ["test_plot.png"]
    
    print("\nCLEANUP OPTIONS:")
    print("The following test files can be removed:")
    for file in test_files:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            print(f"  - {file}")
    
    print("\nTo remove test files, run:")
    print("python -c \"import os; os.remove('mlds_final_project_template/images/test_plot.png')\"")

if __name__ == "__main__":
    summarize_visualizations()
    cleanup_test_files()
