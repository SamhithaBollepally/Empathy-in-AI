import pandas as pd
import numpy as np
import json
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Configuration
BASE_PATH = '/Volumes/Reading/Projects/Empathy-LLM/'

EMOTIONS = ['afraid', 'angry', 'anxious', 'devastated', 'lonely', 'sad', 
            'terrified', 'furious', 'grateful', 'hopeful', 'faithful']

def load_data():
    """Load probing results and similarity analysis results"""
    print("Loading data...")
    
    # Load probing results
    with open(BASE_PATH + 'probing_results.json', 'r') as f:
        probing_data = json.load(f)
    
    # Load similarity analysis
    similarity_df = pd.read_csv(BASE_PATH + 'layer_accuracy_summary.csv')
    
    print(f"✓ Loaded probing results for {len(probing_data['context_layer_accuracies'])} emotions")
    print(f"✓ Loaded similarity results for {len(similarity_df)} layers")
    
    return probing_data, similarity_df

def calculate_correlations(probing_data, similarity_df):
    """Calculate Spearman correlation for each emotion"""
    print("\nCalculating Spearman correlations...")
    print("="*80)
    
    results = []
    
    for emotion in EMOTIONS:
        if emotion not in probing_data['context_layer_accuracies']:
            print(f"⚠ {emotion} not found in probing results")
            continue
        
        # Get probing accuracies per layer
        probing_accs = probing_data['context_layer_accuracies'][emotion]
        
        # Get similarity accuracies per layer
        similarity_col = f'acc_{emotion}'
        
        # Align data by layer
        layers = []
        probing_values = []
        similarity_values = []
        
        for layer in range(29):
            layer_str = str(layer)
            if layer_str in probing_accs and layer in similarity_df['layer'].values:
                layers.append(layer)
                probing_values.append(probing_accs[layer_str])
                similarity_values.append(
                    similarity_df[similarity_df['layer'] == layer][similarity_col].values[0]
                )
        
        # Calculate Spearman correlation
        if len(probing_values) > 2:
            correlation, p_value = spearmanr(probing_values, similarity_values)
            
            results.append({
                'emotion': emotion,
                'spearman_rho': correlation,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'n_layers': len(layers),
                'probing_mean': np.mean(probing_values),
                'similarity_mean': np.mean(similarity_values)
            })
            
            # Print result
            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{emotion:<12}: ρ = {correlation:6.3f}, p = {p_value:.4f} {sig_marker}")
    
    print("="*80)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    
    return pd.DataFrame(results)

def analyze_results(results_df):
    """Analyze correlation results"""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"\n1. OVERALL STATISTICS:")
    print(f"   Mean correlation: {results_df['spearman_rho'].mean():.3f}")
    print(f"   Median correlation: {results_df['spearman_rho'].median():.3f}")
    print(f"   Std deviation: {results_df['spearman_rho'].std():.3f}")
    print(f"   Significant correlations: {results_df['significant'].sum()}/{len(results_df)}")
    
    # Positive vs negative correlations
    positive = results_df[results_df['spearman_rho'] > 0]
    negative = results_df[results_df['spearman_rho'] < 0]
    print(f"\n2. CORRELATION DIRECTION:")
    print(f"   Positive correlations: {len(positive)} emotions")
    print(f"   Negative correlations: {len(negative)} emotions")
    
    # Strongest correlations
    print(f"\n3. STRONGEST POSITIVE CORRELATIONS:")
    top_positive = results_df.nlargest(5, 'spearman_rho')
    for _, row in top_positive.iterrows():
        sig = "✓" if row['significant'] else " "
        print(f"   {sig} {row['emotion']:<12}: ρ = {row['spearman_rho']:6.3f} (p={row['p_value']:.4f})")
    
    print(f"\n4. STRONGEST NEGATIVE CORRELATIONS:")
    top_negative = results_df.nsmallest(5, 'spearman_rho')
    for _, row in top_negative.iterrows():
        sig = "✓" if row['significant'] else " "
        print(f"   {sig} {row['emotion']:<12}: ρ = {row['spearman_rho']:6.3f} (p={row['p_value']:.4f})")
    
    # Interpretation
    print(f"\n5. INTERPRETATION:")
    avg_corr = results_df['spearman_rho'].mean()
    if avg_corr > 0.3:
        print(f"   ✓ MODERATE POSITIVE correlation (ρ={avg_corr:.3f})")
        print(f"   → Layers good at probing ARE also good at response matching")
    elif avg_corr > 0:
        print(f"   ~ WEAK POSITIVE correlation (ρ={avg_corr:.3f})")
        print(f"   → Some alignment between probing and response matching")
    elif avg_corr > -0.3:
        print(f"   ~ WEAK NEGATIVE correlation (ρ={avg_corr:.3f})")
        print(f"   → Little relationship between probing and response matching")
    else:
        print(f"   ✗ MODERATE NEGATIVE correlation (ρ={avg_corr:.3f})")
        print(f"   → Layers good at probing are BAD at response matching")
    
    # Performance comparison
    print(f"\n6. PERFORMANCE COMPARISON:")
    print(f"   Average probing accuracy:   {results_df['probing_mean'].mean():.2%}")
    print(f"   Average similarity accuracy: {results_df['similarity_mean'].mean():.2%}")
    print(f"   Difference: {(results_df['probing_mean'].mean() - results_df['similarity_mean'].mean()):.2%}")
    
    print("\n" + "="*80)
    
    return results_df

def save_results(results_df):
    """Save correlation results"""
    output_file = BASE_PATH + 'correlation_analysis.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved results to {output_file}")

def plot_correlations(probing_data, similarity_df, results_df):
    """Create scatter plots for top correlations"""
    print("\n📊 Creating visualization data...")
    
    # Select top 4 emotions by absolute correlation
    results_df['abs_corr'] = results_df['spearman_rho'].abs()
    top_emotions = results_df.nlargest(4, 'abs_corr')['emotion'].tolist()
    
    print(f"\nTop 4 emotions by correlation strength:")
    for emotion in top_emotions:
        row = results_df[results_df['emotion'] == emotion].iloc[0]
        print(f"  {emotion:<12}: ρ = {row['spearman_rho']:6.3f}")
    
    # Prepare data for plotting
    plot_data = []
    for emotion in top_emotions:
        probing_accs = probing_data['context_layer_accuracies'][emotion]
        similarity_col = f'acc_{emotion}'
        
        layers = []
        probing_vals = []
        similarity_vals = []
        
        for layer in range(29):
            layer_str = str(layer)
            if layer_str in probing_accs and layer in similarity_df['layer'].values:
                layers.append(layer)
                probing_vals.append(probing_accs[layer_str])
                similarity_vals.append(
                    similarity_df[similarity_df['layer'] == layer][similarity_col].values[0]
                )
        
        plot_data.append({
            'emotion': emotion,
            'layers': layers,
            'probing': probing_vals,
            'similarity': similarity_vals,
            'rho': results_df[results_df['emotion'] == emotion]['spearman_rho'].values[0]
        })
    
    print(f"\n✓ Visualization data prepared for {len(plot_data)} emotions")
    print("  (Use matplotlib to plot: probing_accuracy vs similarity_accuracy)")
    
    return plot_data

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("SPEARMAN CORRELATION ANALYSIS")
    print("Probing Accuracy vs Response Similarity Accuracy")
    print("="*80)
    
    # Load data
    probing_data, similarity_df = load_data()
    
    # Calculate correlations
    results_df = calculate_correlations(probing_data, similarity_df)
    
    # Analyze results
    results_df = analyze_results(results_df)
    
    # Save results
    save_results(results_df)
    
    # Prepare visualization data
    plot_data = plot_correlations(probing_data, similarity_df, results_df)
    
    print("\n✓ Analysis complete!")
