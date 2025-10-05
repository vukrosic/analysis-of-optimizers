#!/usr/bin/env python3
"""
Advanced visualization tools for token selection analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os

class AdvancedVisualizationTools:
    """Advanced visualization tools for token selection analysis"""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#9467bd',
            'info': '#17a2b8'
        }
        
    def create_interactive_token_selection(self, analysis_history: List[Dict], 
                                         save_path: str = None) -> str:
        """Create interactive HTML visualization of token selection"""
        
        # Extract data for visualization
        steps = [h['step'] for h in analysis_history]
        selection_data = []
        
        for history in analysis_history:
            if 'analysis' in history and 'selection_patterns' in history['analysis']:
                selection_freq = history['analysis']['selection_patterns']['selection_frequency']
                if isinstance(selection_freq, torch.Tensor):
                    selection_freq = selection_freq.cpu().numpy()
                selection_data.append(selection_freq.tolist())
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add heatmap
        if selection_data:
            fig.add_trace(go.Heatmap(
                z=selection_data,
                x=list(range(len(selection_data[0]) if selection_data else 0)),
                y=steps[:len(selection_data)],
                colorscale='Viridis',
                name='Selection Frequency'
            ))
        
        # Update layout
        fig.update_layout(
            title='Token Selection Patterns Over Time',
            xaxis_title='Token Position',
            yaxis_title='Training Step',
            width=800,
            height=600
        )
        
        # Save as HTML
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'interactive_token_selection.html')
        
        fig.write_html(save_path)
        return save_path
    
    def create_attention_pattern_evolution(self, analysis_history: List[Dict],
                                        save_path: str = None) -> str:
        """Create interactive visualization of attention pattern evolution"""
        
        # Extract attention patterns
        attention_data = []
        steps = []
        
        for history in analysis_history:
            if 'analysis' in history and 'attention_patterns' in history['analysis']:
                attn_patterns = history['analysis']['attention_patterns']
                if isinstance(attn_patterns, torch.Tensor):
                    attn_patterns = attn_patterns.cpu().numpy()
                
                # Average across batch and heads for visualization
                if len(attn_patterns.shape) == 4:  # [batch, heads, seq_len, seq_len]
                    attn_patterns = attn_patterns.mean(axis=(0, 1))
                elif len(attn_patterns.shape) == 3:  # [batch, seq_len, seq_len]
                    attn_patterns = attn_patterns.mean(axis=0)
                
                attention_data.append(attn_patterns.tolist())
                steps.append(history['step'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Attention Pattern Evolution', 'Selection Frequency',
                          'Pattern Entropy', 'Efficiency Metrics'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Add attention pattern heatmap
        if attention_data:
            fig.add_trace(go.Heatmap(
                z=attention_data,
                x=list(range(len(attention_data[0][0]) if attention_data else 0)),
                y=steps[:len(attention_data)],
                colorscale='Blues',
                name='Attention Patterns'
            ), row=1, col=1)
        
        # Add selection frequency plot
        selection_freq_data = []
        for history in analysis_history:
            if 'analysis' in history and 'selection_patterns' in history['analysis']:
                freq = history['analysis']['selection_patterns']['selection_frequency']
                if isinstance(freq, torch.Tensor):
                    freq = freq.cpu().numpy()
                selection_freq_data.append(freq.mean())
        
        if selection_freq_data:
            fig.add_trace(go.Scatter(
                x=steps[:len(selection_freq_data)],
                y=selection_freq_data,
                mode='lines+markers',
                name='Avg Selection Frequency',
                line=dict(color=self.colors['primary'])
            ), row=1, col=2)
        
        # Add pattern entropy plot
        entropy_data = []
        for history in analysis_history:
            if 'analysis' in history and 'selection_patterns' in history['analysis']:
                diversity = history['analysis']['selection_patterns'].get('selection_diversity', 0)
                entropy_data.append(diversity)
        
        if entropy_data:
            fig.add_trace(go.Scatter(
                x=steps[:len(entropy_data)],
                y=entropy_data,
                mode='lines+markers',
                name='Pattern Entropy',
                line=dict(color=self.colors['success'])
            ), row=2, col=1)
        
        # Add efficiency metrics plot
        efficiency_data = []
        for history in analysis_history:
            if 'analysis' in history and 'efficiency_metrics' in history['analysis']:
                efficiency = history['analysis']['efficiency_metrics'].get('cost_reduction', 0)
                efficiency_data.append(efficiency)
        
        if efficiency_data:
            fig.add_trace(go.Scatter(
                x=steps[:len(efficiency_data)],
                y=efficiency_data,
                mode='lines+markers',
                name='Cost Reduction',
                line=dict(color=self.colors['warning'])
            ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title='Attention Pattern Evolution Analysis',
            width=1200,
            height=800,
            showlegend=True
        )
        
        # Save as HTML
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'attention_pattern_evolution.html')
        
        fig.write_html(save_path)
        return save_path
    
    def create_indexer_behavior_analysis(self, analysis_history: List[Dict],
                                      save_path: str = None) -> str:
        """Create interactive visualization of indexer behavior"""
        
        # Extract indexer behavior data
        steps = [h['step'] for h in analysis_history]
        head_data = {}
        score_stats = []
        
        for history in analysis_history:
            if 'analysis' in history and 'indexer_behavior' in history['analysis']:
                behavior = history['analysis']['indexer_behavior']
                
                # Extract score statistics
                if 'score_stats' in behavior:
                    score_stats.append(behavior['score_stats'])
                
                # Extract head analysis
                if 'head_analysis' in behavior:
                    for head_idx, head_info in behavior['head_analysis'].items():
                        if head_idx not in head_data:
                            head_data[head_idx] = {'means': [], 'stds': [], 'sparsities': []}
                        head_data[head_idx]['means'].append(head_info['mean'])
                        head_data[head_idx]['stds'].append(head_info['std'])
                        head_data[head_idx]['sparsities'].append(head_info['sparsity'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Indexer Head Specialization', 'Score Statistics Over Time',
                          'Head Sparsity Patterns', 'Score Distribution'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Add head specialization plot
        for head_idx, data in head_data.items():
            fig.add_trace(go.Scatter(
                x=steps[:len(data['means'])],
                y=data['means'],
                mode='lines+markers',
                name=f'Head {head_idx}',
                error_y=dict(type='data', array=data['stds'], visible=True)
            ), row=1, col=1)
        
        # Add score statistics plot
        if score_stats:
            means = [s['mean'] for s in score_stats]
            stds = [s['std'] for s in score_stats]
            mins = [s['min'] for s in score_stats]
            maxs = [s['max'] for s in score_stats]
            
            fig.add_trace(go.Scatter(
                x=steps[:len(means)],
                y=means,
                mode='lines+markers',
                name='Mean Score',
                line=dict(color=self.colors['primary'])
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=steps[:len(mins)],
                y=mins,
                mode='lines',
                name='Min Score',
                line=dict(color=self.colors['danger'], dash='dash')
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=steps[:len(maxs)],
                y=maxs,
                mode='lines',
                name='Max Score',
                line=dict(color=self.colors['success'], dash='dash')
            ), row=1, col=2)
        
        # Add head sparsity patterns
        for head_idx, data in head_data.items():
            fig.add_trace(go.Scatter(
                x=steps[:len(data['sparsities'])],
                y=data['sparsities'],
                mode='lines+markers',
                name=f'Head {head_idx} Sparsity',
                showlegend=False
            ), row=2, col=1)
        
        # Add score distribution histogram
        if score_stats:
            all_scores = []
            for stats in score_stats:
                # Create synthetic score distribution based on stats
                scores = np.random.normal(stats['mean'], stats['std'], 1000)
                all_scores.extend(scores)
            
            fig.add_trace(go.Histogram(
                x=all_scores,
                nbinsx=50,
                name='Score Distribution',
                opacity=0.7
            ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title='Indexer Behavior Analysis',
            width=1200,
            height=800,
            showlegend=True
        )
        
        # Save as HTML
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'indexer_behavior_analysis.html')
        
        fig.write_html(save_path)
        return save_path
    
    def create_performance_correlation_analysis(self, analysis_history: List[Dict],
                                              performance_history: List[Dict],
                                              save_path: str = None) -> str:
        """Create interactive visualization of performance correlations"""
        
        # Extract performance data
        perf_steps = [p['step'] for p in performance_history]
        losses = [p['loss'] for p in performance_history]
        
        # Extract analysis data
        analysis_steps = [h['step'] for h in analysis_history]
        
        # Create correlation data
        correlation_data = {
            'sparsity_ratios': [],
            'efficiency_scores': [],
            'selection_diversities': [],
            'attention_efficiencies': [],
            'losses': [],
            'steps': []
        }
        
        for history in analysis_history:
            step = history['step']
            if step in perf_steps:
                perf_idx = perf_steps.index(step)
                loss = losses[perf_idx]
                
                if 'analysis' in history:
                    analysis = history['analysis']
                    
                    # Extract metrics
                    if 'selection_patterns' in analysis:
                        sparsity = analysis['selection_patterns'].get('sparsity_ratio', 0)
                        diversity = analysis['selection_patterns'].get('selection_diversity', 0)
                        correlation_data['sparsity_ratios'].append(sparsity)
                        correlation_data['selection_diversities'].append(diversity)
                    
                    if 'efficiency_metrics' in analysis:
                        efficiency = analysis['efficiency_metrics'].get('attention_efficiency', 0)
                        correlation_data['efficiency_scores'].append(efficiency)
                        correlation_data['attention_efficiencies'].append(efficiency)
                    
                    correlation_data['losses'].append(loss)
                    correlation_data['steps'].append(step)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sparsity vs Performance', 'Efficiency vs Performance',
                          'Selection Diversity vs Performance', 'Attention Efficiency vs Performance'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Add correlation plots
        if correlation_data['sparsity_ratios'] and correlation_data['losses']:
            fig.add_trace(go.Scatter(
                x=correlation_data['sparsity_ratios'],
                y=correlation_data['losses'],
                mode='markers',
                name='Sparsity vs Loss',
                marker=dict(color=self.colors['primary'], size=8)
            ), row=1, col=1)
        
        if correlation_data['efficiency_scores'] and correlation_data['losses']:
            fig.add_trace(go.Scatter(
                x=correlation_data['efficiency_scores'],
                y=correlation_data['losses'],
                mode='markers',
                name='Efficiency vs Loss',
                marker=dict(color=self.colors['success'], size=8)
            ), row=1, col=2)
        
        if correlation_data['selection_diversities'] and correlation_data['losses']:
            fig.add_trace(go.Scatter(
                x=correlation_data['selection_diversities'],
                y=correlation_data['losses'],
                mode='markers',
                name='Diversity vs Loss',
                marker=dict(color=self.colors['warning'], size=8)
            ), row=2, col=1)
        
        if correlation_data['attention_efficiencies'] and correlation_data['losses']:
            fig.add_trace(go.Scatter(
                x=correlation_data['attention_efficiencies'],
                y=correlation_data['losses'],
                mode='markers',
                name='Attention Efficiency vs Loss',
                marker=dict(color=self.colors['danger'], size=8)
            ), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title='Performance Correlation Analysis',
            width=1200,
            height=800,
            showlegend=True
        )
        
        # Save as HTML
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'performance_correlation_analysis.html')
        
        fig.write_html(save_path)
        return save_path
    
    def create_comprehensive_dashboard(self, analysis_history: List[Dict],
                                     performance_history: List[Dict],
                                     save_path: str = None) -> str:
        """Create comprehensive interactive dashboard"""
        
        # Create dashboard HTML
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Token Selection Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .dashboard {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                .card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .card h2 {{
                    margin-top: 0;
                    color: #333;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .metric {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 6px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #007bff;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <h1>Token Selection Analysis Dashboard</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{len(analysis_history)}</div>
                    <div class="metric-label">Analysis Steps</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(performance_history)}</div>
                    <div class="metric-label">Performance Steps</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{performance_history[-1]['loss']:.4f if performance_history else 'N/A'}</div>
                    <div class="metric-label">Final Loss</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{analysis_history[-1]['analysis']['selection_patterns']['sparsity_ratio']:.3f if analysis_history and 'analysis' in analysis_history[-1] else 'N/A'}</div>
                    <div class="metric-label">Sparsity Ratio</div>
                </div>
            </div>
            
            <div class="dashboard">
                <div class="card">
                    <h2>Token Selection Patterns</h2>
                    <div id="token-selection-plot"></div>
                </div>
                
                <div class="card">
                    <h2>Attention Pattern Evolution</h2>
                    <div id="attention-evolution-plot"></div>
                </div>
                
                <div class="card">
                    <h2>Indexer Behavior Analysis</h2>
                    <div id="indexer-behavior-plot"></div>
                </div>
                
                <div class="card">
                    <h2>Performance Correlations</h2>
                    <div id="performance-correlation-plot"></div>
                </div>
            </div>
            
            <script>
                // Token selection plot
                {self._generate_token_selection_js(analysis_history)}
                
                // Attention evolution plot
                {self._generate_attention_evolution_js(analysis_history)}
                
                // Indexer behavior plot
                {self._generate_indexer_behavior_js(analysis_history)}
                
                // Performance correlation plot
                {self._generate_performance_correlation_js(analysis_history, performance_history)}
            </script>
        </body>
        </html>
        """
        
        # Save dashboard
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'comprehensive_dashboard.html')
        
        with open(save_path, 'w') as f:
            f.write(dashboard_html)
        
        return save_path
    
    def _generate_token_selection_js(self, analysis_history: List[Dict]) -> str:
        """Generate JavaScript for token selection plot"""
        # Extract data
        steps = [h['step'] for h in analysis_history]
        selection_data = []
        
        for history in analysis_history:
            if 'analysis' in history and 'selection_patterns' in history['analysis']:
                selection_freq = history['analysis']['selection_patterns']['selection_frequency']
                if isinstance(selection_freq, torch.Tensor):
                    selection_freq = selection_freq.cpu().numpy()
                selection_data.append(selection_freq.tolist())
        
        return f"""
        var tokenSelectionData = {json.dumps(selection_data)};
        var tokenSelectionSteps = {json.dumps(steps[:len(selection_data)])};
        
        var tokenSelectionTrace = {{
            z: tokenSelectionData,
            x: Array.from({{length: tokenSelectionData[0]?.length || 0}}, (_, i) => i),
            y: tokenSelectionSteps,
            type: 'heatmap',
            colorscale: 'Viridis'
        }};
        
        var tokenSelectionLayout = {{
            title: 'Token Selection Patterns Over Time',
            xaxis: {{title: 'Token Position'}},
            yaxis: {{title: 'Training Step'}}
        }};
        
        Plotly.newPlot('token-selection-plot', [tokenSelectionTrace], tokenSelectionLayout);
        """
    
    def _generate_attention_evolution_js(self, analysis_history: List[Dict]) -> str:
        """Generate JavaScript for attention evolution plot"""
        # Extract data
        steps = [h['step'] for h in analysis_history]
        sparsity_ratios = []
        
        for history in analysis_history:
            if 'analysis' in history and 'selection_patterns' in history['analysis']:
                sparsity = history['analysis']['selection_patterns'].get('sparsity_ratio', 0)
                sparsity_ratios.append(sparsity)
        
        return f"""
        var attentionSteps = {json.dumps(steps[:len(sparsity_ratios)])};
        var sparsityRatios = {json.dumps(sparsity_ratios)};
        
        var attentionTrace = {{
            x: attentionSteps,
            y: sparsityRatios,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Sparsity Ratio',
            line: {{color: '#1f77b4'}}
        }};
        
        var attentionLayout = {{
            title: 'Attention Pattern Evolution',
            xaxis: {{title: 'Training Step'}},
            yaxis: {{title: 'Sparsity Ratio'}}
        }};
        
        Plotly.newPlot('attention-evolution-plot', [attentionTrace], attentionLayout);
        """
    
    def _generate_indexer_behavior_js(self, analysis_history: List[Dict]) -> str:
        """Generate JavaScript for indexer behavior plot"""
        # Extract data
        steps = [h['step'] for h in analysis_history]
        score_means = []
        
        for history in analysis_history:
            if 'analysis' in history and 'indexer_behavior' in history['analysis']:
                behavior = history['analysis']['indexer_behavior']
                if 'score_stats' in behavior:
                    score_means.append(behavior['score_stats']['mean'])
        
        return f"""
        var indexerSteps = {json.dumps(steps[:len(score_means)])};
        var scoreMeans = {json.dumps(score_means)};
        
        var indexerTrace = {{
            x: indexerSteps,
            y: scoreMeans,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Mean Score',
            line: {{color: '#ff7f0e'}}
        }};
        
        var indexerLayout = {{
            title: 'Indexer Behavior Over Time',
            xaxis: {{title: 'Training Step'}},
            yaxis: {{title: 'Score Value'}}
        }};
        
        Plotly.newPlot('indexer-behavior-plot', [indexerTrace], indexerLayout);
        """
    
    def _generate_performance_correlation_js(self, analysis_history: List[Dict], 
                                           performance_history: List[Dict]) -> str:
        """Generate JavaScript for performance correlation plot"""
        # Extract correlation data
        correlation_data = {'sparsity': [], 'loss': []}
        
        for history in analysis_history:
            step = history['step']
            perf_data = next((p for p in performance_history if p['step'] == step), None)
            
            if perf_data and 'analysis' in history:
                analysis = history['analysis']
                if 'selection_patterns' in analysis:
                    sparsity = analysis['selection_patterns'].get('sparsity_ratio', 0)
                    correlation_data['sparsity'].append(sparsity)
                    correlation_data['loss'].append(perf_data['loss'])
        
        return f"""
        var correlationData = {json.dumps(correlation_data)};
        
        var correlationTrace = {{
            x: correlationData.sparsity,
            y: correlationData.loss,
            type: 'scatter',
            mode: 'markers',
            name: 'Sparsity vs Loss',
            marker: {{color: '#2ca02c', size: 8}}
        }};
        
        var correlationLayout = {{
            title: 'Performance Correlation',
            xaxis: {{title: 'Sparsity Ratio'}},
            yaxis: {{title: 'Loss'}}
        }};
        
        Plotly.newPlot('performance-correlation-plot', [correlationTrace], correlationLayout);
        """

def demo_visualization_tools():
    """Demo function showing how to use AdvancedVisualizationTools"""
    print("🎨 Advanced Visualization Tools Demo")
    print("=" * 40)
    
    # Create sample data
    analysis_history = []
    performance_history = []
    
    for step in range(0, 100, 10):
        # Create sample analysis
        analysis = {
            'step': step,
            'analysis': {
                'selection_patterns': {
                    'selection_frequency': torch.rand(64),
                    'sparsity_ratio': 0.5 + 0.1 * np.sin(step / 10),
                    'selection_diversity': 0.7 + 0.1 * np.cos(step / 10)
                },
                'indexer_behavior': {
                    'score_stats': {
                        'mean': 0.5 + 0.1 * np.sin(step / 10),
                        'std': 0.2 + 0.05 * np.cos(step / 10)
                    }
                },
                'efficiency_metrics': {
                    'cost_reduction': 0.3 + 0.1 * np.sin(step / 10),
                    'attention_efficiency': 0.6 + 0.1 * np.cos(step / 10)
                }
            }
        }
        analysis_history.append(analysis)
        
        # Create sample performance
        performance = {
            'step': step,
            'loss': 2.0 - 0.01 * step + 0.1 * np.random.randn()
        }
        performance_history.append(performance)
    
    # Create visualizer
    visualizer = AdvancedVisualizationTools()
    
    # Create visualizations
    print("Creating interactive visualizations...")
    
    # Token selection visualization
    token_selection_path = visualizer.create_interactive_token_selection(analysis_history)
    print(f"✅ Token selection visualization: {token_selection_path}")
    
    # Attention pattern evolution
    attention_evolution_path = visualizer.create_attention_pattern_evolution(analysis_history)
    print(f"✅ Attention evolution visualization: {attention_evolution_path}")
    
    # Indexer behavior analysis
    indexer_behavior_path = visualizer.create_indexer_behavior_analysis(analysis_history)
    print(f"✅ Indexer behavior visualization: {indexer_behavior_path}")
    
    # Performance correlation analysis
    performance_correlation_path = visualizer.create_performance_correlation_analysis(
        analysis_history, performance_history
    )
    print(f"✅ Performance correlation visualization: {performance_correlation_path}")
    
    # Comprehensive dashboard
    dashboard_path = visualizer.create_comprehensive_dashboard(
        analysis_history, performance_history
    )
    print(f"✅ Comprehensive dashboard: {dashboard_path}")
    
    print("\n🎯 Demo completed! Use AdvancedVisualizationTools to create interactive visualizations.")

if __name__ == "__main__":
    demo_visualization_tools()
