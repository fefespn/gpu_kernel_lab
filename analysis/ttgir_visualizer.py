"""
TTGIR Dependency Graph Visualizer

Generates interactive HTML visualizations of TTGIR dependency graphs.
Uses vis.js for graph rendering with zoom, pan, and tooltips.
"""

import os
import json
from typing import Dict, List, Optional
from .ttgir_parser import DependencyGraph, OpCategory, parse_ttgir, parse_ttgir_file


# Color scheme for operation categories (matching Twill paper style)
CATEGORY_COLORS = {
    OpCategory.MEMORY: {
        'background': '#3498db',  # Blue
        'border': '#2980b9',
        'highlight': {'background': '#5dade2', 'border': '#2980b9'}
    },
    OpCategory.COMPUTE: {
        'background': '#e67e22',  # Orange
        'border': '#d35400',
        'highlight': {'background': '#f39c12', 'border': '#d35400'}
    },
    OpCategory.CONTROL: {
        'background': '#27ae60',  # Green
        'border': '#1e8449',
        'highlight': {'background': '#2ecc71', 'border': '#1e8449'}
    },
    OpCategory.CONSTANT: {
        'background': '#95a5a6',  # Gray
        'border': '#7f8c8d',
        'highlight': {'background': '#bdc3c7', 'border': '#7f8c8d'}
    },
    OpCategory.OTHER: {
        'background': '#9b59b6',  # Purple
        'border': '#8e44ad',
        'highlight': {'background': '#a569bd', 'border': '#8e44ad'}
    }
}

# Shape for operation categories
CATEGORY_SHAPES = {
    OpCategory.MEMORY: 'box',
    OpCategory.COMPUTE: 'ellipse',
    OpCategory.CONTROL: 'diamond',
    OpCategory.CONSTANT: 'dot',
    OpCategory.OTHER: 'triangle'
}


def graph_to_visjs(graph: DependencyGraph) -> Dict:
    """
    Convert dependency graph to vis.js format.
    
    Returns dict with 'nodes' and 'edges' arrays.
    """
    nodes = []
    edges = []
    
    for name, op in graph.operations.items():
        colors = CATEGORY_COLORS[op.category]
        shape = CATEGORY_SHAPES[op.category]
        
        # Create node label (short operation type)
        label = op.op_type.split('.')[-1]
        if len(label) > 12:
            label = label[:10] + '..'
        
        node = {
            'id': name,
            'label': label,
            'title': f"<b>{op.short_name}</b><br>{op.op_type}<br>Line: {op.line_num}<br>Type: {op.result_type[:50] if op.result_type else 'void'}",
            'shape': shape,
            'color': colors,
            'font': {'color': 'white', 'size': 11},
            'category': op.category.value,
            'opType': op.op_type,
            'lineNum': op.line_num
        }
        nodes.append(node)
    
    for i, edge in enumerate(graph.edges):
        e = {
            'id': f'edge_{i}',
            'from': edge.source,
            'to': edge.target,
            'arrows': 'to',
            'color': {'color': '#888', 'highlight': '#333'},
            'smooth': {'type': 'cubicBezier', 'forceDirection': 'vertical'}
        }
        if edge.is_loop_carried:
            e['dashes'] = True
            e['color'] = {'color': '#e74c3c', 'highlight': '#c0392b'}
        edges.append(e)
    
    return {'nodes': nodes, 'edges': edges}


HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <title>TTGIR Dependency Graph</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; 
            color: #eee;
        }
        #container { display: flex; height: 100vh; }
        #graph { flex: 1; background: #16213e; }
        #sidebar {
            width: 300px;
            padding: 20px;
            background: #0f3460;
            overflow-y: auto;
        }
        h1 { font-size: 18px; margin-bottom: 15px; color: #e94560; }
        h2 { font-size: 14px; margin: 15px 0 10px; color: #e94560; }
        .stat { 
            display: flex; 
            justify-content: space-between; 
            padding: 8px; 
            background: #1a1a2e; 
            border-radius: 4px; 
            margin-bottom: 5px;
            font-size: 13px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 13px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 10px;
        }
        .filter-btn {
            padding: 6px 12px;
            margin: 3px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: opacity 0.2s;
        }
        .filter-btn.active { opacity: 1; }
        .filter-btn.inactive { opacity: 0.4; }
        #controls { margin-top: 15px; }
        #controls button {
            padding: 8px 16px;
            margin: 3px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            background: #e94560;
            color: white;
            font-size: 12px;
        }
        #controls button:hover { background: #ff6b6b; }
        #selected-info {
            margin-top: 15px;
            padding: 10px;
            background: #1a1a2e;
            border-radius: 4px;
            font-size: 12px;
            display: none;
        }
        #selected-info.visible { display: block; }
        #selected-info h3 { color: #e94560; margin-bottom: 8px; }
        #selected-info p { margin: 4px 0; }
    </style>
</head>
<body>
    <div id="container">
        <div id="graph"></div>
        <div id="sidebar">
            <h1>🔬 TTGIR Dependency Graph</h1>
            
            <h2>📊 Statistics</h2>
            <div class="stat"><span>Operations:</span><span id="stat-ops">0</span></div>
            <div class="stat"><span>Dependencies:</span><span id="stat-edges">0</span></div>
            <div class="stat"><span>Memory Ops:</span><span id="stat-memory">0</span></div>
            <div class="stat"><span>Compute Ops:</span><span id="stat-compute">0</span></div>
            <div class="stat"><span>Control Ops:</span><span id="stat-control">0</span></div>
            
            <h2>🎨 Legend</h2>
            <div class="legend-item">
                <div class="legend-color" style="background: #3498db;"></div>
                <span>Memory (load, store, async)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #e67e22; border-radius: 50%;"></div>
                <span>Compute (dot, arith)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #27ae60;"></div>
                <span>Control (for, yield)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #95a5a6; border-radius: 50%;"></div>
                <span>Constants</span>
            </div>
            
            <h2>🔍 Filter</h2>
            <div id="filters"></div>
            
            <div id="controls">
                <button onclick="fitGraph()">Fit to View</button>
                <button onclick="resetFilters()">Reset Filters</button>
                <button onclick="togglePhysics()">Toggle Physics</button>
            </div>
            
            <div id="selected-info">
                <h3>Selected Node</h3>
                <p><strong>Name:</strong> <span id="sel-name"></span></p>
                <p><strong>Type:</strong> <span id="sel-type"></span></p>
                <p><strong>Line:</strong> <span id="sel-line"></span></p>
                <p><strong>Inputs:</strong> <span id="sel-inputs"></span></p>
                <p><strong>Outputs:</strong> <span id="sel-outputs"></span></p>
            </div>
        </div>
    </div>
    
    <script>
        const graphData = GRAPH_DATA_PLACEHOLDER;
        
        // Create vis.js network
        const container = document.getElementById('graph');
        const data = {
            nodes: new vis.DataSet(graphData.nodes),
            edges: new vis.DataSet(graphData.edges)
        };
        
        const options = {
            layout: {
                hierarchical: {
                    enabled: true,
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 80,
                    nodeSpacing: 120,
                    treeSpacing: 200
                }
            },
            physics: {
                enabled: false
            },
            interaction: {
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true
            },
            nodes: {
                borderWidth: 2,
                shadow: true
            },
            edges: {
                width: 1.5,
                shadow: true
            }
        };
        
        const network = new vis.Network(container, data, options);
        
        // Update statistics
        document.getElementById('stat-ops').textContent = graphData.nodes.length;
        document.getElementById('stat-edges').textContent = graphData.edges.length;
        
        const countByCategory = (cat) => graphData.nodes.filter(n => n.category === cat).length;
        document.getElementById('stat-memory').textContent = countByCategory('memory');
        document.getElementById('stat-compute').textContent = countByCategory('compute');
        document.getElementById('stat-control').textContent = countByCategory('control');
        
        // Create filter buttons
        const categories = ['memory', 'compute', 'control', 'constant', 'other'];
        const categoryColors = {
            memory: '#3498db',
            compute: '#e67e22', 
            control: '#27ae60',
            constant: '#95a5a6',
            other: '#9b59b6'
        };
        const activeFilters = new Set(categories);
        
        const filtersDiv = document.getElementById('filters');
        categories.forEach(cat => {
            const btn = document.createElement('button');
            btn.className = 'filter-btn active';
            btn.style.background = categoryColors[cat];
            btn.textContent = cat.charAt(0).toUpperCase() + cat.slice(1);
            btn.onclick = () => toggleFilter(cat, btn);
            filtersDiv.appendChild(btn);
        });
        
        function toggleFilter(category, btn) {
            if (activeFilters.has(category)) {
                activeFilters.delete(category);
                btn.classList.remove('active');
                btn.classList.add('inactive');
            } else {
                activeFilters.add(category);
                btn.classList.remove('inactive');
                btn.classList.add('active');
            }
            applyFilters();
        }
        
        function applyFilters() {
            const visibleNodes = graphData.nodes.filter(n => activeFilters.has(n.category));
            const visibleIds = new Set(visibleNodes.map(n => n.id));
            const visibleEdges = graphData.edges.filter(e => 
                visibleIds.has(e.from) && visibleIds.has(e.to)
            );
            
            data.nodes.clear();
            data.nodes.add(visibleNodes);
            data.edges.clear();
            data.edges.add(visibleEdges);
        }
        
        function resetFilters() {
            categories.forEach(cat => activeFilters.add(cat));
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.classList.remove('inactive');
                btn.classList.add('active');
            });
            data.nodes.clear();
            data.nodes.add(graphData.nodes);
            data.edges.clear();
            data.edges.add(graphData.edges);
        }
        
        function fitGraph() {
            network.fit({ animation: true });
        }
        
        let physicsEnabled = false;
        function togglePhysics() {
            physicsEnabled = !physicsEnabled;
            network.setOptions({ physics: { enabled: physicsEnabled } });
        }
        
        // Node selection
        network.on('click', function(params) {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const node = graphData.nodes.find(n => n.id === nodeId);
                
                document.getElementById('selected-info').classList.add('visible');
                document.getElementById('sel-name').textContent = node.id;
                document.getElementById('sel-type').textContent = node.opType;
                document.getElementById('sel-line').textContent = node.lineNum;
                
                const inputs = graphData.edges.filter(e => e.to === nodeId).map(e => e.from);
                const outputs = graphData.edges.filter(e => e.from === nodeId).map(e => e.to);
                
                document.getElementById('sel-inputs').textContent = inputs.length > 0 ? inputs.join(', ') : 'none';
                document.getElementById('sel-outputs').textContent = outputs.length > 0 ? outputs.join(', ') : 'none';
            } else {
                document.getElementById('selected-info').classList.remove('visible');
            }
        });
        
        // Initial fit
        network.once('afterDrawing', fitGraph);
    </script>
</body>
</html>
'''


class TTGIRVisualizer:
    """Generates interactive HTML visualizations of TTGIR dependency graphs."""
    
    def __init__(self):
        pass
    
    def visualize(self, graph: DependencyGraph, output_path: str) -> str:
        """
        Generate HTML visualization of the dependency graph.
        
        Args:
            graph: Parsed dependency graph
            output_path: Directory or file path for output
            
        Returns:
            Path to generated HTML file
        """
        # Convert graph to vis.js format
        visjs_data = graph_to_visjs(graph)
        
        # Generate HTML
        html = HTML_TEMPLATE.replace(
            'GRAPH_DATA_PLACEHOLDER',
            json.dumps(visjs_data, indent=2)
        )
        
        # Determine output file path
        if os.path.isdir(output_path):
            html_path = os.path.join(output_path, 'dependency_graph.html')
        else:
            html_path = output_path
        
        # Write HTML file
        os.makedirs(os.path.dirname(html_path) or '.', exist_ok=True)
        with open(html_path, 'w') as f:
            f.write(html)
        
        return html_path


def visualize_ttgir(ttgir_text: str, output_path: str) -> str:
    """
    Convenience function to parse and visualize TTGIR.
    
    Args:
        ttgir_text: Raw TTGIR text
        output_path: Output directory or file path
        
    Returns:
        Path to generated HTML file
    """
    graph = parse_ttgir(ttgir_text)
    visualizer = TTGIRVisualizer()
    return visualizer.visualize(graph, output_path)


def visualize_ttgir_file(filepath: str, output_path: Optional[str] = None) -> str:
    """
    Parse and visualize TTGIR from a file.
    
    Args:
        filepath: Path to TTGIR file
        output_path: Output directory (default: same as input)
        
    Returns:
        Path to generated HTML file
    """
    with open(filepath, 'r') as f:
        ttgir_text = f.read()
    
    if output_path is None:
        output_path = os.path.dirname(filepath) or '.'
    
    return visualize_ttgir(ttgir_text, output_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ttgir_visualizer.py <ttgir_file> [output_dir]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    html_path = visualize_ttgir_file(filepath, output_path)
    print(f"Visualization saved to: {html_path}")
