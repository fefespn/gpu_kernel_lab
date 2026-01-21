#!/usr/bin/env python3
"""
CLI tool for visualizing TTGIR dependency graphs.

Usage:
    python visualize_ttgir.py <ttgir_file> [output_dir]
    
Examples:
    python visualize_ttgir.py outputs/triton_matmul/matmul_sm100_ttgir.txt
    python visualize_ttgir.py outputs/triton_matmul/matmul_sm100_ttgir.txt ./graphs
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import parse_ttgir_file, visualize_ttgir_file, TTGIRParser


def main():
    parser = argparse.ArgumentParser(
        description='Visualize TTGIR dependency graphs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s outputs/triton_matmul/matmul_sm100_ttgir.txt
  %(prog)s outputs/triton_matmul/matmul_sm100_ttgir.txt --output ./graphs
  %(prog)s outputs/triton_matmul/matmul_sm100_ttgir.txt --stats-only
        """
    )
    parser.add_argument('ttgir_file', help='Path to TTGIR file')
    parser.add_argument('-o', '--output', help='Output directory (default: same as input)')
    parser.add_argument('--stats-only', action='store_true', 
                        help='Only print statistics, do not generate visualization')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open browser after generating')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ttgir_file):
        print(f"Error: File not found: {args.ttgir_file}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Parsing TTGIR: {args.ttgir_file}")
    
    # Parse the file
    graph = parse_ttgir_file(args.ttgir_file)
    
    # Get statistics
    ttgir_parser = TTGIRParser()
    ttgir_parser.graph = graph
    stats = ttgir_parser.get_statistics()
    
    print(f"\n📊 Dependency Graph Statistics:")
    print(f"   Operations: {stats['total_operations']}")
    print(f"   Edges:      {stats['total_edges']}")
    print(f"   Memory:     {stats['memory_ops']} ops (loads, stores, async)")
    print(f"   Compute:    {stats['compute_ops']} ops (dot, arithmetic)")
    print(f"   Control:    {stats['control_ops']} ops (for, yield)")
    print(f"   Constants:  {stats['constant_ops']}")
    
    if args.stats_only:
        return
    
    # Generate visualization
    html_path = visualize_ttgir_file(args.ttgir_file, args.output)
    print(f"\n✅ Visualization saved: {html_path}")
    
    # Open in browser unless disabled
    if not args.no_browser:
        import webbrowser
        abs_path = os.path.abspath(html_path)
        url = f'file://{abs_path}'
        print(f"🌐 Opening in browser: {url}")
        webbrowser.open(url)


if __name__ == '__main__':
    main()
