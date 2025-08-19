"""
Graph Visualizer Component for LightRAG Streamlit Frontend
Interactive knowledge graph visualization with node/edge exploration
"""

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import pandas as pd
import tempfile
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import colorsys
import math

# Import API client
from utils.api_client import get_api_client


class GraphVisualizer:
    """Interactive knowledge graph visualizer component"""
    
    def __init__(self):
        self.api_client = get_api_client()
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state for graph visualization"""
        
        if "graph_data" not in st.session_state:
            st.session_state.graph_data = None
        
        if "graph_config" not in st.session_state:
            st.session_state.graph_config = {
                "max_nodes": 100,
                "layout": "spring",
                "node_size_metric": "degree",
                "show_labels": True,
                "show_edge_labels": False,
                "color_by": "type",
                "physics_enabled": True,
                "node_size_range": [10, 50],
                "edge_width_range": [1, 5]
            }
        
        if "selected_node" not in st.session_state:
            st.session_state.selected_node = None
        
        if "graph_stats" not in st.session_state:
            st.session_state.graph_stats = {}
    
    def render(self):
        """Render the complete graph visualization interface"""
        
        st.title("🕸️ Knowledge Graph Visualizer")
        
        # Control panel
        self.render_control_panel()
        
        # Graph visualization area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self.render_graph_display()
        
        with col2:
            self.render_graph_info_panel()
    
    def render_control_panel(self):
        """Render graph configuration and control panel"""
        
        with st.expander("🎛️ Graph Controls", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔄 Refresh Graph", use_container_width=True):
                    self.load_graph_data()
                
                if st.button("📥 Export Graph", use_container_width=True):
                    self.export_graph_data()
            
            with col2:
                max_nodes = st.slider(
                    "Max Nodes",
                    min_value=10, max_value=500,
                    value=st.session_state.graph_config["max_nodes"],
                    step=10
                )
                st.session_state.graph_config["max_nodes"] = max_nodes
            
            with col3:
                layout = st.selectbox(
                    "Layout Algorithm",
                    ["spring", "circular", "kamada_kawai", "random", "shell"],
                    index=["spring", "circular", "kamada_kawai", "random", "shell"].index(
                        st.session_state.graph_config["layout"]
                    )
                )
                st.session_state.graph_config["layout"] = layout
            
            with col4:
                color_by = st.selectbox(
                    "Color Nodes By",
                    ["type", "degree", "betweenness", "pagerank", "community"],
                    index=["type", "degree", "betweenness", "pagerank", "community"].index(
                        st.session_state.graph_config["color_by"]
                    )
                )
                st.session_state.graph_config["color_by"] = color_by
        
        # Advanced options
        with st.expander("🔧 Advanced Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.graph_config["show_labels"] = st.checkbox(
                    "Show Node Labels",
                    value=st.session_state.graph_config["show_labels"]
                )
                
                st.session_state.graph_config["show_edge_labels"] = st.checkbox(
                    "Show Edge Labels",
                    value=st.session_state.graph_config["show_edge_labels"]
                )
                
                st.session_state.graph_config["physics_enabled"] = st.checkbox(
                    "Enable Physics",
                    value=st.session_state.graph_config["physics_enabled"]
                )
            
            with col2:
                node_size_metric = st.selectbox(
                    "Node Size Based On",
                    ["degree", "betweenness", "pagerank", "uniform"],
                    index=["degree", "betweenness", "pagerank", "uniform"].index(
                        st.session_state.graph_config["node_size_metric"]
                    )
                )
                st.session_state.graph_config["node_size_metric"] = node_size_metric
                
                # Node size range
                size_range = st.slider(
                    "Node Size Range",
                    min_value=5, max_value=100,
                    value=(
                        st.session_state.graph_config["node_size_range"][0],
                        st.session_state.graph_config["node_size_range"][1]
                    ),
                    step=5
                )
                st.session_state.graph_config["node_size_range"] = list(size_range)
    
    def render_graph_display(self):
        """Render the main graph visualization"""
        
        if st.session_state.graph_data is None:
            st.info("📊 Loading graph data...")
            with st.spinner("Fetching knowledge graph..."):
                self.load_graph_data()
        
        if st.session_state.graph_data is None:
            st.error("❌ Failed to load graph data. Make sure documents are uploaded and processed.")
            return
        
        # Choose visualization method
        viz_method = st.radio(
            "Visualization Method:",
            ["Interactive Network (PyVis)", "Static Plot (Plotly)", "NetworkX Layout"],
            index=0,
            horizontal=True
        )
        
        if viz_method == "Interactive Network (PyVis)":
            self.render_pyvis_graph()
        elif viz_method == "Static Plot (Plotly)":
            self.render_plotly_graph()
        else:
            self.render_networkx_graph()
    
    def render_pyvis_graph(self):
        """Render interactive graph using PyVis"""
        
        try:
            # Create NetworkX graph from data
            G = self.create_networkx_graph()
            
            if len(G.nodes()) == 0:
                st.warning("No nodes found in the graph. Try uploading some documents first.")
                return
            
            # Create PyVis network
            net = Network(
                height="600px",
                width="100%",
                bgcolor="#ffffff",
                font_color="#000000"
            )
            
            # Configure physics
            if st.session_state.graph_config["physics_enabled"]:
                net.set_options("""
                var options = {
                  "physics": {
                    "enabled": true,
                    "stabilization": {"iterations": 100}
                  }
                }
                """)
            else:
                net.set_options('{"physics": {"enabled": false}}')
            
            # Add nodes and edges to PyVis network
            self.add_nodes_to_pyvis(net, G)
            self.add_edges_to_pyvis(net, G)
            
            # Generate HTML
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as tmp:
                net.save_graph(tmp.name)
                
                # Read and display the HTML
                with open(tmp.name, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Display in Streamlit
                st.components.v1.html(html_content, height=600)
                
                # Cleanup
                Path(tmp.name).unlink(missing_ok=True)
        
        except Exception as e:
            st.error(f"Error rendering PyVis graph: {str(e)}")
            st.info("Falling back to NetworkX visualization...")
            self.render_networkx_graph()
    
    def render_plotly_graph(self):
        """Render static graph using Plotly"""
        
        try:
            G = self.create_networkx_graph()
            
            if len(G.nodes()) == 0:
                st.warning("No nodes found in the graph.")
                return
            
            # Calculate layout
            layout_func = getattr(nx, f"{st.session_state.graph_config['layout']}_layout", nx.spring_layout)
            pos = layout_func(G)
            
            # Prepare node traces
            node_trace = self.create_plotly_node_trace(G, pos)
            edge_trace = self.create_plotly_edge_trace(G, pos)
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Knowledge Graph Visualization",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text=f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white',
                    height=600
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error rendering Plotly graph: {str(e)}")
    
    def render_networkx_graph(self):
        """Render simple NetworkX matplotlib graph"""
        
        try:
            import matplotlib.pyplot as plt
            
            G = self.create_networkx_graph()
            
            if len(G.nodes()) == 0:
                st.warning("No nodes found in the graph.")
                return
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Calculate layout
            layout_func = getattr(nx, f"{st.session_state.graph_config['layout']}_layout", nx.spring_layout)
            pos = layout_func(G)
            
            # Draw graph
            nx.draw(
                G, pos,
                with_labels=st.session_state.graph_config["show_labels"],
                node_color='lightblue',
                node_size=300,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                alpha=0.7,
                ax=ax
            )
            
            ax.set_title(f"Knowledge Graph ({len(G.nodes())} nodes, {len(G.edges())} edges)")
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close()
        
        except Exception as e:
            st.error(f"Error rendering NetworkX graph: {str(e)}")
    
    def render_graph_info_panel(self):
        """Render graph information and statistics panel"""
        
        st.subheader("📊 Graph Statistics")
        
        if st.session_state.graph_data is None:
            st.info("Load graph data to see statistics")
            return
        
        # Display basic stats
        if st.session_state.graph_stats:
            stats = st.session_state.graph_stats
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Nodes", stats.get("total_nodes", 0))
                st.metric("Edges", stats.get("total_edges", 0))
                st.metric("Density", f"{stats.get('density', 0):.4f}")
            
            with col2:
                st.metric("Avg Degree", f"{stats.get('avg_degree', 0):.2f}")
                st.metric("Connected Components", stats.get("connected_components", 0))
                st.metric("Clustering Coeff", f"{stats.get('clustering', 0):.4f}")
        
        # Node type breakdown
        if st.session_state.graph_data:
            self.render_node_type_breakdown()
        
        # Selected node details
        if st.session_state.selected_node:
            self.render_selected_node_details()
        
        # Export options
        st.markdown("---")
        st.subheader("📤 Export Options")
        
        export_format = st.selectbox(
            "Export Format",
            ["GraphML", "JSON", "CSV (Nodes)", "CSV (Edges)", "GEXF"]
        )
        
        if st.button("📥 Download", use_container_width=True):
            self.export_graph_data(export_format)
    
    def render_node_type_breakdown(self):
        """Render node type distribution"""
        
        st.subheader("🏷️ Node Types")
        
        try:
            G = self.create_networkx_graph()
            
            # Count node types
            node_types = {}
            for node, data in G.nodes(data=True):
                node_type = data.get('type', 'Unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            if node_types:
                # Create pie chart
                fig = px.pie(
                    values=list(node_types.values()),
                    names=list(node_types.keys()),
                    title="Node Type Distribution"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show as table
                type_df = pd.DataFrame([
                    {"Type": k, "Count": v, "Percentage": f"{v/sum(node_types.values())*100:.1f}%"}
                    for k, v in node_types.items()
                ])
                st.dataframe(type_df, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"Error rendering node types: {str(e)}")
    
    def render_selected_node_details(self):
        """Render details for selected node"""
        
        st.subheader("🎯 Selected Node")
        
        node_id = st.session_state.selected_node
        
        try:
            G = self.create_networkx_graph()
            
            if node_id in G.nodes():
                node_data = G.nodes[node_id]
                
                st.write(f"**ID:** {node_id}")
                
                for key, value in node_data.items():
                    if key != 'id':
                        st.write(f"**{key.title()}:** {value}")
                
                # Show neighbors
                neighbors = list(G.neighbors(node_id))
                if neighbors:
                    st.write(f"**Connected to {len(neighbors)} nodes:**")
                    for neighbor in neighbors[:10]:  # Show first 10
                        st.write(f"- {neighbor}")
                    if len(neighbors) > 10:
                        st.write(f"... and {len(neighbors) - 10} more")
        
        except Exception as e:
            st.error(f"Error showing node details: {str(e)}")
    
    def load_graph_data(self):
        """Load graph data from LightRAG API"""
        
        try:
            # Get graph data from API
            result = self.api_client.get_graph_data(
                max_nodes=st.session_state.graph_config["max_nodes"]
            )
            
            if result["success"]:
                st.session_state.graph_data = result["data"]
                
                # Calculate and store graph statistics
                G = self.create_networkx_graph()
                st.session_state.graph_stats = self.calculate_graph_statistics(G)
                
                st.success(f"✅ Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
            else:
                st.error(f"Failed to load graph: {result.get('error', 'Unknown error')}")
                st.session_state.graph_data = None
        
        except Exception as e:
            st.error(f"Error loading graph data: {str(e)}")
            st.session_state.graph_data = None
    
    def create_networkx_graph(self) -> nx.Graph:
        """Create NetworkX graph from loaded data"""
        
        G = nx.Graph()
        
        if not st.session_state.graph_data:
            return G
        
        graph_data = st.session_state.graph_data
        
        # Add nodes
        if "nodes" in graph_data:
            for node in graph_data["nodes"]:
                G.add_node(
                    node.get("id", str(node)),
                    **{k: v for k, v in node.items() if k != "id"}
                )
        
        # Add edges
        if "edges" in graph_data:
            for edge in graph_data["edges"]:
                source = edge.get("source", edge.get("from"))
                target = edge.get("target", edge.get("to"))
                
                if source and target:
                    G.add_edge(
                        source, target,
                        **{k: v for k, v in edge.items() if k not in ["source", "target", "from", "to"]}
                    )
        
        return G
    
    def calculate_graph_statistics(self, G: nx.Graph) -> Dict[str, Any]:
        """Calculate comprehensive graph statistics"""
        
        if len(G.nodes()) == 0:
            return {}
        
        stats = {
            "total_nodes": len(G.nodes()),
            "total_edges": len(G.edges()),
            "density": nx.density(G),
            "connected_components": nx.number_connected_components(G),
        }
        
        # Only calculate these for non-empty graphs
        if len(G.nodes()) > 0:
            degrees = [d for n, d in G.degree()]
            stats["avg_degree"] = sum(degrees) / len(degrees) if degrees else 0
            
            # Clustering coefficient (can be slow for large graphs)
            if len(G.nodes()) < 1000:
                stats["clustering"] = nx.average_clustering(G)
            else:
                stats["clustering"] = 0  # Skip for large graphs
        
        return stats
    
    def add_nodes_to_pyvis(self, net: Network, G: nx.Graph):
        """Add nodes to PyVis network with styling"""
        
        # Calculate metrics for styling
        metrics = {}
        if st.session_state.graph_config["color_by"] == "degree":
            metrics = dict(G.degree())
        elif st.session_state.graph_config["color_by"] == "betweenness":
            metrics = nx.betweenness_centrality(G)
        elif st.session_state.graph_config["color_by"] == "pagerank":
            metrics = nx.pagerank(G)
        
        # Node sizes
        size_metrics = {}
        if st.session_state.graph_config["node_size_metric"] == "degree":
            size_metrics = dict(G.degree())
        elif st.session_state.graph_config["node_size_metric"] == "betweenness":
            size_metrics = nx.betweenness_centrality(G)
        elif st.session_state.graph_config["node_size_metric"] == "pagerank":
            size_metrics = nx.pagerank(G)
        
        # Normalize values for coloring and sizing
        if metrics:
            max_metric = max(metrics.values()) if metrics.values() else 1
            min_metric = min(metrics.values()) if metrics.values() else 0
        
        if size_metrics:
            max_size = max(size_metrics.values()) if size_metrics.values() else 1
            min_size = min(size_metrics.values()) if size_metrics.values() else 0
        
        for node, data in G.nodes(data=True):
            # Node color
            if st.session_state.graph_config["color_by"] == "type":
                color = self.get_type_color(data.get("type", "Unknown"))
            elif metrics:
                # Color based on metric value
                normalized = (metrics.get(node, 0) - min_metric) / (max_metric - min_metric) if max_metric > min_metric else 0
                color = self.get_gradient_color(normalized)
            else:
                color = "#1f77b4"  # Default blue
            
            # Node size
            if size_metrics and st.session_state.graph_config["node_size_metric"] != "uniform":
                normalized_size = (size_metrics.get(node, 0) - min_size) / (max_size - min_size) if max_size > min_size else 0.5
                size = st.session_state.graph_config["node_size_range"][0] + normalized_size * (
                    st.session_state.graph_config["node_size_range"][1] - st.session_state.graph_config["node_size_range"][0]
                )
            else:
                size = sum(st.session_state.graph_config["node_size_range"]) / 2
            
            # Node label
            label = str(node)[:30] if st.session_state.graph_config["show_labels"] else ""
            
            # Hover info
            title = f"Node: {node}\n"
            for key, value in data.items():
                title += f"{key}: {value}\n"
            
            net.add_node(
                node,
                label=label,
                color=color,
                size=size,
                title=title
            )
    
    def add_edges_to_pyvis(self, net: Network, G: nx.Graph):
        """Add edges to PyVis network"""
        
        for source, target, data in G.edges(data=True):
            # Edge label
            label = data.get("type", "") if st.session_state.graph_config["show_edge_labels"] else ""
            
            # Edge width based on weight or default
            width = data.get("weight", 1) * 2
            
            # Hover info
            title = f"Edge: {source} → {target}\n"
            for key, value in data.items():
                title += f"{key}: {value}\n"
            
            net.add_edge(
                source, target,
                label=label,
                width=width,
                title=title
            )
    
    def create_plotly_node_trace(self, G: nx.Graph, pos: Dict) -> go.Scatter:
        """Create Plotly node trace"""
        
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info for hover
            adjacencies = list(G.neighbors(node))
            node_info.append(f"Node: {node}<br>Connections: {len(adjacencies)}")
            node_text.append(str(node) if st.session_state.graph_config["show_labels"] else "")
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            hovertext=node_info,
            textposition="middle center",
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='rgb(50,50,50)')
            )
        )
    
    def create_plotly_edge_trace(self, G: nx.Graph, pos: Dict) -> go.Scatter:
        """Create Plotly edge trace"""
        
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
    
    def get_type_color(self, node_type: str) -> str:
        """Get color for node type"""
        
        type_colors = {
            "entity": "#1f77b4",
            "person": "#ff7f0e", 
            "organization": "#2ca02c",
            "location": "#d62728",
            "event": "#9467bd",
            "concept": "#8c564b",
            "unknown": "#e377c2"
        }
        
        return type_colors.get(node_type.lower(), "#17becf")
    
    def get_gradient_color(self, value: float) -> str:
        """Get color from gradient based on value (0-1)"""
        
        # HSV color gradient from blue to red
        hue = (1 - value) * 240 / 360  # Blue to red
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        
        return f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"
    
    def export_graph_data(self, format_type: str = "JSON"):
        """Export graph data in specified format"""
        
        if not st.session_state.graph_data:
            st.error("No graph data to export")
            return
        
        try:
            G = self.create_networkx_graph()
            
            if format_type == "GraphML":
                # Export as GraphML
                import io
                output = io.StringIO()
                nx.write_graphml(G, output)
                data = output.getvalue()
                
                st.download_button(
                    label="📥 Download GraphML",
                    data=data,
                    file_name="knowledge_graph.graphml",
                    mime="application/xml"
                )
            
            elif format_type == "JSON":
                # Export as JSON
                data = json.dumps(st.session_state.graph_data, indent=2)
                
                st.download_button(
                    label="📥 Download JSON",
                    data=data,
                    file_name="knowledge_graph.json",
                    mime="application/json"
                )
            
            elif format_type == "CSV (Nodes)":
                # Export nodes as CSV
                nodes_data = []
                for node, data in G.nodes(data=True):
                    row = {"id": node}
                    row.update(data)
                    nodes_data.append(row)
                
                df = pd.DataFrame(nodes_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Nodes CSV",
                    data=csv,
                    file_name="graph_nodes.csv",
                    mime="text/csv"
                )
            
            elif format_type == "CSV (Edges)":
                # Export edges as CSV
                edges_data = []
                for source, target, data in G.edges(data=True):
                    row = {"source": source, "target": target}
                    row.update(data)
                    edges_data.append(row)
                
                df = pd.DataFrame(edges_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Edges CSV",
                    data=csv,
                    file_name="graph_edges.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Export failed: {str(e)}") 