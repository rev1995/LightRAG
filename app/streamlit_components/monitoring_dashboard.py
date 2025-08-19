"""
Monitoring Dashboard Component for LightRAG Streamlit Frontend
Analytics dashboard with token usage, performance metrics, and cost tracking
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Import API client
from utils.api_client import get_api_client


class MonitoringDashboard:
    """Analytics and monitoring dashboard component"""
    
    def __init__(self):
        self.api_client = get_api_client()
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state for monitoring"""
        
        if "dashboard_refresh_interval" not in st.session_state:
            st.session_state.dashboard_refresh_interval = 30  # seconds
        
        if "dashboard_data_cache" not in st.session_state:
            st.session_state.dashboard_data_cache = {
                "last_update": None,
                "token_usage": {},
                "performance_metrics": {},
                "query_history": []
            }
    
    def render(self):
        """Render the complete monitoring dashboard"""
        
        st.title("📊 LightRAG Analytics Dashboard")
        
        # Dashboard controls
        self.render_dashboard_controls()
        
        # Key metrics overview
        self.render_key_metrics()
        
        # Usage analytics
        self.render_usage_analytics()
        
        # Performance metrics
        self.render_performance_metrics()
        
        # Cost analysis
        self.render_cost_analysis()
        
        # Query history
        self.render_query_history()
    
    def render_dashboard_controls(self):
        """Render dashboard control panel"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                self.refresh_data()
        
        with col2:
            auto_refresh = st.checkbox("Auto Refresh", value=False)
        
        with col3:
            if auto_refresh:
                refresh_interval = st.selectbox(
                    "Interval (s)",
                    [15, 30, 60, 120],
                    index=1
                )
                st.session_state.dashboard_refresh_interval = refresh_interval
        
        with col4:
            export_format = st.selectbox(
                "Export",
                ["Select...", "CSV", "JSON"],
                index=0
            )
            if export_format != "Select...":
                self.export_data(export_format)
        
        # Auto refresh implementation
        if auto_refresh:
            import time
            if (st.session_state.dashboard_data_cache["last_update"] is None or 
                time.time() - st.session_state.dashboard_data_cache["last_update"] > refresh_interval):
                self.refresh_data()
        
        st.markdown("---")
    
    def render_key_metrics(self):
        """Render key performance metrics"""
        
        st.subheader("📈 Key Metrics")
        
        # Get session stats from chat messages
        chat_messages = st.session_state.get("chat_messages", [])
        
        # Calculate metrics
        total_queries = len([msg for msg in chat_messages if msg["role"] == "user"])
        
        total_tokens = sum(
            msg.get("metadata", {}).get("token_usage", {}).get("total_tokens", 0)
            for msg in chat_messages
            if msg["role"] == "assistant"
        )
        
        total_cost = sum(
            msg.get("metadata", {}).get("cost_estimate", 0)
            for msg in chat_messages
            if msg["role"] == "assistant"
        )
        
        avg_response_time = 0
        response_times = [
            msg.get("metadata", {}).get("response_time", 0)
            for msg in chat_messages
            if msg["role"] == "assistant" and msg.get("metadata", {}).get("response_time")
        ]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
        
        success_rate = 0
        assistant_messages = [msg for msg in chat_messages if msg["role"] == "assistant"]
        if assistant_messages:
            successful = len([msg for msg in assistant_messages if not msg.get("metadata", {}).get("error")])
            success_rate = (successful / len(assistant_messages)) * 100
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Queries",
                total_queries,
                delta=self.get_delta("queries", total_queries)
            )
        
        with col2:
            st.metric(
                "Total Tokens",
                f"{total_tokens:,}",
                delta=self.get_delta("tokens", total_tokens)
            )
        
        with col3:
            st.metric(
                "Avg Response Time",
                f"{avg_response_time:.2f}s",
                delta=f"{self.get_delta('response_time', avg_response_time):.2f}s"
            )
        
        with col4:
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%",
                delta=f"{self.get_delta('success_rate', success_rate):.1f}%"
            )
        
        with col5:
            st.metric(
                "Total Cost",
                f"${total_cost:.4f}",
                delta=f"${self.get_delta('cost', total_cost):.4f}"
            )
    
    def render_usage_analytics(self):
        """Render usage analytics charts"""
        
        st.subheader("📊 Usage Analytics")
        
        chat_messages = st.session_state.get("chat_messages", [])
        
        if not chat_messages:
            st.info("No data available. Start chatting to see analytics!")
            return
        
        # Prepare data
        usage_data = []
        for msg in chat_messages:
            if msg["role"] == "assistant" and msg.get("metadata"):
                metadata = msg["metadata"]
                usage_data.append({
                    "timestamp": msg.get("timestamp", datetime.now()),
                    "token_count": metadata.get("token_usage", {}).get("total_tokens", 0),
                    "prompt_tokens": metadata.get("token_usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": metadata.get("token_usage", {}).get("completion_tokens", 0),
                    "response_time": metadata.get("response_time", 0),
                    "query_mode": metadata.get("query_mode", "unknown"),
                    "cost": metadata.get("cost_estimate", 0),
                    "success": not metadata.get("error")
                })
        
        if not usage_data:
            st.info("No query data available yet.")
            return
        
        df = pd.DataFrame(usage_data)
        
        # Token usage over time
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Token Usage Over Time**")
            fig = px.line(
                df,
                x="timestamp",
                y="token_count",
                title="Token Consumption",
                labels={"token_count": "Tokens", "timestamp": "Time"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Token Breakdown**")
            token_breakdown = df[["prompt_tokens", "completion_tokens"]].sum()
            fig = px.pie(
                values=token_breakdown.values,
                names=token_breakdown.index,
                title="Prompt vs Completion Tokens"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Query mode analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Usage by Query Mode**")
            mode_usage = df.groupby("query_mode")["token_count"].sum().reset_index()
            fig = px.bar(
                mode_usage,
                x="query_mode",
                y="token_count",
                title="Token Usage by Query Mode"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Response Time Distribution**")
            fig = px.histogram(
                df,
                x="response_time",
                title="Response Time Distribution",
                labels={"response_time": "Response Time (s)"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_metrics(self):
        """Render performance metrics"""
        
        st.subheader("⚡ Performance Metrics")
        
        chat_messages = st.session_state.get("chat_messages", [])
        
        # Prepare performance data
        perf_data = []
        for msg in chat_messages:
            if msg["role"] == "assistant" and msg.get("metadata"):
                metadata = msg["metadata"]
                perf_data.append({
                    "response_time": metadata.get("response_time", 0),
                    "query_mode": metadata.get("query_mode", "unknown"),
                    "token_count": metadata.get("token_usage", {}).get("total_tokens", 0),
                    "context_chunks": metadata.get("context_chunks", 0),
                    "entities_used": metadata.get("entities_used", 0),
                    "relationships_used": metadata.get("relationships_used", 0),
                    "success": not metadata.get("error"),
                    "timestamp": msg.get("timestamp", datetime.now())
                })
        
        if not perf_data:
            st.info("No performance data available yet.")
            return
        
        df = pd.DataFrame(perf_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Response Time by Query Mode**")
            fig = px.box(
                df,
                x="query_mode",
                y="response_time",
                title="Response Time Distribution by Mode"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Token Count vs Response Time**")
            fig = px.scatter(
                df,
                x="token_count",
                y="response_time",
                color="query_mode",
                title="Token Count vs Response Time",
                labels={"token_count": "Total Tokens", "response_time": "Response Time (s)"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance summary table
        st.markdown("**Performance Summary by Query Mode**")
        
        if len(df) > 0:
            summary = df.groupby("query_mode").agg({
                "response_time": ["mean", "median", "std"],
                "token_count": ["mean", "median"],
                "success": "mean"
            }).round(3)
            
            # Flatten column names
            summary.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in summary.columns]
            summary = summary.reset_index()
            
            st.dataframe(summary, use_container_width=True)
    
    def render_cost_analysis(self):
        """Render cost analysis"""
        
        st.subheader("💰 Cost Analysis")
        
        chat_messages = st.session_state.get("chat_messages", [])
        
        # Calculate cost data
        cost_data = []
        cumulative_cost = 0
        
        for msg in chat_messages:
            if msg["role"] == "assistant" and msg.get("metadata"):
                metadata = msg["metadata"]
                cost = metadata.get("cost_estimate", 0)
                cumulative_cost += cost
                
                cost_data.append({
                    "timestamp": msg.get("timestamp", datetime.now()),
                    "cost": cost,
                    "cumulative_cost": cumulative_cost,
                    "query_mode": metadata.get("query_mode", "unknown"),
                    "token_count": metadata.get("token_usage", {}).get("total_tokens", 0)
                })
        
        if not cost_data:
            st.info("No cost data available yet.")
            return
        
        df = pd.DataFrame(cost_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Cumulative Cost Over Time**")
            fig = px.line(
                df,
                x="timestamp",
                y="cumulative_cost",
                title="Cumulative Cost",
                labels={"cumulative_cost": "Cost ($)", "timestamp": "Time"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Cost per Query**")
            fig = px.bar(
                df,
                x=range(len(df)),
                y="cost",
                color="query_mode",
                title="Cost per Query",
                labels={"x": "Query Number", "cost": "Cost ($)"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cost breakdown
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_cost = df["cost"].sum()
            st.metric("Total Session Cost", f"${total_cost:.4f}")
        
        with col2:
            avg_cost_per_query = df["cost"].mean() if len(df) > 0 else 0
            st.metric("Avg Cost per Query", f"${avg_cost_per_query:.4f}")
        
        with col3:
            cost_per_1k_tokens = (df["cost"].sum() / (df["token_count"].sum() / 1000)) if df["token_count"].sum() > 0 else 0
            st.metric("Cost per 1K Tokens", f"${cost_per_1k_tokens:.4f}")
    
    def render_query_history(self):
        """Render query history table"""
        
        st.subheader("📝 Query History")
        
        chat_messages = st.session_state.get("chat_messages", [])
        
        # Prepare query history data
        history_data = []
        
        user_queries = [msg for msg in chat_messages if msg["role"] == "user"]
        assistant_responses = [msg for msg in chat_messages if msg["role"] == "assistant"]
        
        for i, user_msg in enumerate(user_queries):
            # Find corresponding assistant response
            assistant_msg = assistant_responses[i] if i < len(assistant_responses) else None
            
            row = {
                "Query": user_msg["content"][:100] + "..." if len(user_msg["content"]) > 100 else user_msg["content"],
                "Timestamp": user_msg.get("timestamp", datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
                "Mode": user_msg.get("metadata", {}).get("config", {}).get("query_mode", "unknown"),
                "Success": "✅" if assistant_msg and not assistant_msg.get("metadata", {}).get("error") else "❌",
                "Response Time": f"{assistant_msg.get('metadata', {}).get('response_time', 0):.2f}s" if assistant_msg else "N/A",
                "Tokens": f"{assistant_msg.get('metadata', {}).get('token_usage', {}).get('total_tokens', 0):,}" if assistant_msg else "0",
                "Cost": f"${assistant_msg.get('metadata', {}).get('cost_estimate', 0):.4f}" if assistant_msg else "$0.0000"
            }
            history_data.append(row)
        
        if history_data:
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No query history available.")
    
    def refresh_data(self):
        """Refresh dashboard data"""
        
        try:
            # Update cache timestamp
            st.session_state.dashboard_data_cache["last_update"] = datetime.now().timestamp()
            
            # Force rerun to refresh UI
            st.rerun()
            
        except Exception as e:
            st.error(f"Failed to refresh data: {str(e)}")
    
    def get_delta(self, metric_name: str, current_value: float) -> float:
        """Calculate delta for metrics (placeholder implementation)"""
        
        # In a real implementation, this would compare with historical data
        # For now, return 0 as we don't have historical storage
        return 0.0
    
    def export_data(self, format_type: str):
        """Export dashboard data"""
        
        chat_messages = st.session_state.get("chat_messages", [])
        
        # Prepare export data
        export_data = []
        for msg in chat_messages:
            if msg["role"] == "assistant" and msg.get("metadata"):
                export_data.append({
                    "timestamp": msg.get("timestamp", datetime.now()).isoformat(),
                    "metadata": msg.get("metadata", {})
                })
        
        if format_type == "CSV":
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"lightrag_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        elif format_type == "JSON":
            json_data = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"lightrag_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            ) 