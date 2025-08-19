"""
Enhanced Token Tracking System
Comprehensive token usage monitoring with database storage and analytics
"""

import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Import from local LightRAG
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LightRAG"))

from lightrag.utils import logger


@dataclass
class QueryMetrics:
    """Comprehensive query metrics data structure"""
    query_id: str
    timestamp: datetime
    query_text: str
    query_mode: str
    response_time: float
    success: bool
    
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # API calls
    llm_calls: int = 0
    embedding_calls: int = 0
    rerank_calls: int = 0
    
    # Cost estimation
    estimated_cost: float = 0.0
    
    # Performance metrics
    context_chunks: int = 0
    entities_used: int = 0
    relationships_used: int = 0
    
    # Error information
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    # Configuration
    top_k: int = 0
    chunk_top_k: int = 0
    max_total_tokens: int = 0
    enable_rerank: bool = False


@dataclass
class SessionMetrics:
    """Session-level aggregated metrics"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_queries: int = 0
    successful_queries: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0


class EnhancedTokenTracker:
    """Enhanced token tracking with persistent storage and analytics"""
    
    def __init__(self, db_path: str = "../storage/token_usage.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Current session
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now()
        
        # Initialize database
        self.init_database()
        
        logger.info(f"✅ Enhanced Token Tracker initialized: {self.db_path}")
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_metrics (
                query_id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp TEXT,
                query_text TEXT,
                query_mode TEXT,
                response_time REAL,
                success BOOLEAN,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                llm_calls INTEGER,
                embedding_calls INTEGER,
                rerank_calls INTEGER,
                estimated_cost REAL,
                context_chunks INTEGER,
                entities_used INTEGER,
                relationships_used INTEGER,
                error_message TEXT,
                error_type TEXT,
                top_k INTEGER,
                chunk_top_k INTEGER,
                max_total_tokens INTEGER,
                enable_rerank BOOLEAN
            )
        """)
        
        # Session metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_metrics (
                session_id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                total_queries INTEGER,
                successful_queries INTEGER,
                total_tokens INTEGER,
                total_cost REAL,
                avg_response_time REAL
            )
        """)
        
        # Cost tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                total_cost REAL,
                total_tokens INTEGER,
                query_count INTEGER
            )
        """)
        
        # Performance benchmarks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                query_mode TEXT,
                avg_response_time REAL,
                avg_tokens REAL,
                success_rate REAL,
                sample_size INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def track_query(self, metrics: QueryMetrics):
        """Track a single query with comprehensive metrics"""
        
        # Set session ID
        metrics.query_id = metrics.query_id or str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert query metrics
        cursor.execute("""
            INSERT OR REPLACE INTO query_metrics VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            metrics.query_id,
            self.session_id,
            metrics.timestamp.isoformat(),
            metrics.query_text,
            metrics.query_mode,
            metrics.response_time,
            metrics.success,
            metrics.prompt_tokens,
            metrics.completion_tokens,
            metrics.total_tokens,
            metrics.llm_calls,
            metrics.embedding_calls,
            metrics.rerank_calls,
            metrics.estimated_cost,
            metrics.context_chunks,
            metrics.entities_used,
            metrics.relationships_used,
            metrics.error_message,
            metrics.error_type,
            metrics.top_k,
            metrics.chunk_top_k,
            metrics.max_total_tokens,
            metrics.enable_rerank
        ))
        
        conn.commit()
        conn.close()
        
        # Update session metrics
        self.update_session_metrics()
        
        logger.debug(f"Tracked query {metrics.query_id}: {metrics.total_tokens} tokens, ${metrics.estimated_cost:.4f}")
    
    def update_session_metrics(self):
        """Update current session aggregated metrics"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate session aggregates
        cursor.execute("""
            SELECT 
                COUNT(*) as total_queries,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries,
                SUM(total_tokens) as total_tokens,
                SUM(estimated_cost) as total_cost,
                AVG(response_time) as avg_response_time
            FROM query_metrics 
            WHERE session_id = ?
        """, (self.session_id,))
        
        result = cursor.fetchone()
        
        if result:
            # Update session metrics
            cursor.execute("""
                INSERT OR REPLACE INTO session_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.session_id,
                self.session_start.isoformat(),
                None,  # end_time (ongoing session)
                result[0],  # total_queries
                result[1],  # successful_queries
                result[2] or 0,  # total_tokens
                result[3] or 0.0,  # total_cost
                result[4] or 0.0   # avg_response_time
            ))
        
        conn.commit()
        conn.close()
    
    def get_session_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a specific session"""
        
        target_session = session_id or self.session_id
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM session_metrics WHERE session_id = ?
        """, (target_session,))
        
        result = cursor.fetchone()
        
        if result:
            stats = {
                "session_id": result[0],
                "start_time": result[1],
                "end_time": result[2],
                "total_queries": result[3],
                "successful_queries": result[4],
                "total_tokens": result[5],
                "total_cost": result[6],
                "avg_response_time": result[7],
                "success_rate": (result[4] / result[3] * 100) if result[3] > 0 else 0
            }
        else:
            stats = {
                "session_id": target_session,
                "total_queries": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "success_rate": 0.0
            }
        
        conn.close()
        return stats
    
    def get_usage_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive usage analytics for the specified period"""
        
        start_date = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_queries,
                SUM(total_tokens) as total_tokens,
                SUM(estimated_cost) as total_cost,
                AVG(response_time) as avg_response_time,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries
            FROM query_metrics 
            WHERE timestamp >= ?
        """, (start_date.isoformat(),))
        
        overall = cursor.fetchone()
        
        # Query mode breakdown
        cursor.execute("""
            SELECT 
                query_mode,
                COUNT(*) as count,
                SUM(total_tokens) as tokens,
                AVG(response_time) as avg_time,
                SUM(estimated_cost) as cost
            FROM query_metrics 
            WHERE timestamp >= ?
            GROUP BY query_mode
            ORDER BY count DESC
        """, (start_date.isoformat(),))
        
        mode_breakdown = cursor.fetchall()
        
        # Daily usage
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as queries,
                SUM(total_tokens) as tokens,
                SUM(estimated_cost) as cost
            FROM query_metrics 
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (start_date.isoformat(),))
        
        daily_usage = cursor.fetchall()
        
        # Error analysis
        cursor.execute("""
            SELECT 
                error_type,
                COUNT(*) as count,
                error_message
            FROM query_metrics 
            WHERE timestamp >= ? AND success = 0
            GROUP BY error_type, error_message
            ORDER BY count DESC
        """, (start_date.isoformat(),))
        
        errors = cursor.fetchall()
        
        conn.close()
        
        return {
            "period_days": days,
            "overall": {
                "total_queries": overall[0] or 0,
                "total_tokens": overall[1] or 0,
                "total_cost": overall[2] or 0.0,
                "avg_response_time": overall[3] or 0.0,
                "success_rate": (overall[4] / overall[0] * 100) if overall[0] > 0 else 0
            },
            "by_mode": [
                {
                    "mode": row[0],
                    "queries": row[1],
                    "tokens": row[2],
                    "avg_response_time": row[3],
                    "cost": row[4]
                }
                for row in mode_breakdown
            ],
            "daily_usage": [
                {
                    "date": row[0],
                    "queries": row[1],
                    "tokens": row[2],
                    "cost": row[3]
                }
                for row in daily_usage
            ],
            "errors": [
                {
                    "type": row[0],
                    "count": row[1],
                    "message": row[2]
                }
                for row in errors
            ]
        }
    
    def get_cost_breakdown(self, days: int = 30) -> Dict[str, Any]:
        """Get detailed cost breakdown and projections"""
        
        start_date = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cost by component
        cursor.execute("""
            SELECT 
                SUM(estimated_cost) as total_cost,
                SUM(prompt_tokens) as prompt_tokens,
                SUM(completion_tokens) as completion_tokens,
                AVG(estimated_cost) as avg_cost_per_query
            FROM query_metrics 
            WHERE timestamp >= ? AND success = 1
        """, (start_date.isoformat(),))
        
        cost_summary = cursor.fetchone()
        
        # Daily cost trend
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                SUM(estimated_cost) as daily_cost,
                COUNT(*) as queries
            FROM query_metrics 
            WHERE timestamp >= ? AND success = 1
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (start_date.isoformat(),))
        
        daily_costs = cursor.fetchall()
        
        conn.close()
        
        # Calculate projections
        total_cost = cost_summary[0] or 0.0
        avg_daily_cost = total_cost / days if days > 0 else 0.0
        
        # Pricing breakdown (Gemini 2.0 Flash)
        prompt_tokens = cost_summary[1] or 0
        completion_tokens = cost_summary[2] or 0
        
        prompt_cost = (prompt_tokens / 1_000_000) * 0.075
        completion_cost = (completion_tokens / 1_000_000) * 0.30
        
        return {
            "period_days": days,
            "total_cost": total_cost,
            "avg_daily_cost": avg_daily_cost,
            "projected_monthly_cost": avg_daily_cost * 30,
            "projected_yearly_cost": avg_daily_cost * 365,
            "cost_breakdown": {
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            },
            "daily_costs": [
                {
                    "date": row[0],
                    "cost": row[1],
                    "queries": row[2],
                    "cost_per_query": row[1] / row[2] if row[2] > 0 else 0
                }
                for row in daily_costs
            ]
        }
    
    def get_performance_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get performance metrics and benchmarks"""
        
        start_date = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Response time statistics
        cursor.execute("""
            SELECT 
                query_mode,
                AVG(response_time) as avg_time,
                MIN(response_time) as min_time,
                MAX(response_time) as max_time,
                COUNT(*) as sample_size
            FROM query_metrics 
            WHERE timestamp >= ? AND success = 1
            GROUP BY query_mode
        """, (start_date.isoformat(),))
        
        response_times = cursor.fetchall()
        
        # Token efficiency
        cursor.execute("""
            SELECT 
                query_mode,
                AVG(total_tokens) as avg_tokens,
                AVG(total_tokens / response_time) as tokens_per_second
            FROM query_metrics 
            WHERE timestamp >= ? AND success = 1 AND response_time > 0
            GROUP BY query_mode
        """, (start_date.isoformat(),))
        
        token_efficiency = cursor.fetchall()
        
        # Success rates
        cursor.execute("""
            SELECT 
                query_mode,
                COUNT(*) as total_queries,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries
            FROM query_metrics 
            WHERE timestamp >= ?
            GROUP BY query_mode
        """, (start_date.isoformat(),))
        
        success_rates = cursor.fetchall()
        
        conn.close()
        
        return {
            "period_days": days,
            "response_times": [
                {
                    "mode": row[0],
                    "avg_time": row[1],
                    "min_time": row[2],
                    "max_time": row[3],
                    "sample_size": row[4]
                }
                for row in response_times
            ],
            "token_efficiency": [
                {
                    "mode": row[0],
                    "avg_tokens": row[1],
                    "tokens_per_second": row[2]
                }
                for row in token_efficiency
            ],
            "success_rates": [
                {
                    "mode": row[0],
                    "total_queries": row[1],
                    "successful_queries": row[2],
                    "success_rate": (row[2] / row[1] * 100) if row[1] > 0 else 0
                }
                for row in success_rates
            ]
        }
    
    def export_data(self, format_type: str = "json", days: int = 30) -> str:
        """Export tracking data in specified format"""
        
        analytics = self.get_usage_analytics(days)
        cost_data = self.get_cost_breakdown(days)
        performance_data = self.get_performance_metrics(days)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "period_days": days,
            "usage_analytics": analytics,
            "cost_breakdown": cost_data,
            "performance_metrics": performance_data
        }
        
        if format_type.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        elif format_type.lower() == "csv":
            # Simplified CSV export
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write headers and data
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Queries", analytics["overall"]["total_queries"]])
            writer.writerow(["Total Tokens", analytics["overall"]["total_tokens"]])
            writer.writerow(["Total Cost", f"${analytics['overall']['total_cost']:.4f}"])
            writer.writerow(["Success Rate", f"{analytics['overall']['success_rate']:.1f}%"])
            
            return output.getvalue()
        else:
            return str(export_data)
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old tracking data to manage database size"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete old query metrics
        cursor.execute("""
            DELETE FROM query_metrics WHERE timestamp < ?
        """, (cutoff_date.isoformat(),))
        
        # Delete old session metrics
        cursor.execute("""
            DELETE FROM session_metrics WHERE start_time < ?
        """, (cutoff_date.isoformat(),))
        
        deleted_queries = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {deleted_queries} old records older than {days_to_keep} days")
        
        return deleted_queries
    
    def close_session(self):
        """Close current session and finalize metrics"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update session end time
        cursor.execute("""
            UPDATE session_metrics 
            SET end_time = ? 
            WHERE session_id = ?
        """, (datetime.now().isoformat(), self.session_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Closed session {self.session_id}")


# Global tracker instance
_global_tracker: Optional[EnhancedTokenTracker] = None

def get_global_tracker() -> EnhancedTokenTracker:
    """Get or create global token tracker instance"""
    global _global_tracker
    
    if _global_tracker is None:
        _global_tracker = EnhancedTokenTracker()
    
    return _global_tracker

def track_query_metrics(
    query_text: str,
    query_mode: str,
    response_time: float,
    success: bool,
    token_usage: Dict[str, int],
    cost_estimate: float,
    **kwargs
) -> str:
    """Convenience function to track query metrics"""
    
    metrics = QueryMetrics(
        query_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        query_text=query_text,
        query_mode=query_mode,
        response_time=response_time,
        success=success,
        prompt_tokens=token_usage.get("prompt_tokens", 0),
        completion_tokens=token_usage.get("completion_tokens", 0),
        total_tokens=token_usage.get("total_tokens", 0),
        estimated_cost=cost_estimate,
        **kwargs
    )
    
    tracker = get_global_tracker()
    tracker.track_query(metrics)
    
    return metrics.query_id 