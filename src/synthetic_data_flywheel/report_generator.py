"""Report Generator - Jinja2-based HTML dashboard with per-cycle metrics visualization."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.models import CycleState


HTML_REPORT_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Synthetic Data Flywheel Report</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            color: white;
            margin-bottom: 2rem;
        }
        h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
        .subtitle { opacity: 0.9; font-size: 1.1rem; }
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background: #f5f5f5;
            font-weight: 600;
            color: #333;
        }
        tr:hover { background: #f9f9f9; }
        .pass-rate { font-weight: bold; }
        .pass-rate.high { color: #4caf50; }
        .pass-rate.medium { color: #ff9800; }
        .pass-rate.low { color: #f44336; }
        .chart-container {
            height: 300px;
            margin-top: 1rem;
        }
        h2 {
            color: #333;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
        }
        .timestamp {
            text-align: center;
            color: white;
            opacity: 0.8;
            margin-top: 2rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔄 Synthetic Data Flywheel</h1>
            <p class="subtitle">Training Data Generation Report</p>
        </header>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{{ total_cycles }}</div>
                <div class="metric-label">Total Cycles</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ total_passed }}</div>
                <div class="metric-label">Passed Pairs</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ avg_pass_rate }}%</div>
                <div class="metric-label">Avg Pass Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ avg_quality }}</div>
                <div class="metric-label">Avg Quality Score</div>
            </div>
        </div>
        
        <div class="card">
            <h2>📊 Cycle History</h2>
            <table>
                <thead>
                    <tr>
                        <th>Cycle</th>
                        <th>Pass Rate</th>
                        <th>Avg Quality</th>
                        <th>Duration (s)</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for cycle in cycles %}
                    <tr>
                        <td>#{{ cycle.cycle_id }}</td>
                        <td class="pass-rate {{ cycle.pass_rate_class }}">{{ cycle.pass_rate }}%</td>
                        <td>{{ cycle.avg_quality }}</td>
                        <td>{{ cycle.duration }}</td>
                        <td>{{ cycle.status }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>📈 Quality Metrics per Cycle</h2>
            <table>
                <thead>
                    <tr>
                        <th>Cycle</th>
                        <th>Coherence</th>
                        <th>Accuracy</th>
                        <th>Helpfulness</th>
                        <th>Overall</th>
                    </tr>
                </thead>
                <tbody>
                    {% for cycle in cycles %}
                    <tr>
                        <td>#{{ cycle.cycle_id }}</td>
                        <td>{{ cycle.coherence }}</td>
                        <td>{{ cycle.accuracy }}</td>
                        <td>{{ cycle.helpfulness }}</td>
                        <td><strong>{{ cycle.overall }}</strong></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>🎯 Summary Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Generated Pairs</td>
                    <td>{{ total_generated }}</td>
                </tr>
                <tr>
                    <td>Total Passed Pairs</td>
                    <td>{{ total_passed }}</td>
                </tr>
                <tr>
                    <td>Total Failed Pairs</td>
                    <td>{{ total_failed }}</td>
                </tr>
                <tr>
                    <td>Average Pass Rate</td>
                    <td>{{ avg_pass_rate }}%</td>
                </tr>
                <tr>
                    <td>Best Cycle</td>
                    <td>#{{ best_cycle }}</td>
                </tr>
            </table>
        </div>
    </div>
    
    <p class="timestamp">Generated on {{ generated_at }}</p>
</body>
</html>'''


class ReportGenerator:
    """Generator for HTML reports."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize report generator."""
        settings = get_settings()
        self.output_dir = Path(output_dir or settings.report_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        cycles: List[CycleState],
        filename: Optional[str] = None,
    ) -> Path:
        """Generate HTML report from cycle data."""
        if not cycles:
            raise ValueError("No cycles to report on")
        
        # Calculate summary statistics
        total_cycles = len(cycles)
        total_passed = sum(len(c.passed_pairs) for c in cycles)
        total_generated = sum(len(c.generated_pairs) for c in cycles)
        total_failed = total_generated - total_passed
        
        avg_pass_rate = sum(c.pass_rate for c in cycles) / len(cycles) * 100
        avg_quality = sum(c.avg_quality_score for c in cycles) / len(cycles)
        
        # Find best cycle
        best_cycle = max(cycles, key=lambda c: c.pass_rate).cycle_id
        
        # Prepare cycle data for template
        cycle_data = []
        for c in cycles:
            pass_rate = c.pass_rate * 100
            
            # Determine pass rate class
            if pass_rate >= 70:
                pass_rate_class = "high"
            elif pass_rate >= 40:
                pass_rate_class = "medium"
            else:
                pass_rate_class = "low"
            
            # Get quality scores from eval_metrics
            eval_metrics = c.eval_metrics or {}
            
            cycle_data.append({
                "cycle_id": c.cycle_id,
                "pass_rate": f"{pass_rate:.1f}",
                "pass_rate_class": pass_rate_class,
                "avg_quality": f"{c.avg_quality_score:.2f}",
                "duration": int(c.timing.get("duration_seconds", 0)),
                "status": c.status,
                "coherence": f"{eval_metrics.get('avg_coherence', 0):.2f}",
                "accuracy": f"{eval_metrics.get('avg_accuracy', 0):.2f}",
                "helpfulness": f"{eval_metrics.get('avg_helpfulness', 0):.2f}",
                "overall": f"{eval_metrics.get('avg_overall', 0):.2f}",
            })
        
        # Render template
        template = Template(HTML_REPORT_TEMPLATE)
        html = template.render(
            total_cycles=total_cycles,
            total_passed=total_passed,
            total_generated=total_generated,
            total_failed=total_failed,
            avg_pass_rate=f"{avg_pass_rate:.1f}",
            avg_quality=f"{avg_quality:.2f}",
            best_cycle=best_cycle,
            cycles=cycle_data,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
        
        # Save report
        if filename is None:
            filename = f"flywheel_report_{datetime.utcnow():%Y%m%d_%H%M%S}.html"
        
        report_path = self.output_dir / filename
        with open(report_path, "w") as f:
            f.write(html)
        
        return report_path
    
    def generate_json_report(
        self,
        cycles: List[CycleState],
        filename: Optional[str] = None,
    ) -> Path:
        """Generate JSON report from cycle data."""
        if not cycles:
            raise ValueError("No cycles to report on")
        
        data = {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_cycles": len(cycles),
                "total_passed_pairs": sum(len(c.passed_pairs) for c in cycles),
                "avg_pass_rate": sum(c.pass_rate for c in cycles) / len(cycles),
                "avg_quality_score": sum(c.avg_quality_score for c in cycles) / len(cycles),
            },
            "cycles": [c.to_dict() for c in cycles],
        }
        
        if filename is None:
            filename = f"flywheel_report_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
        
        report_path = self.output_dir / filename
        with open(report_path, "w") as f:
            json.dump(data, f, indent=2)
        
        return report_path


def create_report_generator(output_dir: Optional[str] = None) -> ReportGenerator:
    """Factory function to create a report generator."""
    return ReportGenerator(output_dir=output_dir)
