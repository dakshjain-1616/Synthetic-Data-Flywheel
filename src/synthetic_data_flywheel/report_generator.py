"""Report Generator - Jinja2-based HTML dashboard with per-cycle metrics visualization."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from jinja2 import Template

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.models import CycleState

logger = structlog.get_logger()


# HTML Template for Flywheel Report
HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Synthetic Data Flywheel Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            color: white;
            padding: 40px 0;
        }
        
        header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }
        
        .stat-card h3 {
            color: #666;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #333;
        }
        
        .stat-card.passed .stat-value { color: #10b981; }
        .stat-card.failed .stat-value { color: #ef4444; }
        .stat-card.total .stat-value { color: #3b82f6; }
        
        .charts-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .chart-card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .chart-card h3 {
            margin-bottom: 20px;
            color: #333;
        }
        
        .cycles-table {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }
        
        th {
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
        }
        
        tr:hover {
            background: #f9fafb;
        }
        
        .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .status-completed {
            background: #d1fae5;
            color: #065f46;
        }
        
        .status-failed {
            background: #fee2e2;
            color: #991b1b;
        }
        
        .status-pending {
            background: #fef3c7;
            color: #92400e;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #34d399);
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        footer {
            text-align: center;
            color: white;
            padding: 20px;
            opacity: 0.8;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔄 Synthetic Data Flywheel</h1>
            <p>Generated on {{ generated_at }}</p>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card total">
                <h3>Total Cycles</h3>
                <div class="stat-value">{{ total_cycles }}</div>
            </div>
            <div class="stat-card total">
                <h3>Total Samples Generated</h3>
                <div class="stat-value">{{ total_samples }}</div>
            </div>
            <div class="stat-card passed">
                <h3>Samples Passed</h3>
                <div class="stat-value">{{ samples_passed }}</div>
            </div>
            <div class="stat-card failed">
                <h3>Overall Pass Rate</h3>
                <div class="stat-value">{{ "%.1f"|format(pass_rate * 100) }}%</div>
            </div>
        </div>
        
        <div class="charts-section">
            <div class="chart-card">
                <h3>Pass Rate per Cycle</h3>
                <canvas id="passRateChart"></canvas>
            </div>
            <div class="chart-card">
                <h3>Average Quality Score per Cycle</h3>
                <canvas id="qualityChart"></canvas>
            </div>
        </div>
        
        <div class="cycles-table">
            <h3>Cycle Details</h3>
            <table>
                <thead>
                    <tr>
                        <th>Cycle</th>
                        <th>Status</th>
                        <th>Generated</th>
                        <th>Passed</th>
                        <th>Pass Rate</th>
                        <th>Avg Quality</th>
                        <th>Duration (s)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for cycle in cycles %}
                    <tr>
                        <td>{{ cycle.cycle_id }}</td>
                        <td><span class="status-badge status-{{ cycle.status }}">{{ cycle.status }}</span></td>
                        <td>{{ cycle.generated_count }}</td>
                        <td>{{ cycle.passed_count }}</td>
                        <td>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {{ cycle.pass_rate * 100 }}%"></div>
                            </div>
                            {{ "%.1f"|format(cycle.pass_rate * 100) }}%
                        </td>
                        <td>{{ "%.2f"|format(cycle.avg_quality) }}</td>
                        <td>{{ "%.0f"|format(cycle.duration) if cycle.duration else "N/A" }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <footer>
            <p>Synthetic Data Flywheel v0.1.0 | Report generated automatically</p>
        </footer>
    </div>
    
    <script>
        const cycleLabels = {{ cycle_labels | safe }};
        const passRates = {{ pass_rates | safe }};
        const qualityScores = {{ quality_scores | safe }};
        
        // Pass Rate Chart
        new Chart(document.getElementById('passRateChart'), {
            type: 'line',
            data: {
                labels: cycleLabels,
                datasets: [{
                    label: 'Pass Rate',
                    data: passRates,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        });
        
        // Quality Score Chart
        new Chart(document.getElementById('qualityChart'), {
            type: 'line',
            data: {
                labels: cycleLabels,
                datasets: [{
                    label: 'Avg Quality',
                    data: qualityScores,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 10
                    }
                }
            }
        });
    </script>
</body>
</html>"""


class ReportGenerator:
    """Generator for HTML reports with metrics visualization."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize report generator.
        
        Args:
            output_dir: Output directory for reports
        """
        settings = get_settings()
        self.output_dir = Path(output_dir or settings.report_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.template = Template(HTML_REPORT_TEMPLATE)
        
        logger.info(
            "report_generator_initialized",
            output_dir=str(self.output_dir),
        )
    
    def _prepare_cycle_data(self, cycles: List[CycleState]) -> List[Dict[str, Any]]:
        """Prepare cycle data for template."""
        cycle_data = []
        for cycle in cycles:
            generated = len(cycle.generated_pairs)
            passed = sum(1 for j in cycle.judgments if j.passed) if cycle.judgments else 0
            
            data = {
                "cycle_id": cycle.cycle_id,
                "status": cycle.status,
                "generated_count": generated,
                "passed_count": passed,
                "pass_rate": cycle.pass_rate,
                "avg_quality": cycle.avg_quality_score,
                "duration": cycle.duration_seconds,
            }
            cycle_data.append(data)
        
        return cycle_data
    
    def generate_flywheel_report(
        self,
        cycles: List[CycleState],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Generate comprehensive flywheel report.
        
        Args:
            cycles: List of completed cycles
            output_path: Optional output path
            
        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = self.output_dir / f"flywheel_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Calculate statistics
        total_cycles = len(cycles)
        total_samples = sum(len(c.generated_pairs) for c in cycles)
        samples_passed = sum(
            sum(1 for j in c.judgments if j.passed)
            for c in cycles if c.judgments
        )
        pass_rate = samples_passed / total_samples if total_samples else 0.0
        
        # Prepare chart data
        cycle_labels = [str(c.cycle_id) for c in cycles]
        pass_rates = [c.pass_rate for c in cycles]
        quality_scores = [c.avg_quality_score for c in cycles]
        
        # Prepare cycle data
        cycle_data = self._prepare_cycle_data(cycles)
        
        # Render template
        html = self.template.render(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_cycles=total_cycles,
            total_samples=total_samples,
            samples_passed=samples_passed,
            pass_rate=pass_rate,
            cycles=cycle_data,
            cycle_labels=json.dumps(cycle_labels),
            pass_rates=json.dumps(pass_rates),
            quality_scores=json.dumps(quality_scores),
        )
        
        # Write report
        with open(output_path, "w") as f:
            f.write(html)
        
        logger.info(
            "flywheel_report_generated",
            path=str(output_path),
            cycles=total_cycles,
            samples=total_samples,
        )
        
        return output_path
    
    def generate_cycle_report(
        self,
        cycle: CycleState,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Generate report for a single cycle.
        
        Args:
            cycle: Cycle state to report on
            output_path: Optional output path
            
        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = self.output_dir / f"cycle_{cycle.cycle_id:03d}_report.html"
        
        # Generate single-cycle report using same template
        return self.generate_flywheel_report([cycle], output_path)
    
    def generate_comparison_report(
        self,
        cycles: List[CycleState],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Generate comparison report across cycles.
        
        Args:
            cycles: List of cycles to compare
            output_path: Optional output path
            
        Returns:
            Path to generated report
        """
        if len(cycles) < 2:
            raise ValueError("Need at least 2 cycles for comparison")
        
        if output_path is None:
            output_path = self.output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Use same template for now (could be extended)
        return self.generate_flywheel_report(cycles, output_path)


def create_report_generator(output_dir: Optional[str] = None) -> ReportGenerator:
    """Factory function to create a report generator.
    
    Args:
        output_dir: Output directory for reports
        
    Returns:
        Configured ReportGenerator
    """
    return ReportGenerator(output_dir=output_dir)
