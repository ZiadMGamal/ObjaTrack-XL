from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from jinja2 import Template

from src.utils.logger import get_logger

logger = get_logger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background:#0f0f1a;color:#e0e0e0;padding:20px}
.container{max-width:1200px;margin:0 auto}
h1{color:#00d4ff;font-size:2em;margin-bottom:5px}
h2{color:#00d4ff;font-size:1.3em;margin:20px 0 10px;border-bottom:1px solid #333;padding-bottom:5px}
.subtitle{color:#888;font-size:0.9em;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:15px;margin:15px 0}
.card{background:#1a1a2e;border-radius:10px;padding:20px;border:1px solid #333}
.card-title{color:#00d4ff;font-size:0.85em;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.card-value{font-size:2em;font-weight:bold;color:#fff}
.card-unit{font-size:0.5em;color:#888;margin-left:5px}
table{width:100%;border-collapse:collapse;margin:10px 0}
th{background:#16213e;color:#00d4ff;padding:10px;text-align:left;font-size:0.85em}
td{padding:10px;border-bottom:1px solid #222;font-size:0.9em}
tr:hover{background:#16213e33}
.good{color:#00ff88}.warn{color:#ffaa00}.bad{color:#ff4444}
.bar{height:20px;border-radius:10px;background:#16213e;overflow:hidden;margin:5px 0}
.bar-fill{height:100%;border-radius:10px;background:linear-gradient(90deg,#00d4ff,#00ff88)}
.footer{text-align:center;color:#555;margin-top:30px;padding-top:15px;border-top:1px solid #222;font-size:0.8em}
</style>
</head>
<body>
<div class="container">
<h1>{{ title }}</h1>
<p class="subtitle">Generated: {{ timestamp }} | Author: {{ author }}</p>

<h2>Performance Summary</h2>
<div class="grid">
{% for card in summary_cards %}
<div class="card">
<div class="card-title">{{ card.label }}</div>
<div class="card-value {% if card.status %}{{ card.status }}{% endif %}">{{ card.value }}<span class="card-unit">{{ card.unit }}</span></div>
</div>
{% endfor %}
</div>

{% if benchmark_table %}
<h2>Model Benchmark Comparison</h2>
<table>
<tr>{% for h in benchmark_headers %}<th>{{ h }}</th>{% endfor %}</tr>
{% for row in benchmark_table %}
<tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>
{% endfor %}
</table>
{% endif %}

{% if latency_breakdown %}
<h2>Pipeline Latency Breakdown</h2>
{% for stage, info in latency_breakdown.items() %}
<div style="margin:8px 0">
<div style="display:flex;justify-content:space-between;font-size:0.9em">
<span>{{ stage }}</span><span>{{ info.avg_ms }}ms ({{ info.percentage }}%)</span>
</div>
<div class="bar"><div class="bar-fill" style="width:{{ info.percentage }}%"></div></div>
</div>
{% endfor %}
{% endif %}

{% if system_info %}
<h2>System Information</h2>
<table>
{% for key, value in system_info.items() %}
<tr><td style="width:30%;color:#888">{{ key }}</td><td>{{ value }}</td></tr>
{% endfor %}
</table>
{% endif %}

{% for section in extra_sections %}
<h2>{{ section.title }}</h2>
{% if section.type == "table" %}
<table>
<tr>{% for h in section.headers %}<th>{{ h }}</th>{% endfor %}</tr>
{% for row in section.rows %}
<tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>
{% endfor %}
</table>
{% elif section.type == "text" %}
<p style="margin:10px 0;line-height:1.6">{{ section.content }}</p>
{% endif %}
{% endfor %}

<div class="footer">ObjaTrack-XL &copy; {{ year }} {{ author }} | High-Performance Object Detection & Tracking</div>
</div>
</body>
</html>"""


class ReportGenerator:

    def __init__(
        self,
        output_path: str = "outputs/reports/report.html",
        title: str = "ObjaTrack-XL Performance Report",
        author: str = "Ziad Mohamed Gamal",
    ) -> None:
        self._output_path = Path(output_path)
        self._title = title
        self._author = author
        self._summary_cards: list[dict[str, Any]] = []
        self._benchmark_headers: list[str] = []
        self._benchmark_table: list[list[str]] = []
        self._latency_breakdown: dict[str, Any] = {}
        self._system_info: dict[str, Any] = {}
        self._extra_sections: list[dict[str, Any]] = []

    def add_summary_card(self, label: str, value: str, unit: str = "", status: str = "") -> None:
        self._summary_cards.append({
            "label": label,
            "value": value,
            "unit": unit,
            "status": status,
        })

    def set_benchmark_table(self, headers: list[str], rows: list[list[str]]) -> None:
        self._benchmark_headers = headers
        self._benchmark_table = rows

    def set_latency_breakdown(self, breakdown: dict[str, Any]) -> None:
        self._latency_breakdown = breakdown

    def set_system_info(self, info: dict[str, Any]) -> None:
        self._system_info = info

    def add_table_section(self, title: str, headers: list[str], rows: list[list[str]]) -> None:
        self._extra_sections.append({
            "title": title,
            "type": "table",
            "headers": headers,
            "rows": rows,
        })

    def add_text_section(self, title: str, content: str) -> None:
        self._extra_sections.append({
            "title": title,
            "type": "text",
            "content": content,
        })

    def generate(self) -> str:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        template = Template(HTML_TEMPLATE)
        html = template.render(
            title=self._title,
            author=self._author,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            year=time.strftime("%Y"),
            summary_cards=self._summary_cards,
            benchmark_headers=self._benchmark_headers,
            benchmark_table=self._benchmark_table,
            latency_breakdown=self._latency_breakdown,
            system_info=self._system_info,
            extra_sections=self._extra_sections,
        )

        with open(self._output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("report_generated", path=str(self._output_path))
        return str(self._output_path)

    def generate_from_metrics(self, metrics: dict[str, Any]) -> str:
        fps_data = metrics.get("fps", {})
        self.add_summary_card("Current FPS", str(fps_data.get("current_fps", 0)), "fps",
                              "good" if fps_data.get("current_fps", 0) >= 25 else "warn")
        self.add_summary_card("Average FPS", str(fps_data.get("average_fps", 0)), "fps")
        self.add_summary_card("Total Frames", str(fps_data.get("total_frames", 0)), "frames")
        self.add_summary_card("Uptime", str(metrics.get("uptime_seconds", 0)), "seconds")

        if "pipeline_breakdown" in metrics:
            self.set_latency_breakdown(metrics["pipeline_breakdown"])

        if "system" in metrics and "info" in metrics["system"]:
            self.set_system_info(metrics["system"]["info"])

        return self.generate()
