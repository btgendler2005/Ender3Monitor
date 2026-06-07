from prometheus_client import Counter, Gauge, Histogram, start_http_server

CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class MonitorMetrics:
    def __init__(self) -> None:
        self.frames_analyzed = Counter(
            "printer_frames_analyzed_total",
            "Total number of frames analyzed",
        )
        self.failures_detected = Counter(
            "printer_failures_detected_total",
            "Total number of failures detected above threshold",
            ["failure_type"],
        )
        self.current_confidence = Gauge(
            "printer_last_confidence_score",
            "Confidence score of the most recent analysis",
        )
        self.confidence_histogram = Histogram(
            "printer_confidence_score_histogram",
            "Distribution of confidence scores over time",
            buckets=CONFIDENCE_BUCKETS,
        )
        self.monitoring_active = Gauge(
            "printer_monitoring_active",
            "1 if monitoring is currently active, 0 otherwise",
        )

    def start_server(self, port: int = 8000) -> None:
        start_http_server(port)
        print(f"Prometheus metrics available at http://localhost:{port}/metrics")

    def record_analysis(self, confidence: float, failure_type: str | None, threshold: float) -> None:
        self.frames_analyzed.inc()
        self.current_confidence.set(confidence)
        self.confidence_histogram.observe(confidence)
        if failure_type and failure_type != "none" and confidence >= threshold:
            self.failures_detected.labels(failure_type=failure_type).inc()
