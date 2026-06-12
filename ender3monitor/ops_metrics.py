"""Operational / SRE metrics for the web app.

Registered on the default Prometheus registry, so they're served from the same
endpoint as the print metrics (default http://localhost:8000/metrics).

prometheus_client already auto-registers process_* and python_* collectors, so
process CPU / memory / open-FDs / GC come for free. This module adds:
  • HTTP request rate, latency, and in-flight count (FastAPI middleware)
  • AI-analysis latency + error count
  • camera stream capture health
  • printer USB connection / reconnect / serial-error counts
  • websocket client count, Telegram command count, push notification results
  • host CPU / memory / disk (via psutil, optional)
"""
from prometheus_client import Counter, Gauge, Histogram, REGISTRY

# ── HTTP server (FastAPI) ─────────────────────────────────────────────────────
http_requests_total = Counter(
    "e3m_http_requests_total", "HTTP requests handled",
    ["method", "path", "status"],
)
http_request_duration_seconds = Histogram(
    "e3m_http_request_duration_seconds", "HTTP request latency (seconds)",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
http_requests_in_progress = Gauge(
    "e3m_http_requests_in_progress", "In-flight HTTP requests",
)

# ── AI analysis ───────────────────────────────────────────────────────────────
analysis_duration_seconds = Histogram(
    "e3m_analysis_duration_seconds", "AI analysis call latency (seconds)",
    ["backend"],
    buckets=(0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120),
)
analysis_errors_total = Counter(
    "e3m_analysis_errors_total", "AI analysis errors", ["backend"],
)

# ── Camera stream ─────────────────────────────────────────────────────────────
camera_frames_total = Counter(
    "e3m_camera_frames_total", "Stream frames captured", ["result"],  # ok | fail
)

# ── Printer / serial ──────────────────────────────────────────────────────────
printer_connected = Gauge("e3m_printer_connected", "1 if the printer is connected over USB")
printer_reconnects_total = Counter("e3m_printer_reconnects_total", "Printer USB reconnects")
printer_serial_errors_total = Counter("e3m_printer_serial_errors_total", "Serial command failures")

# ── Printer telemetry (physical state, updated each poll) ─────────────────────
printer_nozzle_temp = Gauge("e3m_printer_nozzle_temp_celsius", "Nozzle temperature (°C)")
printer_nozzle_target = Gauge("e3m_printer_nozzle_target_celsius", "Nozzle target temperature (°C)")
printer_bed_temp = Gauge("e3m_printer_bed_temp_celsius", "Bed temperature (°C)")
printer_bed_target = Gauge("e3m_printer_bed_target_celsius", "Bed target temperature (°C)")
printer_z_height = Gauge("e3m_printer_z_height_mm", "Current nozzle Z height (mm)")
printer_progress = Gauge("e3m_printer_progress_ratio", "Print progress (0..1)")
printer_printing = Gauge("e3m_printer_printing", "1 while a print is active")
printer_elapsed_seconds = Gauge("e3m_printer_elapsed_seconds", "Elapsed print time (s)")
printer_remaining_seconds = Gauge("e3m_printer_remaining_seconds", "Estimated remaining print time (s)")
printer_lifetime_seconds = Gauge("e3m_printer_lifetime_seconds", "Lifetime print time from EEPROM (s)")

_NAN = float("nan")


def update_printer(status) -> None:
    """Mirror the live PrinterStatus into Prometheus gauges. NaN = unknown so
    graphs show gaps instead of stale values when disconnected."""
    def g(gauge, val):
        gauge.set(val if val is not None else _NAN)

    printer_connected.set(1 if status.connected else 0)
    printer_printing.set(1 if status.printing else 0)
    g(printer_nozzle_temp, status.nozzle_temp)
    g(printer_nozzle_target, status.nozzle_target)
    g(printer_bed_temp, status.bed_temp)
    g(printer_bed_target, status.bed_target)
    g(printer_z_height, status.z_height)
    g(printer_progress, status.progress)
    g(printer_elapsed_seconds, status.elapsed_seconds)
    g(printer_remaining_seconds, status.remaining_seconds)
    g(printer_lifetime_seconds, status.lifetime_print_seconds)

# ── WebSocket / notifications ─────────────────────────────────────────────────
ws_clients = Gauge("e3m_ws_clients", "Connected dashboard websocket clients")
telegram_commands_total = Counter("e3m_telegram_commands_total", "Telegram commands handled", ["command"])
push_notifications_total = Counter("e3m_push_notifications_total", "Push notifications", ["channel", "result"])


# ── Host system metrics (psutil, optional) ────────────────────────────────────
class _SystemCollector:
    """Reads host CPU / memory / disk on each scrape — no background thread."""

    def collect(self):
        try:
            import psutil
            from prometheus_client.core import GaugeMetricFamily
        except Exception:
            return
        try:
            yield GaugeMetricFamily("e3m_system_cpu_percent",
                                    "Host CPU utilization (%)",
                                    value=psutil.cpu_percent(interval=None))
            vm = psutil.virtual_memory()
            yield GaugeMetricFamily("e3m_system_memory_percent",
                                    "Host memory utilization (%)", value=vm.percent)
            yield GaugeMetricFamily("e3m_system_memory_used_bytes",
                                    "Host memory used (bytes)", value=float(vm.used))
            yield GaugeMetricFamily("e3m_system_memory_total_bytes",
                                    "Host memory total (bytes)", value=float(vm.total))
            du = psutil.disk_usage("/")
            yield GaugeMetricFamily("e3m_system_disk_percent",
                                    "Root filesystem utilization (%)", value=du.percent)
            # Process-level (cross-platform; prometheus_client's process_* needs
            # /proc and is unavailable on macOS).
            proc = psutil.Process()
            with proc.oneshot():
                yield GaugeMetricFamily("e3m_process_memory_bytes",
                                        "This process resident memory (bytes)",
                                        value=float(proc.memory_info().rss))
                yield GaugeMetricFamily("e3m_process_cpu_percent",
                                        "This process CPU utilization (%)",
                                        value=proc.cpu_percent(interval=None))
                yield GaugeMetricFamily("e3m_process_threads",
                                        "This process thread count",
                                        value=float(proc.num_threads()))
        except Exception:
            return


_system_collector_registered = False


def register_system_collector() -> None:
    """Register the psutil host collector once (no-op if psutil is missing)."""
    global _system_collector_registered
    if _system_collector_registered:
        return
    try:
        REGISTRY.register(_SystemCollector())
        _system_collector_registered = True
    except Exception:
        pass
