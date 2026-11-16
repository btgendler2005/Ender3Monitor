"""Serial traffic budget: auto-reports, one command in flight, rate limit,
no motion-sync commands mid-print, passive reader, ok-wait metrics.

These guard the fix for print stalls caused by the monitor's own polling
(see docs/serial-stalls.md)."""
import threading
from collections import deque

import pytest

from ender3monitor import ops_metrics
from ender3monitor.printer import (
    FALLBACK_POLL_SECONDS,
    MOTION_SYNC_COMMANDS,
    PrinterController,
    command_code,
)


class FakeMarlin:
    """A Marlin-ish port: scripted replies, unsolicited output, a write log.

    `supports` lists the auto-reports the firmware has ("M155", "M27S").
    `reply_delay` (seconds of fake clock) is added before every "ok".
    """

    def __init__(self, clock, printing=True, supports=("M155", "M27S")):
        self.clock = clock
        self.printing = printing
        self.supports = set(supports)
        self.writes = []
        self.pending = b""          # unsolicited bytes (auto-reports, markers)
        self.reply = deque()
        self.reply_delay = 0.0
        self.swallow_ok = False     # drop the next "ok" (simulate a timeout)

    # pyserial surface
    @property
    def in_waiting(self):
        return len(self.pending)

    def read(self, n):
        out, self.pending = self.pending[:n], self.pending[n:]
        return out

    def reset_input_buffer(self):
        self.pending = b""

    def flush(self):
        pass

    def close(self):
        pass

    def readline(self):
        if not self.reply:
            return b""
        line = self.reply.popleft()
        if line.startswith(b"ok"):
            self.clock[0] += self.reply_delay
        return line

    def write(self, raw):
        cmd = raw.decode().strip()
        self.writes.append(cmd)
        code = command_code(cmd)
        out = []
        if cmd.startswith("M155"):
            if "M155" not in self.supports:
                out.append(f'echo:Unknown command: "{cmd}"')
        elif cmd.startswith("M27 S"):
            # Stock behaviour without AUTO_REPORT_SD_STATUS: S is ignored and
            # it just answers once, like a plain M27.
            out.append(self._sd_line())
        elif code == "M27":
            out.append(self._sd_line())
        elif code == "M105":
            out = []
            self.reply.append(b"ok T:210.0 /210.0 B:60.0 /60.0\n")
            return
        elif code == "M31":
            out.append("echo:Print time: 0h 10m 0s")
        elif code == "M119":
            out += ["Reporting endstop status", "x_min: open", "filament: TRIGGERED"]
        elif code == "M114":
            out.append("X:0.00 Y:0.00 Z:12.00 E:0.00")
        for line in out:
            self.reply.append((line + "\n").encode())
        if self.swallow_ok:
            self.swallow_ok = False
        else:
            self.reply.append(b"ok\n")

    def _sd_line(self):
        return "SD printing byte 500/1000" if self.printing else "Not SD printing"

    def push(self, text):
        self.pending += text.encode()


def make(printing=True, supports=("M155", "M27S"), interval=5.0):
    clock = [1000.0]
    pc = PrinterController(port="", min_command_interval=interval,
                           ok_timeout=0.05, clock=lambda: clock[0])
    fake = FakeMarlin(clock, printing=printing, supports=supports)
    pc._serial = fake
    pc.status.connected = True
    return pc, fake, clock


def run(pc, clock, seconds, step=1.0):
    t = 0.0
    while t < seconds:
        clock[0] += step
        t += step
        pc.refresh_status()


def sample(name, labels=None):
    v = ops_metrics.REGISTRY.get_sample_value(name, labels or {})
    return v or 0.0


# ── auto-reports ─────────────────────────────────────────────────────────────

def test_connect_arms_both_auto_reports_and_then_stays_quiet():
    pc, fake, clock = make()
    run(pc, clock, 10)
    assert fake.writes[:2] == ["M27 S10", "M155 S5"]
    # The firmware now pushes both reports on its own.
    fake.writes.clear()
    for _ in range(12):                      # one minute
        fake.push(" T:210.1 /210.0 B:60.0 /60.0 @:0 B@:0\n")
        fake.push("SD printing byte 600/1000\n")
        pc.listen(0, None)
        run(pc, clock, 5)
    assert pc.status.nozzle_temp == 210.1 and pc.status.progress == 0.6
    # No temperature or progress polls at all; only the slow extras.
    assert "M105" not in fake.writes and "M27" not in fake.writes
    assert set(fake.writes) <= {"M31", "M119"}
    assert len(fake.writes) <= 4


def test_unknown_command_falls_back_to_polling_no_faster_than_10s():
    pc, fake, clock = make(supports=())
    run(pc, clock, 120)
    assert pc._autoreport["temp"] is False
    sent_at = []
    fake.writes.clear()
    orig = fake.write
    fake.write = lambda raw: (sent_at.append((clock[0], raw.decode().strip())), orig(raw))
    run(pc, clock, 120)
    for code in ("M105", "M27"):
        times = [t for t, c in sent_at if c == code]
        assert len(times) >= 5
        assert min(b - a for a, b in zip(times, times[1:])) >= FALLBACK_POLL_SECONDS


def test_m27_s_accepted_but_silent_falls_back_to_polling():
    pc, fake, clock = make(supports=("M155",))
    run(pc, clock, 60)
    assert pc._autoreport["sd"] is False
    assert pc.status.printing is True        # learned from the M27 S reply itself
    fake.writes.clear()
    run(pc, clock, 60)
    assert "M27" in fake.writes


# ── the traffic budget ───────────────────────────────────────────────────────

def test_at_most_one_command_per_interval_while_printing():
    pc, fake, clock = make(supports=())       # worst case: everything polled
    sent_at = []
    orig = fake.write
    fake.write = lambda raw: (sent_at.append(clock[0]), orig(raw))
    run(pc, clock, 300, step=0.5)
    gaps = [b - a for a, b in zip(sent_at, sent_at[1:])]
    assert gaps and min(gaps) >= 5.0
    # …and every kind of poll still got a turn.
    assert {"M27", "M105", "M31", "M119"} <= set(fake.writes)


def test_interval_is_configurable():
    pc, fake, clock = make(supports=(), interval=10.0)
    sent_at = []
    orig = fake.write
    fake.write = lambda raw: (sent_at.append(clock[0]), orig(raw))
    run(pc, clock, 200)
    assert min(b - a for a, b in zip(sent_at, sent_at[1:])) >= 10.0


def test_interventions_are_never_rate_limited():
    pc, fake, clock = make()
    pc.send("M105")
    pc.pause()
    pc.cooldown()
    assert fake.writes == ["M105", "M25", "M25", "M104 S0", "M140 S0", "M107"]


def test_one_command_in_flight():
    pc, fake, clock = make(printing=False)
    pc._print_state_known = True
    inside = []
    orig = fake.write

    def write(raw):
        inside.append(pc._lock.locked())
        orig(raw)
    fake.write = write
    threads = [threading.Thread(target=pc.send, args=("M105",), kwargs={"priority": True})
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert inside == [True] * 4     # every write happened under the serial lock


# ── motion-sync commands ─────────────────────────────────────────────────────

@pytest.mark.parametrize("gcode", ["M114", "M400", "G4 P100", "M503", "M500"])
def test_motion_sync_commands_refused_while_printing(gcode):
    pc, fake, clock = make()
    pc._print_state_known = True
    pc.status.printing = True
    assert pc.send(gcode, priority=True) == ""
    assert fake.writes == []


def test_motion_sync_commands_refused_before_print_state_is_known():
    pc, fake, clock = make(printing=False)
    assert pc.send("M114") == ""
    assert fake.writes == []


def test_m114_is_polled_only_when_idle():
    pc, fake, clock = make(printing=True)
    run(pc, clock, 300)
    assert not any(command_code(w) in MOTION_SYNC_COMMANDS for w in fake.writes)
    fake.printing = False
    fake.push("Done printing file\n")
    pc.listen(0, None)
    assert pc.status.printing is False
    run(pc, clock, 60)
    assert "M114" in fake.writes and pc.status.z_height == 12.0


def test_print_start_clears_stale_idle_z():
    pc, fake, clock = make(printing=False)
    pc.status.z_height = 80.0                  # parked high after the last job
    fake.push("Not SD printing\n")
    pc.listen(0, None)
    fake.push("SD printing byte 1/1000\n")
    pc.listen(0, None)
    assert pc.status.printing and pc.status.z_height is None


# ── the passive reader ───────────────────────────────────────────────────────

def test_reader_parses_reports_and_marker_without_sending_anything():
    pc, fake, clock = make()
    hits = []
    pc.on_snapshot = lambda: hits.append(1)
    fake.push(" T:205.0 /210.0 B:59.0 /60.0 @:0 B@:0\nSD printing byte 2")
    pc.listen(0, None)
    fake.push("50/1000\necho:TIMELAPSE_SNAP\n")   # a report split across reads
    pc.listen(0, None)
    assert fake.writes == []
    assert pc.status.nozzle_temp == 205.0 and pc.status.bed_temp == 59.0
    assert pc.status.file_position == 250 and pc.status.printing is True
    assert hits == [1]
    assert pc.autoreport_active("temp") and pc.autoreport_active("sd")


def test_reports_waiting_before_a_command_are_not_discarded():
    pc, fake, clock = make()
    fake.push(" T:199.0 /210.0 B:58.0 /60.0\n")
    pc.send("M31", priority=True)
    assert pc.status.nozzle_temp == 199.0


def test_late_ok_from_a_timed_out_command_is_not_taken_as_the_next_ones():
    pc, fake, clock = make()
    fake.swallow_ok = True
    assert pc.send("M31", priority=True).endswith("0s")   # no ok → timeout
    fake.reply.append(b"ok\n")   # …its ok turns up late, ahead of the next reply
    resp = pc.send("M119", priority=True)
    assert "filament: TRIGGERED" in resp and resp.endswith("ok")


# ── metrics ──────────────────────────────────────────────────────────────────

def test_ok_wait_histogram_and_slow_counter():
    pc, fake, clock = make()
    lbl = {"command": "M31"}
    sent0 = sample("e3m_printer_commands_sent_total", lbl)
    count0 = sample("e3m_printer_ok_wait_seconds_count", lbl)
    slow0 = sample("e3m_printer_slow_ok_total", lbl)
    fake.reply_delay = 0.2
    pc.send("M31", priority=True)
    fake.reply_delay = 2.5
    pc.send("M31", priority=True)
    assert sample("e3m_printer_commands_sent_total", lbl) == sent0 + 2
    assert sample("e3m_printer_ok_wait_seconds_count", lbl) == count0 + 2
    assert sample("e3m_printer_slow_ok_total", lbl) == slow0 + 1
    assert pc.commands_last_minute() == 2


def test_suppressed_commands_are_counted():
    pc, fake, clock = make()
    lbl = {"command": "M114", "reason": "motion_sync"}
    before = sample("e3m_printer_commands_suppressed_total", lbl)
    pc.send("M114")
    assert sample("e3m_printer_commands_suppressed_total", lbl) == before + 1
