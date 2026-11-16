# Print stalls caused by USB serial traffic

Symptom: during SD prints the printer freezes for 1–3 s, usually at the end of
an extrusion line or before a travel. Unplugging USB makes prints visibly
smoother and faster.

## 1. Audit (before the fix)

Every serial write goes through `PrinterController.send()` in
`ender3monitor/printer.py`. No other module writes to the port.

| Command | Sent by | How often | Waits for `ok`? |
|---|---|---|---|
| `M105` temps | `refresh_status()` | every 5 s, always | Yes, up to 3 s |
| `M27` SD progress | `refresh_status()` | every 5 s, always | Yes, up to 3 s |
| `M31` print time | `refresh_status()` | every 5 s, always | Yes, up to 3 s |
| `M114` position | `refresh_status()` | every 5 s, **including mid-print** | Yes, up to 3 s |
| `M119` endstops / runout | `refresh_status()` | every 5 s while printing; ~60 s idle | Yes, up to 3 s |
| `M78` statistics | `refresh_status()` | ~every 60 s, **including mid-print** | Yes, up to 3 s |
| `M25` / `M24` | pause/resume (web, Telegram, auto-pause) | on demand | Yes |
| `M25`,`M104 S0`,`M140 S0`,`M107` | cool-down | on demand, back to back | Yes, each |
| `M112` | e-stop | on demand | Yes, up to 1 s |

What that added up to mid-print: **5 commands every 5 s (~60/min), plus M78
once a minute**, sent back to back.

- **One in flight?** Mostly. The serial lock is held from write to `ok`. But
  when a reply timed out (3 s), the next command went out anyway, while the
  firmware might still be holding the earlier one. So two could be queued.
- **`reset_input_buffer()` before every command** threw away anything the
  printer had sent on its own since the last read (late `ok`s, status lines).
  After a timeout, the late `ok` could then be read as the reply to the *next*
  command.
- **`M114` every 5 s mid-print.** On builds with `M114_LEGACY` (and older
  Creality-derived firmware) M114 calls `planner.synchronize()`. The printer
  finishes every queued move before it replies, so motion stops. This alone can
  produce the freeze pattern seen: stop at the end of a line, then continue.
- **Shared command queue.** Each host line takes a slot in the same small
  ring buffer the SD reader uses (BUFSIZE is often 4 on Creality builds). Five
  host commands per cycle keep slots taken, so the SD file can't keep the
  planner fed. The multi-line `M119` reply also costs firmware time on serial
  output.
- **Reader.** The passive marker reader (`listen()`) ran only while monitoring
  and printing. It never sent anything, and that is still true.
- **DTR.** `serial.Serial(port, …)` opens with pyserial defaults, and the
  code's comment assumed "Marlin resets on connect". Default Linux/macOS
  `HUPCL` drops DTR whenever the port closes, including when the stall
  watchdog force-closes it. The next reconnect then raises DTR again. That
  is an edge, which resets boards that wire DTR to reset.

## 2. What changed

**Auto-reports instead of polling** (`printer.py`, scheduler section). On
connect the monitor sends `M155 S5` and `M27 S10`. From then on, temperatures and
SD progress arrive unprompted and are parsed by the passive reader. It falls
back to polling (at most every 10 s) when:
- the firmware replies `Unknown command`, or
- the firmware accepts the command but no report arrives within 3 intervals.
  Stock Marlin without `AUTO_REPORT_SD_STATUS` treats `M27 S10` as a plain
  M27: it answers once and never again, so the "Unknown command" check alone
  would miss it.
If reports stop later (for example the board reset while USB stayed
connected), the monitor polls in the meantime and re-arms the report at most
once a minute. The `e3m_printer_autoreport_active{report}` gauge shows which
mode is in use.

**No motion-draining commands mid-print.** `send()` refuses `M114`, `M400`,
`G4`, `M503`, `M500`–`M502`, `G28`/`G29`, wait-for-temperature commands and
similar (`MOTION_SYNC_COMMANDS`) whenever a print *may* be running. That
includes the time after connecting, before the first SD status arrives. Refused
commands are counted in `e3m_printer_commands_suppressed_total{reason="motion_sync"}`.
`M114` is now polled only when idle. Mid-print, Z is the current layer's Z
from the matched G-code index (`status.z_estimated = True`). Without an index,
Z is unknown, and layer-mode timelapse falls back to time-based frames. The
head-parked timelapse marker does not depend on Z.

**One command in flight, with a rate limit.** The serial lock covers write →
`ok`. If a reply times out, the controller remembers that it is owed an `ok`
and discards the next stray `ok` instead of reading it as the reply to the
next command (the debt expires after 30 s). While a print may be running, routine commands go out
at most once per `PRINTER_MIN_COMMAND_INTERVAL_SECONDS` (default 5). When
the budget is tight, the most overdue poll wins, so everything still gets a
turn. Operator interventions (pause, resume, cool-down, e-stop) skip the rate limit
but still wait their turn on the lock. Normal traffic mid-print is now about
**3 commands per minute**: `M31` once a minute (elapsed time is extrapolated
between readings) and `M119` every 30 s. If neither auto-report works, it rises
to about 12 per minute, the cap at the default interval.

**Passive reader.** `listen()` now runs whenever the printer is connected
(30 ms slices while monitoring, for the timelapse marker; 250 ms otherwise to
save CPU). It parses auto-reports, `TIMELAPSE_SNAP`, pause and runout
announcements, and late `ok`s. It never writes. `send()` no longer calls
`reset_input_buffer()`; lines that were already waiting are handled first.

**DTR.** `_open_without_reset()` asks pyserial for DTR/RTS *asserted*, which
is the state the OS just set on open, so pyserial makes no transition. It then
clears `HUPCL`, so closing the port (reconnect, watchdog) leaves DTR asserted
and the next open sees no edge. The very first open after plugging in the
cable cannot avoid an edge, so connect while idle. I have not verified whether
the Ender-3 V3 SE board resets on DTR at all. Many 32-bit Creality boards
don't. The code no longer depends on it either way.

**Metrics** (`ops_metrics.py`). All existing metric names are unchanged.
New metrics:

| Metric | Use |
|---|---|
| `e3m_printer_commands_sent_total{command}` | Traffic by command |
| `e3m_printer_commands_per_minute` | Rolling 60 s count |
| `e3m_printer_ok_wait_seconds{command}` (histogram) | `ok` latency |
| `e3m_printer_slow_ok_total{command}` | `ok` slower than 1 s, timeouts included |
| `e3m_printer_ok_timeouts_total{command}` | No `ok` at all |
| `e3m_printer_commands_suppressed_total{command,reason}` | Guard / budget at work |
| `e3m_printer_autoreport_active{report}` | Auto-report vs polling |

Changed meaning: `e3m_printer_z_height_mm` is measured only when idle. Mid-print
it is estimated from the G-code index, and it is absent when no index matches.
The printer dashboard has a new "Serial link health" row showing all of this.

New setting: `PRINTER_MIN_COMMAND_INTERVAL_SECONDS` (default 5).

## 3. Test plan

Use the same G-code file, on the same SD card and spool, printed twice in a
row. Use a part with lots of short segments and travels (where stalls were
visible), at least 30 minutes long. The file must contain the
`M118 E1 TIMELAPSE_SNAP` layer-change block.

1. **Connected run.** Start the monitor *before* the print with the printer
   idle (this also confirms the board doesn't reboot on connect: the screen
   must not restart). Start monitoring in the web UI, then start the SD print.
   - Watch for freezes at line ends or before travels. Film a few minutes if
     you can.
   - In Grafana's "Serial link health" row, check:
     - **Commands per minute** is about 3 (or about 12 if auto-reports are
       unsupported).
     - **Auto-reports active** is 1 for temp and SD. If either is 0, your
       firmware is being polled, which is fine but is more traffic.
     - **Slow ok (>1 s)** stays at 0.
     - `e3m_printer_commands_suppressed_total{reason="motion_sync"}` stays at
       0. Anything above 0 means something tried to send a blocking command
       and was stopped.
   - Temperatures and progress keep updating in Grafana and the web UI.
   - Note the total print time on the printer's screen at the end.
2. **Disconnected run.** Unplug USB, print the same file, and note the total
   print time.
2b. **Cable in, monitor stopped (only if run 1 still stalls).** Leave USB
   plugged in but stop the monitor completely, then print again. If it still
   stalls, the monitor isn't the cause. Something else on the host is opening
   the port. On Linux, ModemManager probes new serial devices with AT
   commands, and brltty on Ubuntu grabs CH340 adapters. Check with
   `sudo lsof /dev/ttyUSB0` (or `ttyACM0`). Disable whatever shows up:
   `sudo systemctl disable --now ModemManager`.
3. **Pass criteria:**
   - no visible freezes on the connected run;
   - total print time within about 1% of the disconnected run;
   - Grafana updated throughout;
   - one timelapse frame per layer, with the head parked (the dashboard shows
     timelapse mode "parked");
   - slow-ok count 0.
4. **Reconnect check (optional, do it idle):** unplug and replug USB, or
   restart the monitor. The printer must not reboot, and the monitor must
   re-arm the auto-reports.

**If the slow-ok counter still climbs**, the monitor is now sending very
little, so a multi-second `ok` points to a second cause: the printer itself
stalling. A slow `ok` means the command waited behind motion in the firmware
queue. Look at which command it was and when it happened. The SD card has
already been ruled out (two cards, no change; only unplugging USB helped), so
first run step 2b above. If the freezes also happen with USB unplugged, likely
suspects are:
- a dense file (lots of tiny segments) that the firmware can't read and plan
  fast enough, whatever the card. Check whether the slicer's resolution or
  "max segment length" settings produce very short moves;
- planner buffer starvation at high speed with many short segments. Lower the
  speed on the affected features as a test;
- firmware-side work at layer change (the `M400` / `G4` in the snapshot block
  itself pauses motion briefly, which is by design).
A long `ok` wait during a long, slow move is also normal: the command simply
waits its turn. So judge slow oks together with visible freezes, not alone.
