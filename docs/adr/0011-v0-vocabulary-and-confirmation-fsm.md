# ADR-0011: V0 vocabulary, confirmation FSM, dynamic opt-in

**Status:** Accepted
**Date:** 2026-05-18

## Context

After Tier 1 (static gestures, 7 classes), Tier 2 (rule-based swipe
detection), and Tier 3 (pose-activated pointer mode) shipped, real-
world testing surfaced two issues:

1. **Mode collapse during dynamic-gesture attempts.** With swipes and
   pointer running alongside static, the static classifier would
   misfire during transitional poses (e.g., hand mid-swipe getting
   classified as `fist`, dispatching `media.play_pause` then a
   swipe correctly classified afterwards). Even with priority
   ordering, the cross-fire pattern was unpredictable enough to
   undermine demo confidence.

2. **Limited vocabulary.** Only 4 of the 7 trained static classes
   were wired to system actions. `open_palm` and `thumbs_up` had no
   commands at all (they are reserved for the CONFIRMING FSM, which
   no destructive action used yet). `no_gesture` is a sentinel. So
   the demo effectively shipped 4 commands.

The user requested V0 lockdown: more wired gestures, dynamic
detectors gated behind a CLI opt-in, retraining once with the
expanded vocabulary.

## Decisions

### D1: 13-class vocabulary (10 wired + 2 reserved + 1 sentinel)

| HaGRID | System name | Action | Notes |
|---|---|---|---|
| fist | fist | media.play_pause | preserved |
| peace | peace | media.mute | preserved |
| ok | ok | window.maximize | preserved |
| dislike | thumbs_down | system.undo | preserved + reserved |
| call | call | system.launch_terminal | new |
| rock | rock | media.next | new |
| stop | stop | window.close | new + **destructive** |
| three | three | system.screenshot | new |
| four | four | system.redo | new |
| one | one | window.minimize | new |
| palm | open_palm | — (reserved: cancel) | trained, no command |
| like | thumbs_up | — (reserved: confirm) | trained, no command |
| no_gesture | no_gesture | — (sentinel) | preserved |

Selection criteria for the new gestures:
- **Visually distinct from each other and from existing classes.**
  Dropped `two_up` (looks like peace), `three2` (looks like three),
  `stop_inverted` and `palm` variants that look like `stop`,
  `little_finger` and `holy` (low utility for desktop control).
- **Useful as desktop commands.** Each chosen gesture maps to an
  action a real user would want — close window, screenshot, redo,
  minimize, launch terminal, next track.

### D2: `window.close` is destructive and uses the existing CONFIRMING FSM

The interpreter (ADR-0005) already implements CONFIRMING state for
destructive actions; previous Tier 1 mappings had no destructive
actions to exercise this path. `window.close` is the first.

Behaviour:
1. `stop` gesture detected in LISTENING.
2. Interpreter transitions to CONFIRMING with `window.close` pending.
3. Overlay shows confirmation prompt (existing UI hook).
4. User shows `thumbs_up` within 5 s → close fires.
5. OR user shows `open_palm` → cancelled.
6. OR 5 s passes → cancelled silently.

The CONFIRMING_TIMEOUT_NS bumped from 4 s to 5 s based on user
feedback that 4 s felt rushed during real demonstrations.

### D3: Dynamic detectors (Tier 2 + Tier 3) become opt-in via `--ed`

The daemon constructor's `enable_swipes` and `enable_pointer`
parameters default to **False**. The CLI exposes `--ed` /
`--enable-dynamic` which sets both to True. Default `sigil daemon
run --overlay` ships **pure static-gesture mode** with no Tier 2 or
Tier 3 interference.

Rationale: the V0 demo prioritises reliability over feature surface.
Tier 2 and Tier 3 remain available for testing and for the report's
"Future Work" section, but they don't run during normal use. After
the V1 transformer-based dynamic classifier ships (post-V0), they
can move back to default-on.

### D4: HaGRID `dislike` → system name `thumbs_down`

HaGRID's label is `dislike` (not `thumbs_down`). The class allowlist
config (`configs/v0_classes.json`) maps `hagrid_label` to
`system_name`. The interpreter and mapping code use the system
names; the dataset prep step does the translation. This keeps the
mapping code human-readable while accepting whatever HaGRID's
underlying labels actually are.

Same translation: `like` → `thumbs_up`, `palm` → `open_palm`.

### D5: `system.launch_terminal` prefers focusing an existing terminal

The user's primary workflow runs Python scripts from VS Code's
integrated terminal. Launching a fresh Windows Terminal every time
the `call` gesture fires would clutter their workspace.

Implementation:
1. Enumerate top-level windows via `win32gui.EnumWindows`.
2. First-pass: match window class against known terminal classes
   (`ConsoleWindowClass`, `CASCADIA_HOSTING_WINDOW_CLASS`,
   `mintty`, `PuTTY`, etc.). If found, `SetForegroundWindow`.
3. Second-pass: match window title against known IDE titles
   (currently `Visual Studio Code`). If found and no dedicated
   terminal was found, focus the IDE.
4. Fallback: launch `wt.exe` (Windows Terminal) via subprocess,
   then `cmd.exe /k` if Windows Terminal isn't installed.

Terminal classes take priority over VS Code so that a user with
both open gets the more terminal-shaped target.

## Consequences

**Positive:**
- 10 wired commands instead of 4 — substantial vocabulary expansion
  without architectural changes.
- Destructive-action confirmation exercises a code path that was
  previously dead, validating the FSM design.
- Default-static demo is reliable; dynamic features available when
  needed via `--ed`.
- `launch_terminal` respects user workflow rather than spawning
  extra windows.

**Negative:**
- Retraining cost: ~3 hours of wall-clock. Mostly waiting.
- 13-class accuracy will be slightly lower than the 7-class baseline
  (expect 98–99% vs 99.65%). Trade-off: more vocabulary at slight
  per-class accuracy cost.
- Dynamic features now require explicit user knowledge of `--ed`
  flag. Documented in CLI help + RETRAINING.md.

**Neutral:**
- The reserved gestures (`open_palm`, `thumbs_up`) are still
  trained but never dispatched as commands. The classifier must
  learn them so the CONFIRMING FSM can recognise them — they are
  full citizens of the model, just not of the mapping.

## References

- `src/sigil/intelligence/interpreter.py` (mapping + timeout bump)
- `src/sigil/executor/verbs.py` (new verbs)
- `src/sigil/executor/registry.py` (new verb registrations)
- `src/sigil/cli/daemon.py` (`--ed` flag)
- `configs/v0_classes.json` (HaGRID allowlist)
- `RETRAINING.md` (step-by-step retraining commands)
- ADR-0005 (interpreter FSM, CONFIRMING state)
- ADR-0009 (swipe motion-state FSM — now opt-in)
- ADR-0010 (pointer pose-activated FSM — now opt-in)
