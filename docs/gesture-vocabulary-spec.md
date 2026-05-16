# Gesture Vocabulary Specification
**Project: Sigil**
**Document version: v0.2**
**Status: Approved, locked for Phase 0 scaffolding**

---

## 1. Design Principles

1. **Ergonomic.** Every static gesture must be holdable for ≥1.5 seconds without strain. No contorted finger positions.
2. **Discoverable.** A first-time user should be able to guess 80%+ of mappings without reading docs.
3. **Disambiguable.** No two reserved gestures may be confusable by MediaPipe (landmark configurations must be geometrically distinct).
4. **Tiered.** Vocabulary ships in three tiers (MVP, Standard, Power). Users grow into the system; they don't drown in it on day one.
5. **Cancel-first.** Every action is interruptible. **Open Palm anywhere = abort.** This is non-negotiable.
6. **Reversible.** Every action emits an undo event. Last-action undo is always available for 3 seconds.

---

## 2. State Machine Vocabulary

The system has four states. Different gestures are valid in each.

### DORMANT
- Camera is **OFF**. Microphone is on (low-power wake listener).
- Only the voice wake word `"Gesture ON"` transitions out → enters LISTENING.
- All gestures are ignored (impossible — no camera input).

### LISTENING
- Camera is ON, all Tier-1 and Tier-2 gestures fire actions.
- **Open Palm held 1.5s** → DORMANT.
- **60s idle** (no gesture fired) → DORMANT.
- Any successful action restarts the 60s idle timer.

### CONFIRMING
- Entered for **destructive actions only** (close window, lock screen, sleep, send-to-trash).
- **Thumbs Up** → execute → back to LISTENING.
- **Open Palm or Thumbs Down** → cancel → back to LISTENING.
- 3s timeout with no decision → cancel.

### EXECUTING
- Transient state during action dispatch.
- All gestures ignored to prevent double-fire.
- Returns to LISTENING on completion.

---

## 3. Tier 0 — Reserved Gestures (never remappable)

These three are hard-coded. The user can adjust thresholds but cannot change the action.

| Gesture | Action | Rationale |
|---|---|---|
| **Open Palm** (5 fingers spread, palm toward camera) | `meta.cancel` | Universal "stop". Must always work, even when state machine is in a weird state. |
| **Thumbs Up** | `meta.confirm` | Required in CONFIRMING. Universal "yes". |
| **Thumbs Down** | `meta.undo` (last action) | 3-second undo window after any action. |

**Why reserved:** if any of these are remappable, the user can paint themselves into a corner where they cannot cancel out of a misfiring state. This is a safety floor, not a feature.

---

## 4. Tier 1 — MVP Vocabulary (Phase 5 ship target)

Static-only. No context awareness. Six gestures total including reserved.

| # | Gesture | Action | Notes |
|---|---|---|---|
| 1 | Open Palm | `meta.cancel` | reserved |
| 2 | Thumbs Up | `meta.confirm` | reserved |
| 3 | Thumbs Down | `meta.undo` | reserved |
| 4 | **Fist** | `media.play_pause` | Sends VK_MEDIA_PLAY_PAUSE |
| 5 | **Peace** (✌, index + middle up) | `media.mute` | VK_VOLUME_MUTE toggle |
| 6 | **OK** (👌, thumb-index circle) | `window.maximize_toggle` | Affects focused window |

**Detection defaults (Tier 1):** confidence ≥0.85, ≥3 consecutive frames at 15 FPS (~200ms dwell).
**Reserved gestures use stricter threshold:** confidence ≥0.90, ≥4 frames.

---

## 5. Tier 2 — Standard Vocabulary (Phase 6+ ship target)

Adds dynamic gestures + context awareness (active-window aware mappings).

### Static additions
| Gesture | Action |
|---|---|
| Three Fingers Up | `system.notification_center` (Win+N) |
| Call / Shaka (🤙) | `system.screenshot` (Win+Shift+S) |

### Dynamic additions (context-aware)
| Gesture | Default action | Chrome / Edge | PowerPoint | Spotify / VLC |
|---|---|---|---|---|
| Swipe Left | `media.prev_track` | `browser.back` | `presentation.prev_slide` | `media.prev_track` |
| Swipe Right | `media.next_track` | `browser.forward` | `presentation.next_slide` | `media.next_track` |
| Swipe Up | `media.volume_up` (+5%) | — | — | — |
| Swipe Down | `media.volume_down` (−5%) | — | — | — |

**Context inference**: `GetForegroundWindow` → process name → lookup in mapping table. Fallback = default action.

**Detection defaults (dynamic):**
- Trajectory window: 16 frames (~1.1s at 15 FPS)
- Min displacement: 200 px on dominant axis (at 640×480 capture)
- Max duration: 500ms
- Min velocity: 400 px/s
- Sanity check: monotonic displacement on dominant axis (no zigzag)

---

## 6. Tier 3 — Power Vocabulary (opt-in, post-MVP)

Pointer control + two-hand modifiers + compound sequences. Off by default in config.

### Pointer mode (single hand)
| Gesture | Action |
|---|---|
| Index Point (sustained) | `mouse.move_to_landmark` — cursor follows fingertip (relative mapping, touchpad-style) |
| Index + Thumb pinch | `mouse.click_left` |
| Index + Middle pinch | `mouse.click_right` |
| Closing fist | `mouse.drag_start` (release to drop) |

### Two-hand modifiers
*Non-dominant hand holds a "mode" pose while dominant hand executes.*

| Non-dominant | Dominant | Effect |
|---|---|---|
| L-shape (thumb + index) held | Any Tier-1/2 gesture | Forces global context — overrides app-specific mapping |
| Open palm held | Index point | Region highlight/select (drag selection) |
| Pinch held | Pinch held (mirrored) | Zoom — move hands apart = zoom in, together = zoom out |

### Compound sequences (≤1.5s between gestures)
| Sequence | Action |
|---|---|
| Thumbs Up × 2 | `system.lock` (DESTRUCTIVE → CONFIRMING) |
| Fist → Open Palm | `window.close` (DESTRUCTIVE → CONFIRMING) |
| Three → Three | `window.switch_next` (Alt+Tab equivalent) |

---

## 7. Action Catalog

All action verbs the executor can dispatch. Config files reference verbs from this list — adding a new action means adding it here first.

### `media.*` (system-wide media keys via Win32 keybd_event)
- `media.play_pause`
- `media.mute`
- `media.volume_up` (default delta: 5%)
- `media.volume_down`
- `media.next_track`
- `media.prev_track`

### `window.*`
- `window.maximize_toggle`
- `window.minimize`
- `window.close` ⚠ destructive
- `window.switch_next`
- `window.switch_prev`
- `window.snap_left`
- `window.snap_right`

### `mouse.*`
- `mouse.move_to_landmark`
- `mouse.click_left`
- `mouse.click_right`
- `mouse.scroll_up` / `mouse.scroll_down`
- `mouse.drag_start` / `mouse.drag_end`

### `browser.*` (via hotkey injection; app-specific)
- `browser.back` / `browser.forward`
- `browser.refresh`
- `browser.new_tab`
- `browser.close_tab` ⚠ destructive
- `browser.next_tab` / `browser.prev_tab`

### `presentation.*`
- `presentation.next_slide` / `presentation.prev_slide`
- `presentation.start` / `presentation.exit`

### `system.*`
- `system.screenshot`
- `system.lock` ⚠ destructive
- `system.sleep` ⚠ destructive
- `system.search` (Win+S)
- `system.notification_center` (Win+N)

### `meta.*` (system control, HUD)
- `meta.confirm`
- `meta.cancel`
- `meta.undo`
- `meta.help_overlay` (show gesture legend HUD)
- `meta.toggle_listening`

---

## 8. Configuration Schema (YAML)

User config lives at `%APPDATA%/GestureOS/gestures.yaml`. Defaults are loaded from the bundled tier file first; user overrides applied on top.

```yaml
version: 1
tier: standard  # mvp | standard | power | custom

mappings:
  # Reserved (locked: true means cannot be remapped)
  - gesture: thumbs_up
    action: meta.confirm
    locked: true

  - gesture: open_palm
    action: meta.cancel
    locked: true

  - gesture: thumbs_down
    action: meta.undo
    locked: true

  # Static
  - gesture: fist
    action: media.play_pause

  - gesture: peace
    action: media.mute

  - gesture: ok
    action: window.maximize_toggle

  # Dynamic with app context
  - gesture: swipe_left
    action: media.prev_track          # fallback
    contexts:
      - app: chrome.exe
        action: browser.back
      - app: msedge.exe
        action: browser.back
      - app: powerpnt.exe
        action: presentation.prev_slide

  - gesture: swipe_right
    action: media.next_track
    contexts:
      - app: chrome.exe
        action: browser.forward
      - app: powerpnt.exe
        action: presentation.next_slide

# Per-gesture detection tuning
detection:
  defaults:
    static:
      min_confidence: 0.85
      min_frames: 3
    dynamic:
      window_frames: 16
      min_displacement_px: 200
      max_duration_ms: 500
      min_velocity_px_s: 400

  thumbs_up:
    min_confidence: 0.90
    min_frames: 4
  open_palm:
    min_confidence: 0.90
    min_frames: 4
  thumbs_down:
    min_confidence: 0.90
    min_frames: 4

# State machine tuning
state:
  listening_timeout_s: 60
  confirming_timeout_s: 3
  undo_window_s: 3
  debounce_ms: 400

# Actions that always require CONFIRMING
destructive_actions:
  - window.close
  - browser.close_tab
  - system.lock
  - system.sleep

# Wake word
wake:
  keyword: "gesture on"
  model: openwakeword
  sensitivity: 0.6
  cooldown_ms: 500

# Feedback
feedback:
  audio: subtle_beep    # silent | subtle_beep | tts
  hud: always           # always | only_when_listening | never
```

---

## 9. Forbidden / Avoided Gestures

These will not be added regardless of demand:

- **Relaxed / no-gesture hand** — explicit negative class in classifier. HaGRIDv2 supports this with `no_gesture` (~200K samples).
- **Middle finger up alone** — visually too close to "one" pose; also regionally offensive.
- **Hand flat from the side** — MediaPipe failure mode (landmarks collapse to a line).
- **Above-head gestures** — outside webcam FOV for ~80% of users with built-in laptop cams.
- **Asymmetric two-hand combos not in Tier 3** — too cognitively expensive; reserved for power tier only.
- **Gestures requiring specific finger isolation** (e.g., ring finger alone up) — most users physically can't do these reliably.

---

## 10. Disambiguation Rules

When two interpretations are possible:

1. **Reserved gestures always win.** Open Palm and "five" share landmarks; resolver always picks Open Palm = Cancel.
2. **Motion beats stillness for dynamic candidates.** If a fist is held still, it's "fist". If it's clearly moving (>400 px/s) it's a swipe, and the underlying static pose is ignored.
3. **Debounce same-gesture rapid repeats.** Same gesture firing within 400ms is collapsed to one event.
4. **Confidence tiebreaker.** If two non-reserved gestures fire above threshold within 100ms, higher-confidence wins.
5. **Context-mapping fallback.** If the gesture has no mapping for the active app, fall through to the default action. Never silently drop.

---

## 11. Open Questions / Decisions Deferred

These don't block Tier 1 development but need answers before Tier 2/3:

1. **Cursor mapping mode for Index Point** — absolute (calibrated 1:1 with screen) or relative (touchpad-style)?
   *Recommendation: relative for v1, absolute as opt-in for v2.*

2. **App context granularity** — process name only, or window title + process?
   *Recommendation: process name only for v1.*

3. **Wake word language** — English only or localized phrases?
   *Recommendation: English only for v1. Tooling for retraining OpenWakeWord on a custom locale is straightforward but data collection is expensive.*

4. **Audio feedback default** — silent, beep, or TTS confirmation?
   *Recommendation: subtle beep, user-configurable to silent or TTS.*

5. **False-positive autotuning** — if user undoes >N actions in M minutes, raise thresholds automatically?
   *Recommendation: log only in v1, autotune in v2.*

6. **Two-hand gestures with single-handed users** — what's the accessibility story?
   *Recommendation: Tier 3 must be skippable; ensure full system is usable on Tier 1+2 alone.*

---

## 12. Vocabulary Version Log

- **v0.2** — Idle timeout raised from 10s → 60s based on user feedback. Project named **Sigil**.
- **v0.1** — Initial draft. 3 tiers. Reserved: 3. Tier 1: +3 = 6 total. Tier 2: +6 = 12 total. Tier 3: +9 = 21 total. State machine: 4 states.
