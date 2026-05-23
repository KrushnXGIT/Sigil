"""Voice wake-word listener using OpenWakeWord.

Listens on the system microphone in a background thread. When the
configured wake phrase (default "hey jarvis") is detected above the
score threshold, it raises a thread-safe flag that the daemon's main
loop consumes to pull the interpreter out of DORMANT.

Design notes:
  - The audio loop runs in its own thread. It NEVER touches the
    interpreter directly — the interpreter is single-threaded and
    not safe to drive from two threads. Instead the listener sets a
    threading.Event; the daemon's main loop calls
    ``consume_wake()`` once per frame and performs the activation
    itself, on the main thread.
  - A post-detection cooldown prevents a single utterance (which
    spans multiple audio chunks) from firing several activations.
  - Heavy deps (openwakeword, sounddevice) are imported lazily so
    importing this module never breaks a camera-only run.

Audio format: OpenWakeWord expects 16 kHz, mono, 16-bit PCM, fed in
1280-sample (80 ms) chunks. sounddevice captures exactly that.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from sigil.logging import get_logger

log = get_logger(__name__)


# OpenWakeWord's fixed audio contract.
_SAMPLE_RATE = 16_000
_CHANNELS = 1
_CHUNK = 1280  # 80 ms at 16 kHz — the frame size OWW expects

# Default pretrained model key. Available: "alexa", "hey_mycroft",
# "hey_jarvis", "timer", "weather".
DEFAULT_WAKE_MODEL = "hey_jarvis"

# Human-readable phrase per model key, for banners/logs.
WAKE_PHRASES: dict[str, str] = {
    "hey_jarvis": "Hey Jarvis",
    "alexa": "Alexa",
    "hey_mycroft": "Hey Mycroft",
    "timer": "Timer",
    "weather": "Weather",
}


class WakeWordError(RuntimeError):
    """Raised when the wake-word subsystem cannot initialise."""


class WakeWordListener:
    """Background-thread microphone listener for a pretrained wake word.

    Usage::

        listener = WakeWordListener(model_key="hey_jarvis", threshold=0.5)
        listener.start()
        ...
        # in the main loop, once per frame:
        if listener.consume_wake():
            interpreter.activate(now_ns)
        ...
        listener.stop()

    The listener is inert until :meth:`start` is called and stops
    cleanly on :meth:`stop`. It owns its audio stream and thread.
    """

    def __init__(
        self,
        *,
        model_key: str = DEFAULT_WAKE_MODEL,
        threshold: float = 0.5,
        cooldown_s: float = 2.0,
        on_detect: Callable[[], None] | None = None,
    ) -> None:
        self.model_key = model_key
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._on_detect = on_detect

        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._model = None  # lazily constructed inside the thread
        self._last_fire_ns = 0
        self.detections = 0

    @property
    def phrase(self) -> str:
        """Human-readable wake phrase for the configured model."""
        return WAKE_PHRASES.get(self.model_key, self.model_key)

    # ----- lifecycle -----

    def start(self) -> None:
        """Construct the model, open the mic, and spawn the audio thread.

        Model construction + mic open happen here (on the caller's
        thread) so failures surface synchronously with a clear error,
        rather than dying silently inside the background thread.
        """
        if self._thread is not None:
            return

        # Build the model + verify the mic up-front so errors are loud.
        self._model = self._build_model()
        self._verify_microphone()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="sigil-wakeword",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "wakeword_listener_started",
            model=self.model_key,
            phrase=self.phrase,
            threshold=self.threshold,
        )

    def stop(self) -> None:
        """Signal the audio thread to exit and join it. Idempotent."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        log.info("wakeword_listener_stopped", detections=self.detections)

    def consume_wake(self) -> bool:
        """Return True exactly once per detected wake event, then reset.

        Called by the daemon's main loop once per frame. Returns True
        if a wake word fired since the last call; clears the flag so
        the same utterance doesn't activate repeatedly.
        """
        if self._wake_event.is_set():
            self._wake_event.clear()
            return True
        return False

    # ----- internal -----

    def _build_model(self):
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as exc:
            raise WakeWordError(
                "Wake word needs openwakeword. Install with:\n"
                "    pip install openwakeword sounddevice",
            ) from exc

        # Download pretrained models on first use (no-op if cached).
        try:
            openwakeword.utils.download_models()
        except Exception as exc:  # noqa: BLE001
            log.warning("wakeword_model_download_failed", error=str(exc))

        try:
            model = Model(
                wakeword_models=[self.model_key],
                inference_framework="onnx",  # Windows default; no tflite wheels
            )
        except Exception as exc:  # noqa: BLE001
            raise WakeWordError(
                f"Failed to load wake-word model {self.model_key!r}: {exc}. "
                f"Available pretrained: alexa, hey_jarvis, hey_mycroft, "
                f"timer, weather.",
            ) from exc
        return model

    def _verify_microphone(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise WakeWordError(
                "Wake word needs sounddevice for mic capture. Install with:\n"
                "    pip install sounddevice",
            ) from exc
        try:
            # Querying default input raises if there's no input device.
            sd.check_input_settings(
                samplerate=_SAMPLE_RATE, channels=_CHANNELS, dtype="int16",
            )
        except Exception as exc:  # noqa: BLE001
            raise WakeWordError(
                f"No usable microphone for wake word: {exc}. "
                f"Check that a mic is connected and not held by another app.",
            ) from exc

    def _run_loop(self) -> None:
        """Audio capture + inference loop. Runs on the background thread."""
        import sounddevice as sd

        try:
            stream = sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=_CHANNELS,
                dtype="int16",
                blocksize=_CHUNK,
            )
            stream.start()
        except Exception as exc:  # noqa: BLE001
            log.error("wakeword_stream_open_failed", error=str(exc))
            return

        log.info("wakeword_audio_loop_running")
        try:
            while not self._stop_event.is_set():
                try:
                    audio, _overflowed = stream.read(_CHUNK)
                except Exception as exc:  # noqa: BLE001
                    log.warning("wakeword_audio_read_failed", error=str(exc))
                    time.sleep(0.05)
                    continue

                # (frames, channels) int16 → flat (frames,) int16.
                samples = audio.reshape(-1)

                try:
                    scores = self._model.predict(samples)  # type: ignore[union-attr]
                except Exception as exc:  # noqa: BLE001
                    log.warning("wakeword_predict_failed", error=str(exc))
                    continue

                score = float(scores.get(self.model_key, 0.0))
                if score >= self.threshold:
                    self._handle_detection(score)
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
            log.info("wakeword_audio_loop_exited")

    def _handle_detection(self, score: float) -> None:
        now_ns = time.monotonic_ns()
        if now_ns - self._last_fire_ns < self.cooldown_s * 1e9:
            return  # within cooldown — same utterance, ignore
        self._last_fire_ns = now_ns
        self.detections += 1
        self._wake_event.set()
        log.info(
            "wakeword_detected",
            phrase=self.phrase,
            score=f"{score:.3f}",
            count=self.detections,
        )
        if self._on_detect is not None:
            try:
                self._on_detect()
            except Exception as exc:  # noqa: BLE001
                log.warning("wakeword_on_detect_failed", error=str(exc))


__all__ = [
    "DEFAULT_WAKE_MODEL",
    "WAKE_PHRASES",
    "WakeWordError",
    "WakeWordListener",
]
