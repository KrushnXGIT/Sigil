from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sigil.perception.landmarks import HandLandmarkerWrapper
from sigil.perception.normalize import normalize_landmarks

MODEL_PATH = Path("models/static-classifier/run-001/best.onnx")
META_PATH = Path("models/static-classifier/run-001/best.meta.json")

CONFIDENCE_THRESHOLD = 0.70


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / exp.sum()


def main() -> int:
    metadata = json.loads(META_PATH.read_text())
    label_map = metadata["label_map"]

    sess = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    with HandLandmarkerWrapper(num_hands=1) as landmarker:
        print("Press 'q' to quit")

        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            captured = type(
                "CapturedFrame",
                (),
                {
                    "pixels": rgb,
                    "timestamp_ns": time.time_ns(),
                    "frame_index": frame_index,
                    "shape": rgb.shape[:2],
                },
            )()

            result = landmarker.process(captured)

            label = "no_hand"
            confidence = 0.0

            if result.hands:
                hand = result.hands[0]

                normalized = normalize_landmarks(hand.keypoints.astype(np.float32))

                inp = normalized[None, :, :].astype(np.float32)

                logits = sess.run(
                    ["logits"],
                    {"landmarks": inp},
                )[0]

                probs = softmax(logits[0])

                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])

                if confidence >= CONFIDENCE_THRESHOLD:
                    label = label_map[str(pred_idx)]
                else:
                    label = "uncertain"

            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Sigil ONNX Demo", frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
