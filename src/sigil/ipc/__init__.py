"""Inter-worker communication primitives.

Shared-memory ring buffers for frames, ZeroMQ pub/sub for events. Decouples
the workers so any one of them can stall or restart without blocking others.

Phase 5 deliverable; not yet implemented.
"""
