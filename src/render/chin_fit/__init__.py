"""Chin/jaw fit against the approved Chenger art through the offline candidate renderer.

``objective``  -- reference chains (step-2 master, step-4 receipts), fixed MediaPipe contract on
                  the rendered image, frozen per-view comparison frame, residual vector.
``controls``   -- the 36 handoff controls (central chin bones + mirrored CheekLow) and the
                  persistent length-1 ABMX encoding.
``protection`` -- feature guard measured on the renderer's own skinned head streams.
See ``docs/hs2_chin_fit.md``.
"""
