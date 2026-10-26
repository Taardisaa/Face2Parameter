"""Differentiable (PyTorch) HS2 head geometry chain -- see docs/hs2_diffgeo.md.

card + bridge patch (ABMX rows, native59)
  -> ``forward.TorchHeadForward``   CapturedHeadForward.evaluate mirror (card-load / captured
                                    baselines, ShapeHeadInfoFemale ``native_shape``, ABMX ``abmx``)
  -> EyeLook (hosted, exact, held constant) and FBS blend shapes (probe weights, exact port)
  -> ``unity``                      UnityPlayer.dll hierarchy / palette / CPU skinning, lifted from
                                    the installed binary (``native_generated``)
  -> ``session.DiffGeoSession``     window data, exact reference (``exact_streams``), torch streams
                                    (``torch_streams``), frozen bridge root (``frozen_root``)
  -> ``camera``                     bridge camera at any yaw, target pixels
  -> ``chin_proxy``                 a PROXY of the chin objective (not the authoritative score)

Arithmetic modes (``fp.Arith``): 'exact' reproduces the exact chain's float32 bytes; 'f64' is the
same graph without rounding (finite-difference gradient checks).
"""
