"""
PHI-RF-CLASSIFIER: Battlefield RF Spectrum Simulator Package
Synthesizes realistic multi-emitter complex I/Q baseband waveforms for HMGCC SIGINT training & inference.
"""
from simulator.spectrum_generator import SpectrumGenerator, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS

__all__ = ["SpectrumGenerator", "CLASSES", "CLASS_TO_IDX", "IDX_TO_CLASS"]
