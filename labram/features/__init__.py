"""Handcrafted EEG features (time, spectral, wavelet, entropy, covariance).

``extractors`` holds dataset-agnostic per-window feature functions; ``tuab``
applies them to the TUAB per-recording ``.npy`` export. Driven by
``notebooks/eeg_feature_extraction_tuab.ipynb``.
"""
