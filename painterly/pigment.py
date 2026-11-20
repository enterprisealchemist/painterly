"""Pigment-like colour mixing: blue and yellow make green, not grey.

Each colour is turned into a 10-band reflectance spectrum, the spectra are mixed by a
weighted geometric mean (which behaves like subtractive paint mixing), and the result is
turned back into RGB. Mixing happens in linear light.

The spectral tables and the method are from libmypaint (helpers.c: rgb_to_spectral,
spectral_to_rgb, mix_colors), used under its licence:

    libmypaint - The MyPaint Brush Library
    Copyright (C) 2008-2011 Martin Renold and contributors

    Permission to use, copy, modify, and/or distribute this software for any
    purpose with or without fee is hereby granted, provided that the above
    copyright notice and this permission notice appear in all copies.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""

from __future__ import annotations

import numpy as np

_EPS = 0.001  # WGM_EPSILON in libmypaint

_T = np.array([
    [0.026595621243689, 0.049779426257903, 0.022449850859496, -0.218453689278271, -0.256894883201278,
     0.445881722194840, 0.772365886289756, 0.194498761382537, 0.014038157587820, 0.007687264480513],
    [-0.032601672674412, -0.061021043498478, -0.052490001018404, 0.206659098273522, 0.572496335158169,
     0.317837248815438, -0.021216624031211, -0.019387668756117, -0.001521339050858, -0.000835181622534],
    [0.339475473216284, 0.635401374177222, 0.771520797089589, 0.113222640692379, -0.055251113343776,
     -0.048222578468680, -0.012966666339586, -0.001523814504223, -0.000094718948810, -0.000051604594741],
], dtype=np.float32)  # spectrum (10) -> RGB

_BASIS = np.array([
    [0.009281362787953, 0.009732627042016, 0.011254252737167, 0.015105578649573, 0.024797924177217,
     0.083622585502406, 0.977865045723212, 1.000000000000000, 0.999961046144372, 0.999999992756822],
    [0.002854127435775, 0.003917589679914, 0.012132151699187, 0.748259205918013, 1.000000000000000,
     0.865695937531795, 0.037477469241101, 0.022816789725717, 0.021747419446456, 0.021384940572308],
    [0.537052150373386, 0.546646402401469, 0.575501819073983, 0.258778829633924, 0.041709923751716,
     0.012662638828324, 0.007485593127390, 0.006766900622462, 0.006699764779016, 0.006676219883241],
], dtype=np.float32)  # RGB -> spectrum (10)


def to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)


def _log_spectrum(rgb_lin: np.ndarray) -> np.ndarray:
    c = rgb_lin * (1.0 - _EPS) + _EPS
    return np.log(c @ _BASIS)  # (..., 10)


def mix(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    """Mix sRGB colours `a` and `b` like paint, `t` (0..1, broadcastable) being the share of `b`.

    `a`, `b` are (..., 3) arrays in 0..1; `t` has shape (...) or (..., 1).
    """
    t = np.asarray(t, dtype=np.float32)
    if t.ndim and t.shape[-1] != 1:
        t = t[..., None]
    la, lb = _log_spectrum(to_linear(a)), _log_spectrum(to_linear(b))
    spec = np.exp(la * (1.0 - t) + lb * t)
    rgb = spec @ _T.T
    rgb = (rgb - _EPS) / (1.0 - _EPS)
    return to_srgb(rgb)
