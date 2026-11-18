import cv2
import numpy as np

from porteiro.quality import detect_borders, detect_cutout, detect_fill, noise_sigma, safe_to_crop


def _textured(h=200, w=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(130, 30, (h, w)), 0, 255).astype(np.uint8)


def test_letterbox_detected_and_safe_to_crop():
    img = _textured()
    img[:30] = 0
    img[-20:] = 0
    b = detect_borders(img)
    assert abs(b["border_top"] - 30 / 200) < 1e-6
    assert abs(b["border_bottom"] - 20 / 200) < 1e-6
    assert b["border_left"] == 0 and not b["colagem"]
    assert safe_to_crop(b, [(50, 40, 100, 150)])
    assert not safe_to_crop(b, [(50, 5, 100, 150)])  # anotação dentro da borda


def test_gradual_dark_is_not_border():
    img = _textured()
    ramp = np.linspace(0, 130, 40)[:, None]
    img[:40] = np.clip(ramp + np.random.default_rng(1).normal(0, 1, (40, 300)), 0, 255).astype(np.uint8)
    b = detect_borders(img)
    assert b["border_top"] == 0


def test_collage_band_in_middle():
    img = _textured()
    img[:, 140:150] = 0
    assert detect_borders(img)["colagem"]


def test_rotation_fill_detected_but_not_on_plain_texture():
    img = _textured(320, 320)
    M = cv2.getRotationMatrix2D((160, 160), 25, 1.0)
    rot = cv2.warpAffine(img, M, (320, 320), borderValue=0)
    assert detect_fill(rot)["fill_frac"] > 0.05
    assert detect_fill(img)["fill_frac"] < 0.01


def test_cutout_detected():
    img = _textured(320, 320)
    img[100:150, 80:160] = 255
    c = detect_cutout(img)
    assert c["cutout_n"] == 1 and abs(c["cutout_frac"] - 50 * 80 / 320 ** 2) < 1e-3


def test_noise_sigma_tracks_noise_level():
    flat = np.full((200, 200), 120, np.uint8)
    rng = np.random.default_rng(0)
    noisy = np.clip(120 + rng.normal(0, 10, flat.shape), 0, 255).astype(np.uint8)
    assert noise_sigma(flat) < 0.5 and 7 < noise_sigma(noisy) < 13
