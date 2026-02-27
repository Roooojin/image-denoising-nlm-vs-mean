import numpy as np

from src.denoise_nlm.metrics import mse_u8, psnr_u8


def test_mse_zero():
    a = np.zeros((4, 4), dtype=np.uint8)
    assert mse_u8(a, a) == 0.0


def test_psnr_inf_on_identical():
    a = np.zeros((4, 4), dtype=np.uint8)
    assert psnr_u8(a, a) == float("inf")


def test_mse_simple():
    a = np.zeros((2, 2), dtype=np.uint8)
    b = np.ones((2, 2), dtype=np.uint8) * 10
    assert abs(mse_u8(a, b) - 100.0) < 1e-6
