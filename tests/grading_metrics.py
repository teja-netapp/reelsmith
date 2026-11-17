"""Colour-difference helpers for tests: CIEDE2000 on the 24 ColorChecker patches + a grey ramp."""
import numpy as np

from app import grading as G

CC = np.array([[115,82,68],[194,150,130],[98,122,157],[87,108,67],[133,128,177],[103,189,170],[214,126,44],
  [80,91,166],[193,90,99],[94,60,108],[157,188,64],[224,163,46],[56,61,150],[70,148,73],[175,54,60],[231,199,31],
  [187,86,149],[8,133,161],[243,243,242],[200,200,200],[160,160,160],[122,122,121],[85,85,85],[52,52,52]], float)
RAMP = np.repeat(np.linspace(10, 245, 12)[:, None], 3, 1)
PTS = np.vstack([CC, RAMP]) / 255.0

def lab_raw(rgb):
    a = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = a @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6/29) ** 3, np.cbrt(xyz), xyz / (3 * (6/29) ** 2) + 4/29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)

def de2000(l1, l2):
    L1, a1, b1 = l1.T; L2, a2, b2 = l2.T
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G_ = 0.5 * (1 - np.sqrt(Cb**7 / (Cb**7 + 25**7)))
    a1p, a2p = (1 + G_) * a1, (1 + G_) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360; h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1; dCp = C2p - C1p
    dhp = h2p - h1p; dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where(C1p * C2p == 0, 0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbp = (L1 + L2) / 2; Cbp = (C1p + C2p) / 2
    hbp = np.where(np.abs(h1p - h2p) > 180, (h1p + h2p + 360) / 2, (h1p + h2p) / 2)
    hbp = np.where(C1p * C2p == 0, h1p + h2p, hbp)
    T = 1 - 0.17*np.cos(np.radians(hbp-30)) + 0.24*np.cos(np.radians(2*hbp)) + 0.32*np.cos(np.radians(3*hbp+6)) - 0.2*np.cos(np.radians(4*hbp-63))
    dth = 30 * np.exp(-((hbp - 275) / 25) ** 2)
    Rc = 2 * np.sqrt(Cbp**7 / (Cbp**7 + 25**7))
    Sl = 1 + 0.015 * (Lbp - 50)**2 / np.sqrt(20 + (Lbp - 50)**2); Sc = 1 + 0.045 * Cbp; Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp/Sl)**2 + (dCp/Sc)**2 + (dHp/Sh)**2 + Rt * (dCp/Sc) * (dHp/Sh))



def lab(rgb):
    return lab_raw(np.clip(rgb, 0, 1))


def graded(look, k=1.0):
    return G.sample_cube(G.build_cube(look, k, None, n=33), PTS)
