"""Lõi engine định giá: nạp mô hình + dữ liệu, dự báo nhu cầu ở 6 mức giá và chọn mức giá lời nhất.

Logic giống hệt Notebook 2 (Bước 8):
  lợi nhuận dự kiến = (giá − giá vốn) × min(nhu cầu dự báo, tồn kho) − giá vốn × max(0, hàng sắp hết hạn − nhu cầu dự báo)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

BUNDLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_bundle")


class PricingEngine:
    def __init__(self, bundle_dir: str = BUNDLE_DIR):
        self.dir = bundle_dir
        self.meta = json.load(open(os.path.join(bundle_dir, "meta.json"), encoding="utf-8"))
        self.model = joblib.load(os.path.join(bundle_dir, "model.joblib"))
        d = np.load(os.path.join(bundle_dir, "data.npz"), allow_pickle=False)
        self.X, self.LDL1, self.QH = d["X"], d["LDL1"], d["QH"]
        self.PACT, self.REC, self.F = d["PACT"], d["REC"], d["F"]
        self.keys, self.cat, self.store, self.prod = d["KEYS"], d["CAT"], d["STORE"], d["PROD"]
        self.dates, self.split = d["DATES"], d["SPLIT"]
        self.feats = self.meta["FEATS"]
        self.LV = np.array(self.meta["LV"])
        self.cal = float(self.meta["CAL"])
        self.i_ld, self.i_dc = self.feats.index("log_disc"), self.feats.index("disc_chg")
        self.key_index = {k: i for i, k in enumerate(self.keys)}
        self.date_index = {dt: i for i, dt in enumerate(self.dates)}

    # ---------- tra cứu ----------
    def series_table(self) -> pd.DataFrame:
        return pd.DataFrame({"key": self.keys, "Danh mục": self.cat, "Cửa hàng": self.store, "Sản phẩm": self.prod,
                             "Nhu cầu TB/ngày": self.F.mean(1)})

    def feature_value(self, s: int, j: int, name: str) -> float:
        return float(self.X[s, j, self.feats.index(name)])

    def feature_range(self, name: str):
        v = self.X[:, :, self.feats.index(name)]
        return float(np.nanmin(v)), float(np.nanmax(v))

    # ---------- dự báo ----------
    def forecast(self, s: int, j: int, overrides: dict | None = None) -> np.ndarray:
        """Nhu cầu dự báo của chuỗi s, ngày j ở từng mức giá (chạy Random Forest trực tiếp)."""
        x = np.repeat(self.X[s, j][None, :].astype(np.float64), len(self.LV), axis=0)
        for name, val in (overrides or {}).items():
            x[:, self.feats.index(name)] = val
        x[:, self.i_ld] = np.log(self.LV)
        x[:, self.i_dc] = np.log(self.LV) - np.nan_to_num(self.LDL1[s, j], nan=0.0)
        return np.clip(np.expm1(self.model.predict(x)), 0, None) * self.cal

    # ---------- ra quyết định ----------
    def recommend(self, s: int, j: int, stock: float, expiring: float = 0.0, margin: float = 0.30,
                  overrides: dict | None = None) -> pd.DataFrame:
        c = 1 - margin
        q = self.forecast(s, j, overrides)
        sales = np.minimum(q, stock)
        waste = np.maximum(0.0, expiring - q)
        profit = (self.LV - c) * sales - c * waste
        out = pd.DataFrame({"Mức giá": (self.LV * 100).round().astype(int), "Nhu cầu dự báo": q,
                            "Lượng bán dự kiến": sales, "Hàng hết hạn không bán được": waste,
                            "Doanh thu dự kiến": self.LV * sales, "Lợi nhuận dự kiến": profit})
        out["Đề xuất"] = np.arange(len(self.LV)) == int(np.argmax(profit))
        return out


class PrecomputedEngine(PricingEngine):
    """Bản toàn bộ chuỗi (8179): mô hình Random Forest quá lớn để nạp vào web, nên nhu cầu dự báo ở 6 mức giá
    đã được CHÍNH mô hình đó tính sẵn trong Notebook 2 (Bước 11) cho mọi sản phẩm–ngày và các kịch bản
    bật/tắt khuyến mãi của sàn, ngày lễ. Phần chọn giá (recommend) giữ nguyên, chạy trực tiếp theo tồn kho nhập vào."""

    def __init__(self, bundle_dir: str):
        self.dir = bundle_dir
        self.meta = json.load(open(os.path.join(bundle_dir, "meta.json"), encoding="utf-8"))
        d = np.load(os.path.join(bundle_dir, "data.npz"), allow_pickle=False)
        self.QS, self.QH, self.FLAGV = d["QS"], d["QH"], d["FLAGV"]
        self.PACT, self.REC, self.F = d["PACT"], d["REC"], d["F"]
        self.keys, self.cat, self.store, self.prod = d["KEYS"], d["CAT"], d["STORE"], d["PROD"]
        self.dates, self.split = d["DATES"], d["SPLIT"]
        self.flags = list(self.meta["FLAGS"])
        self.combos = [tuple(c) for c in self.meta["COMBOS"]]
        self.LV = np.array(self.meta["LV"])
        self.key_index = {k: i for i, k in enumerate(self.keys)}
        self.date_index = {dt: i for i, dt in enumerate(self.dates)}

    def feature_value(self, s: int, j: int, name: str) -> float:
        return float(self.FLAGV[self.flags.index(name), s, j])

    def feature_range(self, name: str):
        return 0.0, 1.0

    def forecast(self, s: int, j: int, overrides: dict | None = None) -> np.ndarray:
        cur = tuple(int(round(float(self.FLAGV[i, s, j]))) for i in range(len(self.flags)))
        want = tuple(int(round(float((overrides or {}).get(f, cur[i])))) for i, f in enumerate(self.flags))
        if want == cur:
            return self.QH[s, j].astype(np.float64)
        return self.QS[self.combos.index(want), s, j].astype(np.float64)
