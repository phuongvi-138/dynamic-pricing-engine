# Bước 11 (bản 8179): đóng gói cho web app dạng "dự báo tính sẵn" và đẩy lên GitHub.
# Mô hình Random Forest bản toàn bộ chuỗi quá lớn để nạp vào web (Streamlit Cloud ~1 GB RAM),
# nên ở đây chạy chính mô hình đó TRƯỚC cho mọi sản phẩm–ngày (validation + test) ở 6 mức giá,
# kèm các kịch bản bật/tắt khuyến mãi của sàn và ngày lễ. Web chỉ tra các con số này rồi
# tính tiền lời và chọn giá ngay khi người dùng nhập tồn kho (logic giống hệt Bước 8).
import os, json, shutil, subprocess, glob, platform, itertools, time
import numpy as np, pandas as pd

APP_DIR = "app_bundle_8179"
BUNDLE = os.path.join(OUT, APP_DIR)
shutil.rmtree(BUNDLE, ignore_errors=True)
os.makedirs(os.path.join(BUNDLE, "tables"), exist_ok=True); os.makedirs(os.path.join(BUNDLE, "figures"), exist_ok=True)

# 1) Dự báo tính sẵn: kịch bản theo cờ khuyến mãi của sàn / ngày lễ
t0 = time.time()
FLAGS = [c for c in ["activity_flag", "holiday_flag"] if c in FEATS]
COMBOS = list(itertools.product([0, 1], repeat=len(FLAGS)))
QS = np.zeros((len(COMBOS), S_N, len(DAYS), A), dtype=np.float32)
for jj, t in enumerate(DAYS):
    rows = np.arange(S_N) * T_N + t
    base = XR[rows].copy(); ldl = np.nan_to_num(LDL1[rows], nan=0.0)
    for ci, combo in enumerate(COMBOS):
        X = base.copy()
        for f, v in zip(FLAGS, combo): X[:, FEATS.index(f)] = v
        Xs = np.repeat(X[None], A, axis=0)                        # (mức giá, chuỗi, đặc trưng)
        Xs[:, :, I_LD] = np.log(LV)[:, None]; Xs[:, :, I_DC] = np.log(LV)[:, None] - ldl[None, :]
        pred = np.clip(np.expm1(MODEL.predict(Xs.reshape(-1, X.shape[1]))), 0, None) * CAL
        QS[ci, :, jj, :] = pred.reshape(A, S_N).T
idx = np.arange(S_N)[:, None] * T_N + DAYS[None, :]
FLAGV = np.stack([XR[idx][:, :, FEATS.index(f)] for f in FLAGS]).astype(np.float32) if FLAGS else np.zeros((0, S_N, len(DAYS)), np.float32)
print(f"Tính sẵn dự báo {len(COMBOS)} kịch bản × {S_N:,} chuỗi × {len(DAYS)} ngày × {A} mức giá xong sau {time.time()-t0:.0f}s")

STORE = P["store_id"].values.reshape(S_N, T_N)[:, 0]; PROD = P["product_id"].values.reshape(S_N, T_N)[:, 0]
np.savez_compressed(os.path.join(BUNDLE, "data.npz"),
    QS=QS, QH=QH[:, DAYS, :].astype(np.float32), FLAGV=FLAGV,
    PACT=PACT[:, DAYS].astype(np.float32), REC=REC[:, DAYS].astype(np.float32), F=F[:, DAYS].astype(np.float32),
    KEYS=KEYS.astype(str), CAT=CAT, STORE=STORE.astype(int), PROD=PROD.astype(int),
    DATES=np.array([str(d.date()) for d in DATES[DAYS]]), SPLIT=np.where(DAYS >= T_TEST0, "test", "validation"))

# 2) Nhật ký mô phỏng trên test
logs = []
for k in POL:
    lg_ = RES[k]["log"].copy(); lg_["policy"] = k; logs.append(lg_)
sim = pd.concat(logs); sim["date"] = DATES[sim.t.values].strftime("%Y-%m-%d")
sim.drop(columns="t").astype({"s": "int32"}).to_parquet(os.path.join(BUNDLE, "sim_test.parquet"), index=False)

# 3) Bảng, hình, mô tả
for f in glob.glob(os.path.join(OUT2, "*.csv")): shutil.copy(f, os.path.join(BUNDLE, "tables"))
for f in glob.glob(os.path.join(OUT2, "*.png")): shutil.copy(f, os.path.join(BUNDLE, "figures"))
import sklearn
meta_app = dict(MODE="precomputed", FLAGS=FLAGS, COMBOS=[list(c) for c in COMBOS], WHATIF=FLAGS, LV=LV.tolist(),
                MARGIN=CFG2["MARGIN"], SHELF=CFG2["SHELF"], K=float(K), ELAS=ELAS, MODEL=SEL, S_N=int(S_N), N_DAYS=int(len(DAYS)),
                CITY=META["CFG"].get("CITY", 0), NOTES=NOTES,
                VERSIONS={"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__})
json.dump(meta_app, open(os.path.join(BUNDLE, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
sizes = {os.path.relpath(f, BUNDLE): os.path.getsize(f) / 1e6 for f in glob.glob(os.path.join(BUNDLE, "**", "*"), recursive=True) if os.path.isfile(f)}
print(f"Đã đóng gói {APP_DIR} ({sum(sizes.values()):.1f} MB; tệp lớn nhất {max(sizes.values()):.1f} MB)")
assert max(sizes.values()) < 95, "Có tệp vượt 95 MB"

# 4) Đẩy lên GitHub vào thư mục riêng (không đụng app_bundle/ và app.py của bản 2.003 chuỗi)
PUSH_TO_GITHUB = True
def push_to_github():
    from google.colab import userdata
    try:
        def get(n):
            for x in (n.upper(), n.lower()):
                try:
                    return userdata.get(x).strip()
                except userdata.SecretNotFoundError:
                    continue
            raise userdata.SecretNotFoundError(n)
        token, repo = get("GITHUB_TOKEN"), get("GITHUB_REPO")
    except Exception as e:
        print("Chưa đọc được GITHUB_TOKEN / GITHUB_REPO trong Secrets – bỏ qua bước đẩy lên GitHub.", type(e).__name__); return
    work = "/content/_repo"; shutil.rmtree(work, ignore_errors=True)
    url = f"https://x-access-token:{token}@github.com/{repo}.git"
    def run(*a, cwd=work):
        r = subprocess.run(list(a), cwd=cwd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(("Lỗi git: " + r.stderr).replace(token, "***"))
        return r
    run("git", "clone", "--depth", "1", url, work, cwd="/content")
    shutil.rmtree(os.path.join(work, APP_DIR), ignore_errors=True); shutil.copytree(BUNDLE, os.path.join(work, APP_DIR))
    os.makedirs(os.path.join(work, "notebooks"), exist_ok=True)
    for nb_ in ["Notebook1_Du_bao_cau_8179.ipynb", "Notebook2_Engine_dinh_gia_8179.ipynb"]:
        src = os.path.join(BASE_DIR, nb_)
        if os.path.exists(src): shutil.copy(src, os.path.join(work, "notebooks", nb_))
    run("git", "config", "user.name", "Claude"); run("git", "config", "user.email", "noreply@anthropic.com")
    run("git", "add", "-A")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=work).returncode == 0:
        print("Không có thay đổi để đẩy lên GitHub."); return
    run("git", "commit", "-m", f"Dữ liệu web app bản toàn bộ chuỗi ({S_N} chuỗi, dự báo tính sẵn)\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    run("git", "push", "origin", "HEAD")
    print(f"Đã đẩy {APP_DIR}/ và 2 notebook bản 8179 lên https://github.com/{repo}")
if IN_COLAB and PUSH_TO_GITHUB:
    push_to_github()
