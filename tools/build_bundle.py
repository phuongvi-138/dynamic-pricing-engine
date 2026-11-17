# Bước 11 của Notebook 2: đóng gói mô hình + dữ liệu cho web app và (tuỳ chọn) đẩy lên GitHub.
# Chạy trong cùng runtime với Notebook 2 (dùng các biến MODEL, XR, LDL1, QH, RES, ... đã có).
import os, json, shutil, subprocess, glob, platform
import numpy as np, pandas as pd, joblib, sklearn

BUNDLE = os.path.join(OUT, "app_bundle")
shutil.rmtree(BUNDLE, ignore_errors=True)
os.makedirs(os.path.join(BUNDLE, "tables"), exist_ok=True); os.makedirs(os.path.join(BUNDLE, "figures"), exist_ok=True)

# 1) Mô hình (nén để vừa giới hạn 100 MB/tệp của GitHub)
mpath = os.path.join(BUNDLE, "model.joblib")
joblib.dump(MODEL, mpath, compress=("xz", 3))
msize = os.path.getsize(mpath) / 1e6
print(f"Mô hình {SEL}: {msize:.1f} MB")
assert msize < 95, "Tệp mô hình vượt 95 MB – cần chia nhỏ trước khi đẩy lên GitHub"

# 2) Dữ liệu cho các ngày validation + test
idx = np.arange(S_N)[:, None] * T_N + DAYS[None, :]
X_APP = XR[idx].astype(np.float32)                                   # (chuỗi, ngày, đặc trưng)
STORE = P["store_id"].values.reshape(S_N, T_N)[:, 0]; PROD = P["product_id"].values.reshape(S_N, T_N)[:, 0]
np.savez_compressed(os.path.join(BUNDLE, "data.npz"),
    X=X_APP, LDL1=LDL1[idx].astype(np.float32), QH=QH[:, DAYS, :].astype(np.float32),
    PACT=PACT[:, DAYS].astype(np.float32), REC=REC[:, DAYS].astype(np.float32), F=F[:, DAYS].astype(np.float32),
    KEYS=KEYS.astype(str), CAT=CAT, STORE=STORE.astype(int), PROD=PROD.astype(int),
    DATES=np.array([str(d.date()) for d in DATES[DAYS]]), SPLIT=np.where(DAYS >= T_TEST0, "test", "validation"))

# 3) Nhật ký mô phỏng trên test (để xem từng sản phẩm qua 22 ngày)
logs = []
for k in POL:
    lg_ = RES[k]["log"].copy(); lg_["policy"] = k; logs.append(lg_)
sim = pd.concat(logs); sim["date"] = DATES[sim.t.values].strftime("%Y-%m-%d")
sim.drop(columns="t").astype({"s": "int32"}).to_parquet(os.path.join(BUNDLE, "sim_test.parquet"), index=False)

# 4) Bảng và hình kết quả
for f in glob.glob(os.path.join(OUT2, "*.csv")): shutil.copy(f, os.path.join(BUNDLE, "tables"))
for f in glob.glob(os.path.join(OUT2, "*.png")): shutil.copy(f, os.path.join(BUNDLE, "figures"))

# 5) Thông tin mô tả
WHATIF = [c for c in ["activity_flag", "holiday_flag", "precpt", "avg_temperature"] if c in FEATS]
meta_app = dict(FEATS=FEATS, WHATIF=WHATIF, CAL=float(CAL), LV=LV.tolist(), MARGIN=CFG2["MARGIN"], SHELF=CFG2["SHELF"], K=float(K),
                ELAS=ELAS, MODEL=SEL, S_N=int(S_N), N_DAYS=int(len(DAYS)), CITY=META["CFG"].get("CITY", 0),
                NOTES=NOTES, VERSIONS={"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                                       "pandas": pd.__version__, "joblib": joblib.__version__})
json.dump(meta_app, open(os.path.join(BUNDLE, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
total = sum(os.path.getsize(f) for f in glob.glob(os.path.join(BUNDLE, "**", "*"), recursive=True) if os.path.isfile(f)) / 1e6
print(f"Đã đóng gói app_bundle ({total:.1f} MB):", os.path.relpath(BUNDLE, BASE_DIR))

# 6) Đẩy lên GitHub (chỉ chạy trên Colab khi đã khai báo GITHUB_TOKEN và GITHUB_REPO trong mục Secrets 🔑)
def push_to_github():
    from google.colab import userdata
    try:
        token, repo = userdata.get("GITHUB_TOKEN"), userdata.get("GITHUB_REPO")
    except Exception as e:
        print("Chưa khai báo GITHUB_TOKEN / GITHUB_REPO trong Secrets – bỏ qua bước đẩy lên GitHub.", e); return
    work = "/content/_repo"; shutil.rmtree(work, ignore_errors=True)
    url = f"https://x-access-token:{token}@github.com/{repo}.git"
    def run(*a, cwd=work):
        r = subprocess.run(list(a), cwd=cwd, capture_output=True, text=True)
        if r.returncode != 0:   # không in token ra màn hình
            raise RuntimeError(("Lỗi git: " + r.stderr).replace(token, "***"))
        return r
    run("git", "clone", "--depth", "1", url, work, cwd="/content")
    shutil.rmtree(os.path.join(work, "app_bundle"), ignore_errors=True); shutil.copytree(BUNDLE, os.path.join(work, "app_bundle"))
    os.makedirs(os.path.join(work, "notebooks"), exist_ok=True)
    for nb_ in ["Notebook1_Du_bao_cau.ipynb", "Notebook2_Engine_dinh_gia.ipynb"]:
        src = os.path.join(BASE_DIR, nb_)
        if os.path.exists(src): shutil.copy(src, os.path.join(work, "notebooks", nb_))
    v = meta_app["VERSIONS"]
    open(os.path.join(work, "requirements.txt"), "w").write(
        f"streamlit>=1.50\naltair>=5\npyarrow\nscikit-learn=={v['scikit-learn']}\nnumpy=={v['numpy']}\npandas=={v['pandas']}\njoblib=={v['joblib']}\n")
    run("git", "config", "user.name", "Colab export"); run("git", "config", "user.email", "colab-export@users.noreply.github.com")
    run("git", "add", "-A")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=work).returncode == 0:
        print("Không có thay đổi để đẩy lên GitHub."); return
    run("git", "commit", "-m", f"Cập nhật mô hình, dữ liệu và notebook cho web app ({S_N} chuỗi)")
    run("git", "push", "origin", "HEAD")
    print(f"Đã đẩy app_bundle, notebooks và requirements.txt lên https://github.com/{repo}")
if IN_COLAB:
    push_to_github()
