"""Web demo: Engine định giá linh hoạt cho hàng tươi sống (FreshRetailNet-50K).

Chạy: streamlit run app.py
"""
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from engine import BUNDLE_DIR, PricingEngine

st.set_page_config(page_title="Engine định giá linh hoạt", page_icon="🏷️", layout="centered")

BLUE, ORANGE, GREY, LIGHT = "#378ADD", "#D85A30", "#888780", "#C9C8C2"
POL_NAME = {"actual": "Giá thực tế", "none": "Không khuyến mãi", "ml": "Engine ML"}
POL_COLOR = alt.Scale(domain=list(POL_NAME.values()), range=[ORANGE, GREY, BLUE])
WHATIF_LABEL = {"activity_flag": "Sàn đang chạy chương trình khuyến mãi", "holiday_flag": "Ngày lễ",
                "precpt": "Lượng mưa", "avg_temperature": "Nhiệt độ trung bình (°C)"}


@st.cache_resource(show_spinner="Đang nạp mô hình…")
def load_engine():
    return PricingEngine()


@st.cache_data
def load_table(name, index_col=None):
    path = os.path.join(BUNDLE_DIR, "tables", name + ".csv")
    return pd.read_csv(path, index_col=index_col) if os.path.exists(path) else None


@st.cache_data
def default_key(_e, date):
    """Sản phẩm bán chạy đầu tiên mà với số liệu mặc định engine đề xuất giảm giá (để ví dụ mở đầu dễ hiểu)."""
    j = _e.date_index[date]
    for s in np.argsort(-_e.F[:, j])[:300]:
        f = float(_e.F[s, j])
        if not _e.recommend(s, j, round(1.3 * f, 1), round(0.3 * f, 1))["Đề xuất"].iloc[-1]:
            return str(_e.keys[s])
    return str(_e.keys[int(np.argmax(_e.F[:, j]))])


@st.cache_data
def load_sim():
    return pd.read_parquet(os.path.join(BUNDLE_DIR, "sim_test.parquet"))


def fig(name, caption=None):
    path = os.path.join(BUNDLE_DIR, "figures", name + ".png")
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")


def num(x, d=0):
    """Định dạng số kiểu Việt Nam: 1.234,5"""
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def vnd(x):
    return num(round(x / 100) * 100) + " đ"


E = load_engine()
M = E.meta

st.title("🏷️ Hôm nay nên bán giá bao nhiêu?")
st.caption("Engine định giá cho hàng tươi sống · dữ liệu thật FreshRetailNet-50K · mô hình dự báo Random Forest")

tab1, tab2 = st.tabs(["🎯 Đề xuất giá", "📊 Kết quả nghiên cứu"])

# =====================================================================
# Tab 1 — nhập thông tin, nhận giá đề xuất
# =====================================================================
with tab1:
    st.subheader("① Chọn sản phẩm và ngày")
    ST = E.series_table().sort_values("Nhu cầu TB/ngày", ascending=False)
    c1, c2 = st.columns([1, 2])
    cats = ["Tất cả"] + [int(c) for c in sorted(ST["Danh mục"].unique())]
    cat = c1.selectbox("Danh mục", cats)
    sub = ST if cat == "Tất cả" else ST[ST["Danh mục"] == cat]
    labels = {r.key: f"Sản phẩm {r['Sản phẩm']} – cửa hàng {r['Cửa hàng']} (bán ~{num(r['Nhu cầu TB/ngày'], 1)}/ngày)" for _, r in sub.iterrows()}
    test_dates = [str(d) for d, sp in zip(E.dates, E.split) if sp == "test"]
    opts = list(labels); dk = default_key(E, test_dates[0])
    key = c2.selectbox("Sản phẩm", opts, index=opts.index(dk) if dk in opts else 0, format_func=labels.get)
    THU = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]
    date = st.selectbox("Ngày", test_dates, format_func=lambda d: f"{pd.Timestamp(d):%d/%m/%Y} ({THU[pd.Timestamp(d).dayofweek]})")
    s, j = E.key_index[key], E.date_index[date]
    f0 = float(E.F[s, j])
    st.caption(f"Nếu giữ giá gốc, mô hình dự báo hôm đó bán được khoảng **{num(f0, 1)} đơn vị**.")

    st.subheader("② Nhập tình hình hàng hôm nay")
    k = f"{key}_{date}"   # đổi sản phẩm/ngày thì các ô nhập trở về giá trị gợi ý
    with st.form("inputs"):
        i1, i2, i3 = st.columns(3)
        price = i1.number_input("Giá gốc (đồng)", min_value=1000, value=20000, step=1000, key="price")
        stock = i2.number_input("Số hàng đang có", min_value=0.0, value=round(1.3 * f0, 1), step=0.5, key="stock" + k)
        expiring = i3.number_input("Trong đó hết hạn cuối ngày", min_value=0.0, value=round(0.3 * f0, 1), step=0.5, key="exp" + k,
                                   help="Phần hàng cũ, không bán hết hôm nay thì phải bỏ.")
        overrides = {}
        with st.expander("Tùy chọn thêm"):
            margin = st.slider("Biên lợi nhuận ở giá gốc (%)", 10, 50, int(M["MARGIN"] * 100), 5, key="margin",
                               help="Giá vốn = giá gốc × (1 − biên lợi nhuận). Mặc định 30%.") / 100
            st.caption("Bối cảnh ngày đó (mặc định là số liệu thật, có thể đổi để thử):")
            w1, w2 = st.columns(2)
            for idx_, name in enumerate(M.get("WHATIF", [])):
                col = w1 if idx_ % 2 == 0 else w2
                cur = E.feature_value(s, j, name)
                if name.endswith("_flag"):
                    overrides[name] = float(col.checkbox(WHATIF_LABEL.get(name, name), value=cur > 0.5, key=name + k))
                else:
                    lo, hi = E.feature_range(name)
                    overrides[name] = col.slider(WHATIF_LABEL.get(name, name), float(np.floor(lo)), float(np.ceil(hi)),
                                                 float(round(cur, 1)), 0.1, key=name + k)
        st.form_submit_button("Tính giá đề xuất", type="primary", width="stretch")

    expiring = min(expiring, stock)
    R = E.recommend(s, j, stock, expiring, margin, overrides)
    R["Giá bán"] = R["Mức giá"] / 100 * price
    R["Tiền lời"] = R["Lợi nhuận dự kiến"] * price
    b, base = R[R["Đề xuất"]].iloc[0], R.iloc[-1]

    st.subheader("③ Kết quả")
    r100 = lambda x: round(x / 100) * 100
    gain = r100(b["Tiền lời"]) - r100(base["Tiền lời"])   # khớp với số hiển thị trong bảng
    head = f"### 👉 Nên bán **{int(b['Mức giá'])}% giá gốc** = **{vnd(b['Giá bán'])}**"
    if int(b["Mức giá"]) == 100:
        body = (f"Tiền lời dự kiến **{vnd(b['Tiền lời'])}**. Giảm giá có làm khách mua thêm, nhưng không đủ bù phần giá bị giảm, "
                "nên giữ giá gốc là lời nhất.")
    else:
        body = (f"Tiền lời dự kiến **{vnd(b['Tiền lời'])}**, nhiều hơn giữ giá gốc **{vnd(gain)}**. "
                f"Giảm giá giúp bán thêm {num(b['Lượng bán dự kiến'] - base['Lượng bán dự kiến'], 1)} đơn vị"
                + (f" và bớt {num(base['Hàng hết hạn không bán được'] - b['Hàng hết hạn không bán được'], 1)} đơn vị hàng phải bỏ."
                   if base["Hàng hết hạn không bán được"] > b["Hàng hết hạn không bán được"] + 1e-9 else "."))
    st.success(head + "\n\n" + body)

    st.markdown("**Nếu bán ở từng mức giá:**")
    T = pd.DataFrame({
        "": np.where(R["Đề xuất"], "✅", ""),
        "Mức giá": R["Mức giá"].astype(str) + "%",
        "Giá bán": R["Giá bán"].map(vnd),
        "Bán được (dự kiến)": R["Lượng bán dự kiến"].map(lambda x: num(x, 1)),
        "Hàng phải bỏ": R["Hàng hết hạn không bán được"].map(lambda x: num(x, 1)),
        "Tiền lời dự kiến": R["Tiền lời"].map(vnd),
    })
    st.dataframe(T, hide_index=True, width="stretch")

    R["Mức"] = R["Mức giá"].astype(str) + "%"
    R["Chọn"] = np.where(R["Đề xuất"], "Đề xuất", "Khác")
    chart = alt.Chart(R).mark_bar().encode(
        x=alt.X("Mức:N", sort=list(R["Mức"]), title="Mức giá (so với giá gốc)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Tiền lời:Q", title="Tiền lời dự kiến (đ)"),
        color=alt.Color("Chọn:N", scale=alt.Scale(domain=["Đề xuất", "Khác"], range=[BLUE, LIGHT]), legend=None),
        tooltip=[alt.Tooltip("Mức:N", title="Mức giá"), alt.Tooltip("Tiền lời:Q", title="Tiền lời (đ)", format=",.0f")])
    st.altair_chart(chart.properties(height=240), width="stretch")

    with st.expander("Engine tính như thế nào?"):
        st.markdown(f"""
1. Mô hình **Random Forest** (học từ dữ liệu bán hàng thật) dự báo khách sẽ mua bao nhiêu ở 6 mức giá: 70%, 80%, 85%, 90%, 95%, 100% giá gốc.
2. Ở mỗi mức: **bán được** = min(nhu cầu dự báo, số hàng đang có); **hàng phải bỏ** = phần hàng hết hạn cuối ngày không bán kịp.
3. **Tiền lời** = (giá bán − giá vốn) × bán được − giá vốn × hàng phải bỏ, với giá vốn = {int((1 - margin) * 100)}% giá gốc.
4. Chọn mức có tiền lời cao nhất.

*Đơn vị hàng theo thang đã chuẩn hóa của bộ dữ liệu. Hôm đó cửa hàng thật bán ở {num(E.PACT[s, j] * 100)}% giá gốc.*
""")

# =====================================================================
# Tab 2 — kết quả nghiên cứu (tóm tắt + chi tiết thu gọn)
# =====================================================================
with tab2:
    T2 = load_table("bang_e2_so_sanh_chinh_sach", index_col=0)
    G = "Lợi nhuận tăng thêm (% doanh thu)"
    st.markdown(f"Chạy lại **22 ngày** (11/6–2/7/2024) cho **{num(M['S_N'])} sản phẩm–cửa hàng** trên cửa hàng mô phỏng, "
                "so sánh 3 cách đặt giá:")
    if T2 is not None:
        k1, k2, k3 = st.columns(3)
        k1.metric("Giá thực tế của cửa hàng", f"KM {num(T2.loc['Giá thực tế', 'Tỷ lệ ngày khuyến mãi (%)'])}% số ngày",
                  f"giảm TB {num(T2.loc['Giá thực tế', 'Chiết khấu TB (%)'], 1)}%", delta_color="off")
        k2.metric("Không khuyến mãi", f"+{num(T2.loc['Không khuyến mãi', G], 1)}% doanh thu", "lợi nhuận tăng thêm")
        k3.metric("Engine ML", f"+{num(T2.loc['Engine ML', G], 1)}% doanh thu", "lợi nhuận tăng thêm")
    st.info("**Kết luận:** cửa hàng đang giảm giá quá nhiều và quá sâu. Chỉ cần giảm ít và nông hơn là lợi nhuận tăng khoảng 10% doanh thu. "
            "Engine định giá linh hoạt phát huy rõ nhất khi hàng sắp hỏng hoặc nhu cầu giảm đột ngột.")
    fig("e3_so_sanh_chinh_sach")

    with st.expander("📅 Xem sản phẩm đang chọn qua 22 ngày"):
        d = load_sim()
        d = d[d.s == s].copy()
        d["Cách đặt giá"] = d.policy.map(POL_NAME)
        d["Giá (% giá gốc)"] = d.p * 100
        st.altair_chart(alt.Chart(d).mark_line(point=True, interpolate="step-after").encode(
            x=alt.X("date:T", title="Ngày"), y=alt.Y("Giá (% giá gốc):Q", scale=alt.Scale(domain=[65, 102])),
            color=alt.Color("Cách đặt giá:N", scale=POL_COLOR, legend=alt.Legend(orient="bottom"))).properties(height=240), width="stretch")
        agg = d.groupby("Cách đặt giá").agg(**{"Tổng lợi nhuận": ("profit", "sum"), "Hàng phải bỏ": ("waste", "sum"),
                                               "Số ngày khuyến mãi": ("p", lambda x: int((x < 0.99).sum()))})
        agg["Tổng tiền lời (đ)"] = (agg.pop("Tổng lợi nhuận") * 20000).map(vnd)
        st.dataframe(agg.reindex(list(POL_NAME.values())), width="stretch")
        st.caption("Tiền lời quy đổi với giá gốc 20.000 đ.")

    with st.expander("🛡️ Kiểm tra độ vững"):
        T91, T92 = load_table("bang_e3_danh_gia_cheo", index_col=0), load_table("bang_e4_do_nhay", index_col=0)
        if T91 is not None:
            st.markdown("**Chấm lại bằng một mô hình khác** (độ co giãn từ hồi quy) — lợi nhuận tăng thêm so với giá thực tế, % doanh thu:")
            st.dataframe(T91, width="stretch")
        if T92 is not None:
            st.markdown("**Khi giả định thay đổi:**")
            st.dataframe(T92, width="stretch")

    with st.expander("🗂️ Theo danh mục sản phẩm"):
        fig("e7_theo_danh_muc")
        T10 = load_table("bang_e5_theo_danh_muc")
        if T10 is not None:
            st.dataframe(T10, hide_index=True, width="stretch")

    with st.expander("ℹ️ Phương pháp và nguồn dữ liệu"):
        st.markdown(f"""
- **Dữ liệu:** [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (Dingdong, 28/3–2/7/2024), {num(M['S_N'])} sản phẩm–cửa hàng.
- **Dự báo:** so sánh 5 mô hình, chọn **{M['MODEL']}** vì vừa chính xác vừa phản ứng với giá sát ước lượng hồi quy (độ co giãn {num(M['ELAS'], 2)}).
- **Engine:** dự báo nhu cầu ở 6 mức giá, chọn mức lời nhất (*forecast-then-optimize*).
- **Đánh giá:** cửa hàng mô phỏng trên nhu cầu thật (nhập hàng base-stock, hạn dùng {M['SHELF']} ngày); đây là mô phỏng trên dữ liệu lịch sử, không phải thử nghiệm A/B.

**Tài liệu chính:** Cohen et al. (2017), *Operations Research* 65(2); Hua et al. (2021), KDD; Haijema et al. (2026), *IJPE* 292; Birr et al. (2026), arXiv:2606.13741; Wang et al. (2025), arXiv:2505.16319.

Mã nguồn: [github.com/phuongvi-138/dynamic-pricing-engine](https://github.com/phuongvi-138/dynamic-pricing-engine)
""")
