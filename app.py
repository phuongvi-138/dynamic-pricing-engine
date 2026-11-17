"""Web demo: Engine định giá linh hoạt cho hàng tươi sống (FreshRetailNet-50K).

Chạy: streamlit run app.py
"""
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from engine import BUNDLE_DIR, PricingEngine

st.set_page_config(page_title="Engine định giá linh hoạt", page_icon="🏷️", layout="wide")

BLUE, ORANGE, GREY, LIGHT = "#378ADD", "#D85A30", "#888780", "#C9C8C2"
POL_NAME = {"actual": "Giá thực tế", "none": "Không khuyến mãi", "ml": "Engine ML"}
POL_COLOR = alt.Scale(domain=list(POL_NAME.values()), range=[ORANGE, GREY, BLUE])
WHATIF_LABEL = {"activity_flag": "Có chương trình khuyến mãi của sàn", "holiday_flag": "Ngày lễ",
                "precpt": "Lượng mưa", "avg_temperature": "Nhiệt độ trung bình (°C)"}


@st.cache_resource(show_spinner="Đang nạp mô hình Random Forest…")
def load_engine():
    return PricingEngine()


@st.cache_data
def load_table(name, index_col=None):
    path = os.path.join(BUNDLE_DIR, "tables", name + ".csv")
    return pd.read_csv(path, index_col=index_col) if os.path.exists(path) else None


@st.cache_data
def load_sim():
    return pd.read_parquet(os.path.join(BUNDLE_DIR, "sim_test.parquet"))


def fig(name, caption=None):
    path = os.path.join(BUNDLE_DIR, "figures", name + ".png")
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")


def fmt(x, d=2):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


E = load_engine()
M = E.meta

# ---------------- Thanh bên: chọn sản phẩm và ngày ----------------
st.sidebar.title("🏷️ Engine định giá")
st.sidebar.caption(f"Dữ liệu FreshRetailNet-50K · {M['S_N']:,} sản phẩm–cửa hàng · mô hình {M['MODEL']}".replace(",", "."))
ST = E.series_table().sort_values("Nhu cầu TB/ngày", ascending=False)
cats = sorted(ST["Danh mục"].unique())
cat = st.sidebar.selectbox("Danh mục sản phẩm", ["Tất cả"] + [int(c) for c in cats])
sub = ST if cat == "Tất cả" else ST[ST["Danh mục"] == cat]
labels = {r.key: f"{r.key} · CH {r['Cửa hàng']} · SP {r['Sản phẩm']} · ~{r['Nhu cầu TB/ngày']:.1f}/ngày" for _, r in sub.iterrows()}
key = st.sidebar.selectbox("Sản phẩm–cửa hàng (xếp theo lượng bán)", list(labels), format_func=labels.get)
show_val = st.sidebar.checkbox("Cho chọn cả ngày validation", value=False)
date_opts = [d for d, sp in zip(E.dates, E.split) if show_val or sp == "test"]
date = st.sidebar.selectbox("Ngày", date_opts)
s, j = E.key_index[key], E.date_index[date]
st.sidebar.divider()
st.sidebar.markdown("**Giai đoạn test:** 11/6–2/7/2024  \n**Giá xét:** 70–100% giá gốc  \n**Giá vốn:** 70% giá gốc (mặc định)")

st.title("Engine định giá linh hoạt cho hàng tươi sống")
st.caption("Dự báo nhu cầu bằng Random Forest ở 6 mức giá → chọn mức giá cho lợi nhuận dự kiến cao nhất (forecast-then-optimize).")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Đề xuất giá", "📅 Mô phỏng 22 ngày", "📊 Kết quả toàn hệ thống", "🛡️ Độ vững & danh mục", "ℹ️ Phương pháp"])

# ---------------- Tab 1: Engine đề xuất giá ----------------
with tab1:
    f0 = float(E.F[s, j])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giá cửa hàng đã áp dụng", f"{E.PACT[s, j] * 100:.0f}% giá gốc")
    c2.metric("Nhu cầu thật (ước lượng)", fmt(E.REC[s, j]))
    c3.metric("Dự báo ở giá gốc", fmt(f0))
    c4.metric("Danh mục / Cửa hàng", f"{int(E.cat[s])} / {int(E.store[s])}")

    st.subheader("Tình hình hôm nay")
    i1, i2, i3 = st.columns(3)
    stock = i1.number_input("Tồn kho hiện có", min_value=0.0, value=round(1.3 * f0, 2), step=max(round(f0 / 10, 2), 0.01),
                            help="Cùng đơn vị với nhu cầu. Mặc định = 1,3 × dự báo ở giá gốc (giống cửa hàng mô phỏng).")
    expiring = i2.number_input("Trong đó: hàng sắp hết hạn (hết hạn cuối ngày)", min_value=0.0, max_value=float(stock),
                               value=min(round(0.3 * f0, 2), float(stock)), step=max(round(f0 / 10, 2), 0.01))
    margin = i3.slider("Biên lợi nhuận gộp ở giá gốc (%)", 10, 50, int(M["MARGIN"] * 100), 5) / 100

    overrides = {}
    with st.expander("Giả định thêm – thử thay đổi bối cảnh (what-if)"):
        st.caption("Mặc định lấy đúng giá trị thật của ngày đã chọn. Thay đổi để xem mô hình phản ứng thế nào.")
        w1, w2 = st.columns(2)
        for i, name in enumerate(M.get("WHATIF", [])):
            col = w1 if i % 2 == 0 else w2
            cur = E.feature_value(s, j, name)
            if name.endswith("_flag"):
                overrides[name] = float(col.checkbox(WHATIF_LABEL.get(name, name), value=cur > 0.5))
            else:
                lo, hi = E.feature_range(name)
                overrides[name] = col.slider(WHATIF_LABEL.get(name, name), float(np.floor(lo)), float(np.ceil(hi)), float(round(cur, 1)), 0.1)

    R = E.recommend(s, j, stock, expiring, margin, overrides)
    b = R[R["Đề xuất"]].iloc[0]
    base = R.iloc[-1]
    st.subheader("Đề xuất của engine")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Giá đề xuất", f"{int(b['Mức giá'])}% giá gốc")
    m2.metric("Lợi nhuận dự kiến", fmt(b["Lợi nhuận dự kiến"], 3), f"{fmt(b['Lợi nhuận dự kiến'] - base['Lợi nhuận dự kiến'], 3)} so với giữ giá gốc")
    m3.metric("Lượng bán dự kiến", fmt(b["Lượng bán dự kiến"]), f"{fmt(b['Lượng bán dự kiến'] - base['Lượng bán dự kiến'])} so với giá gốc")
    m4.metric("Hàng hết hạn bỏ đi", fmt(b["Hàng hết hạn không bán được"]))

    if int(b["Mức giá"]) == 100:
        why = ("Giữ **giá gốc** là lời nhất: giảm giá có làm nhu cầu tăng, nhưng phần bán thêm không đủ bù phần giá bị mất"
               + (", và hàng sắp hết hạn vẫn bán hết ở giá gốc." if base["Hàng hết hạn không bán được"] < 1e-9 else "."))
    else:
        why = (f"Giảm còn **{int(b['Mức giá'])}%** giúp bán thêm {fmt(b['Lượng bán dự kiến'] - base['Lượng bán dự kiến'])} đơn vị"
               + (f" và giảm hàng phải hủy từ {fmt(base['Hàng hết hạn không bán được'])} xuống {fmt(b['Hàng hết hạn không bán được'])}" if base["Hàng hết hạn không bán được"] > 1e-9 else "")
               + ". Giảm sâu hơn nữa thì phần bán thêm không còn bù được phần giá mất (hoặc đã hết hàng để bán).")
    st.info(why)

    R["Mức giá (%)"] = R["Mức giá"].astype(str) + "%"
    R["Lựa chọn"] = np.where(R["Đề xuất"], "Đề xuất", "Mức khác")
    order = list(R["Mức giá (%)"])
    g1, g2 = st.columns(2)
    bars = alt.Chart(R).mark_bar(color=GREY).encode(
        x=alt.X("Mức giá (%):N", sort=order, title="Mức giá", axis=alt.Axis(labelAngle=0)), y=alt.Y("Nhu cầu dự báo:Q"),
        tooltip=["Mức giá (%)", alt.Tooltip("Nhu cầu dự báo:Q", format=".2f")])
    rule = alt.Chart(pd.DataFrame({"y": [stock]})).mark_rule(strokeDash=[5, 4], color="black").encode(y=alt.Y("y:Q", title="Nhu cầu dự báo"))
    g1.markdown("**① Random Forest dự báo nhu cầu ở 6 mức giá** (đường nét đứt = tồn kho)")
    g1.altair_chart((bars + rule).properties(height=300), width="stretch")
    pb = alt.Chart(R).mark_bar().encode(
        x=alt.X("Mức giá (%):N", sort=order, title="Mức giá", axis=alt.Axis(labelAngle=0)), y=alt.Y("Lợi nhuận dự kiến:Q"),
        color=alt.Color("Lựa chọn:N", scale=alt.Scale(domain=["Đề xuất", "Mức khác"], range=[BLUE, LIGHT]), legend=None),
        tooltip=["Mức giá (%)", alt.Tooltip("Lợi nhuận dự kiến:Q", format=".3f")])
    g2.markdown("**② Engine chọn mức có lợi nhuận dự kiến cao nhất**")
    g2.altair_chart(pb.properties(height=300), width="stretch")

    show = R[["Mức giá (%)", "Nhu cầu dự báo", "Lượng bán dự kiến", "Hàng hết hạn không bán được", "Doanh thu dự kiến", "Lợi nhuận dự kiến"]].copy()
    show.insert(0, "", np.where(R["Đề xuất"], "◀ đề xuất", ""))
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[2:]}), hide_index=True, width="stretch")
    st.caption(f"Lợi nhuận dự kiến = (giá − giá vốn) × lượng bán dự kiến − giá vốn × hàng sắp hết hạn không bán được. Giá gốc = 1, giá vốn = {1 - margin:.2f}.")

# ---------------- Tab 2: Mô phỏng 22 ngày cho sản phẩm đã chọn ----------------
with tab2:
    st.markdown(f"Cửa hàng mô phỏng chạy lại 22 ngày test cho **{key}** với 3 cách đặt giá. Nhu cầu mỗi ngày dựa trên nhu cầu thật; "
                "nhập hàng theo base-stock (k × dự báo); hàng để được 2 ngày.")
    sim = load_sim()
    d = sim[sim.s == s].copy()
    d["Cách đặt giá"] = d.policy.map(POL_NAME)
    d["Giá (% giá gốc)"] = d.p * 100
    c = alt.Chart(d).mark_line(point=True, interpolate="step-after").encode(
        x=alt.X("date:T", title="Ngày"), y=alt.Y("Giá (% giá gốc):Q", scale=alt.Scale(domain=[65, 102])),
        color=alt.Color("Cách đặt giá:N", scale=POL_COLOR), tooltip=["date:T", "Cách đặt giá", "Giá (% giá gốc)"])
    st.altair_chart(c.properties(height=260, title="Giá theo ngày"), width="stretch")
    pc = alt.Chart(d).mark_bar().encode(
        x=alt.X("yearmonthdate(date):O", title="Ngày", axis=alt.Axis(labelAngle=-45, format="%d/%m")), xOffset="Cách đặt giá:N",
        y=alt.Y("profit:Q", title="Lợi nhuận trong ngày"), color=alt.Color("Cách đặt giá:N", scale=POL_COLOR),
        tooltip=["date:T", "Cách đặt giá", alt.Tooltip("profit:Q", format=".3f")])
    st.altair_chart(pc.properties(height=260, title="Lợi nhuận theo ngày"), width="stretch")
    agg = d.groupby("Cách đặt giá").agg(**{"Tổng lợi nhuận": ("profit", "sum"), "Tổng lượng bán": ("sales", "sum"),
                                           "Hàng hủy": ("waste", "sum"), "Số ngày khuyến mãi": ("p", lambda x: int((x < 0.99).sum()))})
    st.dataframe(agg.reindex(list(POL_NAME.values())).style.format({"Tổng lợi nhuận": "{:.2f}", "Tổng lượng bán": "{:.2f}", "Hàng hủy": "{:.2f}"}),
                 width="stretch")

# ---------------- Tab 3: Kết quả toàn hệ thống ----------------
with tab3:
    T = load_table("bang_e2_so_sanh_chinh_sach", index_col=0)
    G = "Lợi nhuận tăng thêm (% doanh thu)"
    if T is not None:
        k1, k2, k3 = st.columns(3)
        k1.metric("Engine ML: lợi nhuận tăng thêm", f"+{fmt(T.loc['Engine ML', G])}% doanh thu",
                  f"lợi nhuận +{fmt(T.loc['Engine ML', 'Lợi nhuận so với giá thực tế (%)'], 1)}%")
        k2.metric("Không khuyến mãi: lợi nhuận tăng thêm", f"+{fmt(T.loc['Không khuyến mãi', G])}% doanh thu",
                  f"lợi nhuận +{fmt(T.loc['Không khuyến mãi', 'Lợi nhuận so với giá thực tế (%)'], 1)}%")
        k3.metric("Cửa hàng hiện khuyến mãi", f"{fmt(T.loc['Giá thực tế', 'Tỷ lệ ngày khuyến mãi (%)'], 0)}% số ngày",
                  f"chiết khấu TB {fmt(T.loc['Giá thực tế', 'Chiết khấu TB (%)'], 1)}%", delta_color="off")
        st.dataframe(T.round(2), width="stretch")
    fig("e3_so_sanh_chinh_sach", "So sánh 3 cách đặt giá trên test (2.003 sản phẩm–cửa hàng, 22 ngày)")
    fig("e4_loi_nhuan_theo_ngay")
    if "8_rq2" in M.get("NOTES", {}):
        st.success(M["NOTES"]["8_rq2"])

# ---------------- Tab 4: Độ vững và danh mục ----------------
with tab4:
    st.subheader("Kiểm tra độ vững")
    c1, c2 = st.columns(2)
    T91, T92 = load_table("bang_e3_danh_gia_cheo", index_col=0), load_table("bang_e4_do_nhay", index_col=0)
    if T91 is not None:
        c1.markdown("**Đánh giá chéo:** chấm lại bằng độ co giãn của hồi quy (Bước 4B) thay vì Random Forest")
        c1.dataframe(T91, width="stretch")
    if T92 is not None:
        c2.markdown("**Phân tích độ nhạy:** lợi nhuận tăng thêm so với giá thực tế (% doanh thu)")
        c2.dataframe(T92, width="stretch")
    fig("e6_do_vung")
    st.subheader("Theo danh mục sản phẩm")
    T10 = load_table("bang_e5_theo_danh_muc")
    if T10 is not None:
        st.dataframe(T10, hide_index=True, width="stretch")
    fig("e7_theo_danh_muc")

# ---------------- Tab 5: Phương pháp ----------------
with tab5:
    st.markdown(f"""
### Pipeline
1. **Dữ liệu:** FreshRetailNet-50K (Dingdong, 28/3–2/7/2024), thành phố {M.get('CITY', 0)}, {M['S_N']:,} sản phẩm–cửa hàng. Nguồn: [Hugging Face](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K).
2. **Khôi phục nhu cầu bị che khuất** khi hết hàng, dựa trên tỷ lệ bán theo giờ.
3. **Dự báo nhu cầu:** so sánh Linear Regression, Random Forest, XGBoost, LSTM, Hybrid; chọn **{M['MODEL']}** vì vừa chính xác vừa phản ứng với giá gần nhất với ước lượng hồi quy (độ co giãn {M['ELAS']:.2f}).
4. **Engine:** với mỗi sản phẩm–ngày, dự báo nhu cầu ở 6 mức giá (70–100% giá gốc) và chọn mức có lợi nhuận dự kiến cao nhất.
5. **Đánh giá:** cửa hàng mô phỏng trên dữ liệu thật (nhập hàng base-stock, k = {M['K']}, hạn dùng {M['SHELF']} ngày), so sánh với giá thực tế và không khuyến mãi; kiểm tra chéo bằng mô hình hồi quy và phân tích độ nhạy.

### Tài liệu tham khảo chính
- Cohen, M. C., Leung, N.-H. Z., Panchamgam, K., Perakis, G., & Smith, A. (2017). The impact of linear optimization on promotion planning. *Operations Research*, 65(2), 446–468.
- Hua, J., Yan, L., Xu, H., & Yang, C. (2021). Markdowns in e-commerce fresh retail: A counterfactual prediction and multi-period optimization approach. *KDD 2021*.
- Haijema, R., et al. (2026). Dynamic expiration date-based discounting of fresh food products. *International Journal of Production Economics*, 292.
- Birr, S., et al. (2026). High-frequency pricing at scale for e-commerce. arXiv:2606.13741.
- Wang, Y., et al. (2025). FreshRetailNet-50K: A stockout-annotated censored demand dataset for perishable goods. arXiv:2505.16319.

### Lưu ý
Kết quả là **mô phỏng phản thực tế** trên dữ liệu lịch sử, không phải thử nghiệm A/B. Đơn vị nhu cầu theo thang của bộ dữ liệu (đã chuẩn hóa).
""")
    st.caption(f"Phiên bản thư viện khi huấn luyện: {M.get('VERSIONS', {})}")
