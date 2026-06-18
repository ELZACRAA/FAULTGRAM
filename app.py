"""
STEREONET-DIGITAL APP
Klasifikasi Sesar Rickard 1972 & Analisis Kekar Konjugasi
===========================================================
Mengikuti langkah kerja:
 1-5. Gash fracture  -> pole -> kontur -> titik harga umum -> bidang harga umum
 6.   Shear fracture -> (langkah 1-5 yang sama)
 7.   Input bidang sesar
 8.   Sigma 2 = perpotongan bidang harga umum gash & shear
 9.   Bidang bantu (tegak lurus bidang sesar, mengandung sigma 2)
 10.  Sigma 1 = bisektor bidang sesar & bidang bantu
 11.  Sigma 3 = bisektor lainnya (tegak lurus sigma1 & sigma2)
 12.  Net slip = proyeksi sigma1 ke bidang sesar
 13.  Pitch = sudut net slip terhadap jurus bidang sesar
 14.  Klasifikasi Rickard (1972)

Jalankan:
    pip install streamlit pandas numpy scikit-learn matplotlib mplstereonet
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import mplstereonet
    HAS_MPLSTEREONET = True
except ImportError:
    HAS_MPLSTEREONET = False

st.set_page_config(page_title="FAULTGRAM", layout="wide",
                    initial_sidebar_state="expanded")


# =====================================================================
# FUNGSI KONVERSI GEOMETRI DASAR
# =====================================================================

def strike_dip_to_pole(strike, dip):
    """Konversi Strike/Dip (RHR) -> vektor pole 3D (X=North, Y=East, Z=Down)."""
    strike_r = np.radians(strike)
    dip_r = np.radians(dip)
    dip_dir_r = strike_r + np.pi / 2
    plunge = np.radians(90 - dip)
    x = np.cos(plunge) * np.cos(dip_dir_r)
    y = np.cos(plunge) * np.sin(dip_dir_r)
    z = np.sin(plunge)
    return np.array([x, y, z])


def line_to_vector(trend, plunge):
    """Konversi Trend/Plunge -> vektor 3D (X=North, Y=East, Z=Down)."""
    tr, pl = np.radians(trend), np.radians(plunge)
    return np.array([np.cos(pl) * np.cos(tr), np.cos(pl) * np.sin(tr), np.sin(pl)])


def vector_to_trend_plunge(vec):
    """Konversi vektor 3D -> (trend, plunge) derajat. Selalu plunge >= 0."""
    x, y, z = vec
    if z < 0:
        x, y, z = -x, -y, -z
    plunge = np.degrees(np.arcsin(np.clip(z, -1, 1)))
    trend = np.degrees(np.arctan2(y, x)) % 360
    return trend, plunge


def normal_to_plane(vec):
    """
    Konversi vektor normal/pole bidang -> (strike, dip) bidang tersebut.
    Kebalikan dari strike_dip_to_pole: trend pole = dip direction,
    plunge pole = 90 - dip, sehingga strike = dip_dir - 90.
    """
    trend, plunge = vector_to_trend_plunge(vec)
    dip = 90 - plunge
    strike = (trend - 90) % 360
    return strike, dip


def angle_between_vectors(v1, v2):
    """Sudut (derajat, 0-180) antara dua vektor 3D."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))


# =====================================================================
# LANGKAH 1-5 & 6: ANALISIS KEKAR (GASH / SHEAR) -> BIDANG HARGA UMUM
# =====================================================================

def find_dominant_plane(strike_arr, dip_arr, grid_step=5):
    """
    Mencari 'titik harga umum' (puncak kontur kerapatan pole) lalu
    mengembalikan 'bidang harga umum' (strike, dip) yang bersesuaian.

    Metode: grid-search seluruh kemungkinan bidang (step=grid_step derajat),
    untuk setiap kandidat bidang dihitung pole-nya, lalu dijumlahkan
    kedekatan (kernel kosinus) terhadap seluruh pole data. Kandidat dengan
    densitas tertinggi = bidang harga umum (pole-nya = pusat kerapatan
    data pole, persis seperti puncak kontur pada stereonet).

    Returns
    -------
    dom_strike, dom_dip, density_map (untuk visualisasi opsional)
    """
    data_poles = np.array([strike_dip_to_pole(s, d) for s, d in zip(strike_arr, dip_arr)])

    best_s, best_d, best_density = 0, 0, -1
    for s in range(0, 360, grid_step):
        for d in range(0, 91, grid_step):
            p = strike_dip_to_pole(s, d)
            cos_ang = np.clip(data_poles @ p, -1, 1)
            density = np.sum(np.exp((cos_ang - 1) * 25))  # kernel sudut sempit
            if density > best_density:
                best_s, best_d, best_density = s, d, density

    return best_s, best_d, best_density


# =====================================================================
# LANGKAH 8-13: SUMBU TEGASAN & KINEMATIKA (METODE KONJUGASI)
# =====================================================================

def calculate_sigma2(gash_strike, gash_dip, shear_strike, shear_dip):
    """
    Langkah 8: Sigma 2 = garis perpotongan antara bidang harga umum
    gash fracture dan bidang harga umum shear fracture (cross product
    dari kedua vektor normal/pole-nya).
    """
    n_gash = strike_dip_to_pole(gash_strike, gash_dip)
    n_shear = strike_dip_to_pole(shear_strike, shear_dip)
    v = np.cross(n_gash, n_shear)
    if np.linalg.norm(v) < 1e-9:
        v = np.array([0.0, 0.0, 1.0])
    v = v / np.linalg.norm(v)
    trend, plunge = vector_to_trend_plunge(v)
    return v, trend, plunge


def calculate_auxiliary_plane(fault_strike, fault_dip, sigma2_vec):
    """
    Langkah 9: Bidang bantu adalah bidang yang TEGAK LURUS terhadap bidang
    sesar dan MENGANDUNG sumbu sigma 2. Normal bidang bantu harus tegak
    lurus terhadap normal bidang sesar (agar kedua bidang saling tegak
    lurus) dan tegak lurus terhadap sigma2 (agar sigma2 terletak pada
    bidang bantu) -> normal_aux = fault_normal x sigma2.
    """
    fault_normal = strike_dip_to_pole(fault_strike, fault_dip)
    aux_normal = np.cross(fault_normal, sigma2_vec)
    aux_normal = aux_normal / np.linalg.norm(aux_normal)
    aux_strike, aux_dip = normal_to_plane(aux_normal)
    return aux_normal, aux_strike, aux_dip, fault_normal


def calculate_sigma1_sigma3(fault_normal, aux_normal):
    """
    Langkah 10-11: Sigma 1 dan Sigma 3 adalah BISEKTOR dari normal bidang
    sesar dan normal bidang bantu (keduanya otomatis tegak lurus sigma2,
    karena sigma2 tegak lurus terhadap kedua normal tersebut). Dua
    bisektor yang saling tegak lurus ini masing-masing merupakan kandidat
    sigma1 (kompresi maksimum) dan sigma3 (ekstensi maksimum).

    NOTE: penentuan mana yang sigma1 vs sigma3 di sini menggunakan
    asumsi default (b1=sigma1, b2=sigma3). Jika arah ini tidak sesuai
    dengan indikator pergerakan sesar di lapangan, gunakan opsi 'Tukar
    Sigma1 <-> Sigma3' pada aplikasi.
    """
    b1 = fault_normal + aux_normal
    b2 = fault_normal - aux_normal
    b1 = b1 / np.linalg.norm(b1)
    b2 = b2 / np.linalg.norm(b2)
    return b1, b2


def calculate_netslip_pitch(fault_strike, fault_dip, fault_normal, sigma1_vec):
    """
    Langkah 12-13:
    - Net slip = proyeksi sigma1 ke dalam bidang sesar (komponen sigma1
      yang tegak lurus normal bidang sesar dihilangkan).
    - Pitch = sudut antara net slip dan jurus (strike) bidang sesar,
      diukur pada bidang sesar (0-90 derajat).
    Juga dihitung sense vertikal (normal/reverse) & lateral (kanan/kiri)
    sebagai heuristik untuk klasifikasi Rickard.
    """
    ns = sigma1_vec - np.dot(sigma1_vec, fault_normal) * fault_normal
    ns = ns / np.linalg.norm(ns)
    netslip_trend, netslip_plunge = vector_to_trend_plunge(ns)

    strike_r = np.radians(fault_strike)
    strike_vec = np.array([np.cos(strike_r), np.sin(strike_r), 0.0])

    raw_angle = angle_between_vectors(ns, strike_vec)
    pitch = raw_angle if raw_angle <= 90 else 180 - raw_angle

    # Komponen vertikal & lateral net slip (sebelum dijadikan plunge>=0)
    x, y, z = ns
    sense_hint = "normal-like" if z >= 0 else "reverse-like"

    comp_strike = np.dot(ns, strike_vec)
    lateral_hint = "dextral" if comp_strike >= 0 else "sinistral"

    return {
        "netslip_trend": netslip_trend,
        "netslip_plunge": netslip_plunge,
        "pitch": pitch,
        "sense_hint": sense_hint,
        "lateral_hint": lateral_hint,
    }


# =====================================================================
# LANGKAH 14: KLASIFIKASI RICKARD 1972 (22 NAMA SESAR)
# =====================================================================

def classify_rickard_1972(dip, pitch, sense_hint, lateral_hint="dextral"):
    """
    Klasifikasi 22 nama sesar Rickard (1972):
      - Komponen vertikal: Normal / Thrust (dip<45) / Reverse (dip>=45)
      - Komponen lateral : Right (dekstral) / Left (sinistral)
      - pitch < 22.5  -> murni lateral  : "{Lateral} Slip Fault"
      - pitch > 67.5  -> murni vertikal : "{Vertikal} Slip Fault"
      - 22.5-67.5     -> oblique, komponen dominan ditulis lebih dulu
                         (pitch<=45 -> lateral dulu, >45 -> vertikal dulu)
    """
    if sense_hint == "normal-like":
        vertical = "Normal"
    else:
        vertical = "Thrust" if dip < 45 else "Reverse"

    lateral = "Right" if lateral_hint == "dextral" else "Left"

    if pitch < 22.5:
        kategori = "Strike-Slip"
        nama_sesar = f"{lateral} Slip Fault"
        arah = (f"Dominan mendatar, bergerak relatif ke "
                f"{'kanan (dekstral)' if lateral=='Right' else 'kiri (sinistral)'}")
    elif pitch > 67.5:
        kategori = "Dip-Slip"
        nama_sesar = f"{vertical} Slip Fault"
        arah = ("Hanging wall turun terhadap footwall" if vertical == "Normal"
                else "Hanging wall naik terhadap footwall")
    else:
        kategori = "Oblique-Slip"
        if pitch <= 45:
            nama_sesar = f"{lateral} {vertical} Slip Fault"
        else:
            nama_sesar = f"{vertical} {lateral} Slip Fault"
        vert_desc = "turun (normal)" if vertical == "Normal" else "naik (thrust/reverse)"
        lat_desc = "kanan (dekstral)" if lateral == "Right" else "kiri (sinistral)"
        arah = f"Kombinasi pergerakan {vert_desc} dan mendatar ke {lat_desc}"

    if dip < 30:
        dip_class = "Low-angle"
    elif dip < 60:
        dip_class = "Moderate-angle"
    else:
        dip_class = "High-angle"

    return {
        "nama_sesar": nama_sesar,
        "arah_pergerakan": arah,
        "kategori_rickard": kategori,
        "dip_class": dip_class,
    }


# =====================================================================
# FUNGSI DATA & VISUALISASI
# =====================================================================

def load_data(uploaded_file):
    """Baca CSV/TXT (delimiter auto-detect), bersihkan nama kolom."""
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
        df.columns = [c.strip() for c in df.columns]
        return df, None
    except Exception as e:
        return None, str(e)


def extract_columns(df, strike_col, dip_col):
    """Ambil kolom strike/dip, drop NaN, pastikan numerik."""
    if strike_col not in df.columns or dip_col not in df.columns:
        return None
    sub = df[[strike_col, dip_col]].copy()
    sub = sub.apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna()
    sub.columns = ["strike", "dip"]
    return sub.reset_index(drop=True)


def plot_fracture_analysis(strike_arr, dip_arr, dom_strike, dom_dip, title):
    """
    Plot kontur kerapatan pole (langkah 2-3) + pole data + bidang & pole
    harga umum (langkah 4-5).
    """
    if HAS_MPLSTEREONET:
        fig, ax = mplstereonet.subplots(figsize=(6, 6))
        try:
            ax.density_contourf(strike_arr, dip_arr, measurement="poles",
                                 cmap="Blues", alpha=0.6)
        except Exception:
            pass
        ax.pole(np.array(strike_arr, dtype=float), np.array(dip_arr, dtype=float),
                "o", color="navy", markersize=4, label="Pole Data")
        ax.pole(dom_strike, dom_dip, "r*", markersize=18, label="Pole Harga Umum")
        ax.plane(dom_strike, dom_dip, color="red", lw=2, label="Bidang Harga Umum")
        try:
            ax.grid(True)
        except Exception:
            pass
        ax.set_title(title, fontsize=12, pad=20)
        ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.35, 1.1))
        return fig
    else:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 90)
        for s, d in zip(strike_arr, dip_arr):
            t, p = vector_to_trend_plunge(strike_dip_to_pole(s, d))
            ax.plot(np.radians(t), 90 - p, "o", color="navy", markersize=4)
        t, p = vector_to_trend_plunge(strike_dip_to_pole(dom_strike, dom_dip))
        ax.plot(np.radians(t), 90 - p, "r*", markersize=18, label="Pole Harga Umum")
        ax.set_title(title + "\n(fallback - mplstereonet tidak tersedia)", fontsize=10)
        ax.legend()
        return fig


def plot_stress_stereonet(fault_strike, fault_dip, aux_strike, aux_dip,
                            sigma1_vec, sigma2_vec, sigma3_vec, title):
    """Plot bidang sesar, bidang bantu, dan ketiga sumbu tegasan."""
    s1t, s1p = vector_to_trend_plunge(sigma1_vec)
    s2t, s2p = vector_to_trend_plunge(sigma2_vec)
    s3t, s3p = vector_to_trend_plunge(sigma3_vec)

    if HAS_MPLSTEREONET:
        fig, ax = mplstereonet.subplots(figsize=(6, 6))
        ax.plane(fault_strike, fault_dip, color="steelblue", lw=2, label="Bidang Sesar")
        ax.plane(aux_strike, aux_dip, color="seagreen", lw=2, ls="--", label="Bidang Bantu")
        ax.line(s1p, s1t, "rs", markersize=10, label=f"σ1 ({s1t:.0f}/{s1p:.0f})")
        ax.line(s2p, s2t, "o", color="orange", markersize=10, label=f"σ2 ({s2t:.0f}/{s2p:.0f})")
        ax.line(s3p, s3t, "^", color="purple", markersize=10, label=f"σ3 ({s3t:.0f}/{s3p:.0f})")
        try:
            ax.grid(True)
        except Exception:
            pass
        ax.set_title(title, fontsize=12, pad=20)
        ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.4, 1.1))
        return fig
    else:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 90)
        for (t, p, m, c, lab) in [(s1t, s1p, "s", "red", "σ1"),
                                    (s2t, s2p, "o", "orange", "σ2"),
                                    (s3t, s3p, "^", "purple", "σ3")]:
            ax.plot(np.radians(t), 90 - p, m, color=c, markersize=10, label=lab)
        ax.set_title(title + "\n(fallback - mplstereonet tidak tersedia)", fontsize=10)
        ax.legend()
        return fig


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


# =====================================================================
# MAIN APP
# =====================================================================

def main():
    with st.sidebar:
        st.header("⚙️ Panel Kontrol Input")
        st.markdown("Upload data lapangan dalam format **CSV/TXT** (maks 200MB).")
        uploaded_file = st.file_uploader("Upload File Data Lapangan", type=["csv", "txt"])
        st.markdown("---")
        swap_sigma = st.checkbox("Tukar Sigma1 <-> Sigma3", value=False,
                                  help="Aktifkan jika hasil sigma1/sigma3 terbalik "
                                       "dibanding indikator lapangan.")

    st.title("FAULTGRAM")
    st.subheader("Analisis Kekar Konjugasi & Klasifikasi Sesar Rickard 1972")
    st.markdown("---")

    if uploaded_file is None:
        st.info("👈 Silakan upload file CSV/TXT data lapangan pada sidebar.")
        st.markdown("""
        **Kolom yang dibutuhkan (nama bisa berbeda, akan dipetakan manual):**
        - Gash Fracture: Strike, Dip
        - Shear Fracture: Strike, Dip
        - Bidang Sesar (Fault Plane): Strike, Dip
        """)
        return

    df, err = load_data(uploaded_file)
    if err:
        st.error(f"Gagal membaca file: {err}")
        return

    st.success(f"File berhasil dimuat: **{uploaded_file.name}** "
               f"({df.shape[0]} baris, {df.shape[1]} kolom)")

    with st.expander("🔧 Pemetaan Kolom Data", expanded=True):
        cols = ["(Tidak ada)"] + list(df.columns)
        c1, c2, c3 = st.columns(3)
        with c1:
            gash_strike_col = st.selectbox("Gash Fracture - Strike", cols, key="gs")
            gash_dip_col = st.selectbox("Gash Fracture - Dip", cols, key="gd")
        with c2:
            shear_strike_col = st.selectbox("Shear Fracture - Strike", cols, key="ss")
            shear_dip_col = st.selectbox("Shear Fracture - Dip", cols, key="sd")
        with c3:
            fault_strike_col = st.selectbox("Bidang Sesar - Strike", cols, key="fls")
            fault_dip_col = st.selectbox("Bidang Sesar - Dip", cols, key="fld")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Viewer",
        "🔹 Gash Fracture",
        "🔸 Shear Fracture",
        "🧭 Sumbu Tegasan & Kinematika",
        "🏷️ Klasifikasi Rickard 1972",
    ])

    # ---------------- TAB 1: DATA VIEWER ----------------
    with tab1:
        st.subheader("📊 Data Viewer")
        st.dataframe(df, use_container_width=True)
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.markdown("**Statistik Deskriptif:**")
            st.dataframe(numeric_df.describe(), use_container_width=True)

    # ---------------- ANALISIS GASH FRACTURE ----------------
    gash_dom = None
    with tab2:
        st.subheader("🔹 Analisis Gash Fracture (Langkah 1-5)")
        gash_df = extract_columns(df, gash_strike_col, gash_dip_col)
        if gash_df is not None and len(gash_df) > 0:
            st.markdown(f"Jumlah data: **{len(gash_df)}**")
            dom_s, dom_d, _ = find_dominant_plane(gash_df["strike"], gash_df["dip"])
            gash_dom = (dom_s, dom_d)

            col1, col2 = st.columns(2)
            col1.metric("Strike Harga Umum", f"{dom_s:.0f}°")
            col2.metric("Dip Harga Umum", f"{dom_d:.0f}°")

            fig = plot_fracture_analysis(gash_df["strike"], gash_df["dip"], dom_s, dom_d,
                                          "Gash Fracture: Pole, Kontur & Bidang Harga Umum")
            st.pyplot(fig)
        else:
            st.info("Pilih kolom Gash Fracture (Strike & Dip) pada Pemetaan Kolom Data.")

    # ---------------- ANALISIS SHEAR FRACTURE ----------------
    shear_dom = None
    with tab3:
        st.subheader("🔸 Analisis Shear Fracture (Langkah 6)")
        shear_df = extract_columns(df, shear_strike_col, shear_dip_col)
        if shear_df is not None and len(shear_df) > 0:
            st.markdown(f"Jumlah data: **{len(shear_df)}**")
            dom_s, dom_d, _ = find_dominant_plane(shear_df["strike"], shear_df["dip"])
            shear_dom = (dom_s, dom_d)

            col1, col2 = st.columns(2)
            col1.metric("Strike Harga Umum", f"{dom_s:.0f}°")
            col2.metric("Dip Harga Umum", f"{dom_d:.0f}°")

            fig = plot_fracture_analysis(shear_df["strike"], shear_df["dip"], dom_s, dom_d,
                                          "Shear Fracture: Pole, Kontur & Bidang Harga Umum")
            st.pyplot(fig)
        else:
            st.info("Pilih kolom Shear Fracture (Strike & Dip) pada Pemetaan Kolom Data.")

    # ---------------- SUMBU TEGASAN & KINEMATIKA ----------------
    kinematics_result = None
    with tab4:
        st.subheader("🧭 Sumbu Tegasan & Kinematika (Langkah 7-13)")

        fault_df = extract_columns(df, fault_strike_col, fault_dip_col)

        if gash_dom is None or shear_dom is None:
            st.warning("Lengkapi dulu Tab 'Gash Fracture' dan 'Shear Fracture' "
                       "agar bidang harga umum tersedia (dibutuhkan untuk Sigma 2).")
        elif fault_df is None or len(fault_df) == 0:
            st.info("Pilih kolom Bidang Sesar (Strike & Dip) pada Pemetaan Kolom Data.")
        else:
            f_strike = float(fault_df.iloc[0]["strike"])
            f_dip = float(fault_df.iloc[0]["dip"])
            st.markdown(f"**Bidang Sesar digunakan:** Strike = {f_strike:.0f}°, "
                       f"Dip = {f_dip:.0f}° (baris data pertama)")

            # Langkah 8: Sigma 2
            sigma2_vec, sigma2_trend, sigma2_plunge = calculate_sigma2(
                gash_dom[0], gash_dom[1], shear_dom[0], shear_dom[1])

            # Langkah 9: Bidang bantu
            aux_normal, aux_strike, aux_dip, fault_normal = calculate_auxiliary_plane(
                f_strike, f_dip, sigma2_vec)

            # Langkah 10-11: Sigma 1 & Sigma 3
            b1, b2 = calculate_sigma1_sigma3(fault_normal, aux_normal)
            sigma1_vec, sigma3_vec = (b2, b1) if swap_sigma else (b1, b2)
            s1_trend, s1_plunge = vector_to_trend_plunge(sigma1_vec)
            s3_trend, s3_plunge = vector_to_trend_plunge(sigma3_vec)

            # Langkah 12-13: Net slip & pitch
            kin = calculate_netslip_pitch(f_strike, f_dip, fault_normal, sigma1_vec)
            kinematics_result = {
                "fault_strike": f_strike, "fault_dip": f_dip,
                "aux_strike": aux_strike, "aux_dip": aux_dip,
                "sigma1": (s1_trend, s1_plunge),
                "sigma2": (sigma2_trend, sigma2_plunge),
                "sigma3": (s3_trend, s3_plunge),
                **kin,
            }

            # --- Tampilkan hasil ---
            st.markdown("### 📋 Hasil Kalkulasi")
            res_df = pd.DataFrame({
                "Parameter": ["Bidang Bantu", "Sigma 1 (Trend/Plunge)",
                               "Sigma 2 (Trend/Plunge)", "Sigma 3 (Trend/Plunge)",
                               "Net Slip (Trend/Plunge)", "Pitch"],
                "Nilai": [
                    f"{aux_strike:.0f}°/{aux_dip:.0f}°",
                    f"{s1_trend:.0f}°/{s1_plunge:.0f}°",
                    f"{sigma2_trend:.0f}°/{sigma2_plunge:.0f}°",
                    f"{s3_trend:.0f}°/{s3_plunge:.0f}°",
                    f"{kin['netslip_trend']:.0f}°/{kin['netslip_plunge']:.0f}°",
                    f"{kin['pitch']:.1f}°",
                ]
            })
            st.dataframe(res_df, use_container_width=True, hide_index=True)

            fig = plot_stress_stereonet(f_strike, f_dip, aux_strike, aux_dip,
                                         sigma1_vec, sigma2_vec, sigma3_vec,
                                         "Bidang Sesar, Bidang Bantu & Sumbu Tegasan")
            st.pyplot(fig)

            csv_buffer = io.StringIO()
            res_df.to_csv(csv_buffer, index=False)
            st.download_button("⬇️ Download Hasil Kalkulasi (CSV)", csv_buffer.getvalue(),
                               "hasil_kalkulasi.csv", "text/csv")

            img_buf = fig_to_bytes(fig)
            st.download_button("⬇️ Download Gambar Stereonet (PNG)", img_buf,
                               "stereonet_tegasan.png", "image/png")

    # ---------------- KLASIFIKASI RICKARD 1972 ----------------
    with tab5:
        st.subheader("🏷️ Klasifikasi Sesar - Rickard 1972 (Langkah 14)")
        if kinematics_result is not None:
            rickard = classify_rickard_1972(
                kinematics_result["fault_dip"], kinematics_result["pitch"],
                kinematics_result["sense_hint"], kinematics_result["lateral_hint"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Nama Sesar (Rickard 1972):**")
                st.success(rickard["nama_sesar"])
                st.markdown("**Arah Pergerakan:**")
                st.info(rickard["arah_pergerakan"])
            with col2:
                s1t, s1p = kinematics_result["sigma1"]
                st.markdown("**Arah Gaya Utama (σ1):**")
                st.warning(f"Trend = {s1t:.1f}°, Plunge = {s1p:.1f}°")
                st.markdown(f"**Kategori Pitch:** {rickard['kategori_rickard']}")
                st.markdown(f"**Kelas Dip Bidang Sesar:** {rickard['dip_class']}")

            st.markdown("---")
            st.markdown("### 📊 Posisi pada Diagram Rickard (Dip vs Pitch)")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.axhline(22.5, color="gray", linestyle="--", lw=1)
            ax2.axhline(67.5, color="gray", linestyle="--", lw=1)
            ax2.axvline(45, color="gray", linestyle=":", lw=1)
            ax2.text(2, 10, "Strike-Slip", fontsize=9, color="dimgray")
            ax2.text(2, 45, "Oblique-Slip", fontsize=9, color="dimgray")
            ax2.text(2, 80, "Dip-Slip", fontsize=9, color="dimgray")
            ax2.plot(kinematics_result["fault_dip"], kinematics_result["pitch"],
                     "o", color="red", markersize=12, label="Data Sesar")
            ax2.set_xlim(0, 90)
            ax2.set_ylim(0, 90)
            ax2.set_xlabel("Dip Bidang Sesar (°)")
            ax2.set_ylabel("Pitch Net Slip (°)")
            ax2.set_title("Diagram Klasifikasi Rickard (1972) - Skema")
            ax2.legend()
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)

            st.caption("⚠️ Penentuan sigma1/sigma3 dan sense (normal/reverse, "
                      "dekstral/sinistral) bersifat heuristik geometris. "
                      "Jika tidak sesuai indikator lapangan, gunakan checkbox "
                      "'Tukar Sigma1 <-> Sigma3' di sidebar.")
        else:
            st.info("Lengkapi Tab 'Sumbu Tegasan & Kinematika' terlebih dahulu.")


if __name__ == "__main__":
    main()