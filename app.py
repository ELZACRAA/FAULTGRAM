"""
STEREONET-DIGITAL APP
Klasifikasi Sesar Rickard 1971 & Analisis Kekar Konjugasi
===========================================================
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Mengamankan backend matplotlib agar tidak crash di Streamlit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

try:
    import mplstereonet
    HAS_MPLSTEREONET = True
except ImportError:
    HAS_MPLSTEREONET = False

st.set_page_config(page_title="FAULTGRAM", layout="wide", initial_sidebar_state="expanded")


# =====================================================================
# FUNGSI GEOMETRI & KONVERSI DASAR (VERSI ASLI)
# =====================================================================

def strike_dip_to_pole(strike, dip):
    strike_r = np.radians(strike)
    dip_dir_r = strike_r + np.pi / 2
    plunge = np.radians(90 - dip)
    x = np.cos(plunge) * np.cos(dip_dir_r)
    y = np.cos(plunge) * np.sin(dip_dir_r)
    z = np.sin(plunge)
    return np.array([x, y, z])


def vector_to_trend_plunge(vec):
    x, y, z = vec
    if z < 0:
        x, y, z = -x, -y, -z
    plunge = np.degrees(np.arcsin(np.clip(z, -1, 1)))
    trend = np.degrees(np.arctan2(y, x)) % 360
    return trend, plunge


def normal_to_plane(vec):
    trend, plunge = vector_to_trend_plunge(vec)
    dip = 90 - plunge
    strike = (trend - 90) % 360
    return strike, dip


def angle_between_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))


# =====================================================================
# ANALISIS POPULASI & PEAK DENSITY
# =====================================================================

def find_dominant_plane(strike_arr, dip_arr, grid_step=4):
    """Mencari satu bidang dominan harga umum."""
    data_poles = np.array([strike_dip_to_pole(s, d) for s, d in zip(strike_arr, dip_arr)])
    best_s, best_d, best_density = 0, 0, -1
    for s in range(0, 360, grid_step):
        for d in range(0, 91, grid_step):
            p = strike_dip_to_pole(s, d)
            cos_ang = np.clip(data_poles @ p, -1, 1)
            density = np.sum(np.exp((cos_ang - 1) * 30))
            if density > best_density:
                best_s, best_d, best_density = s, d, density
    return best_s, best_d


def find_conjugate_shear_planes(strike_arr, dip_arr):
    """Memisahkan populasi data menjadi 2 kelompok menggunakan K-Means."""
    poles = np.array([strike_dip_to_pole(s, d) for s, d in zip(strike_arr, dip_arr)])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(poles)
    labels = kmeans.labels_
    
    idx1 = np.where(labels == 0)[0]
    idx2 = np.where(labels == 1)[0]
    
    s1, d1 = find_dominant_plane(strike_arr[idx1], dip_arr[idx1])
    s2, d2 = find_dominant_plane(strike_arr[idx2], dip_arr[idx2])
    
    return (s1, d1), (s2, d2), labels


# =====================================================================
# VISUALISASI STEREONET GASH & SHEAR CONJUGATE (TIDAK DIUBAH)
# =====================================================================

def plot_fracture_analysis(strike_arr, dip_arr, dom_strike, dom_dip, title):
    fig = plt.figure(figsize=(6, 6))
    if HAS_MPLSTEREONET:
        try:
            ax = fig.add_subplot(111, projection='stereonet')
            s_arr, d_arr = np.array(strike_arr, dtype=float), np.array(dip_arr, dtype=float)
            ax.pole(s_arr, d_arr, "o", color="#1f77b4", markersize=4, alpha=0.6, label="Pole Data")
            try:
                ax.density_contourf(s_arr, d_arr, measurement="poles", cmap="RdYlBu_r", alpha=0.5)
            except:
                pass
            ax.pole(dom_strike, dom_dip, "*", color="red", markersize=12, label="Pole Harga Umum")
            ax.plane(dom_strike, dom_dip, color="crimson", lw=2, label="Bidang Harga Umum")
            ax.pole(dom_strike, dom_dip, "o", mfc="none", mec="black", markersize=8, lw=1.5, label="Polar Titik Harga Umum")
            ax.grid(True)
            ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.35, 0.1))
            return fig
        except:
            pass
    ax = fig.add_subplot(111, projection='polar')
    return fig


def plot_conjugate_shear_stereonet(strike_arr, dip_arr, set1, set2, labels):
    fig = plt.figure(figsize=(6, 6))
    if HAS_MPLSTEREONET:
        try:
            ax = fig.add_subplot(111, projection='stereonet')
            s_arr, d_arr = np.array(strike_arr, dtype=float), np.array(dip_arr, dtype=float)
            ax.pole(s_arr, d_arr, "o", color="#1f77b4", markersize=3, alpha=0.6, label="Pole Data Mentah")
            try:
                ax.density_contourf(s_arr, d_arr, measurement="poles", cmap="RdYlBu_r", alpha=0.5)
            except:
                pass
            ax.pole(set1[0], set1[1], "*", color="red", markersize=10, label="Pole Harga Umum (Dua Set)")
            ax.pole(set2[0], set2[1], "*", color="red", markersize=10)
            ax.plane(set1[0], set1[1], color="crimson", lw=2, label="Bidang Harga Umum (Dua Set)")
            ax.plane(set2[0], set2[1], color="crimson", lw=2)
            ax.pole(set1[0], set1[1], "o", mfc="none", mec="black", markersize=8, lw=1.5, label="Polar Titik Harga Umum")
            ax.pole(set2[0], set2[1], "o", mfc="none", mec="black", markersize=8, lw=1.5)
            
            n1 = strike_dip_to_pole(set1[0], set1[1])
            n2 = strike_dip_to_pole(set2[0], set2[1])
            sig2_vec = np.cross(n1, n2)
            if np.linalg.norm(sig2_vec) > 1e-9:
                sig2_vec /= np.linalg.norm(sig2_vec)
                s2t, s2p = vector_to_trend_plunge(sig2_vec)
                ax.line(s2p, s2t, "ks", markersize=7, label="σ2 (Intersection)")
            ax.grid(True)
            ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.4, 0.1))
            return fig
        except:
            pass
    ax = fig.add_subplot(111, projection='polar')
    return fig


def plot_stress_stereonet(fault_strike, fault_dip, aux_strike, aux_dip, sigma1_vec, sigma2_vec, sigma3_vec, title):
    s1t, s1p = vector_to_trend_plunge(sigma1_vec)
    s2t, s2p = vector_to_trend_plunge(sigma2_vec)
    s3t, s3p = vector_to_trend_plunge(sigma3_vec)
    fig = plt.figure(figsize=(6, 6))
    if HAS_MPLSTEREONET:
        try:
            ax = fig.add_subplot(111, projection='stereonet')
            ax.plane(fault_strike, fault_dip, color="steelblue", lw=2, label="Bidang Sesar")
            ax.plane(aux_strike, aux_dip, color="seagreen", lw=2, ls="--", label="Bidang Bantu")
            ax.line(s1p, s1t, "rs", markersize=8, label=f"σ1 ({s1t:.0f}/{s1p:.0f})")
            ax.line(s2p, s2t, "o", color="orange", markersize=8, label=f"σ2 ({s2t:.0f}/{s2p:.0f})")
            ax.line(s3p, s3t, "^", color="purple", markersize=8, label=f"σ3 ({s3t:.0f}/{s3p:.0f})")
            ax.grid(True)
            ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.35, 1.1))
            return fig
        except:
            pass
    return fig


# =====================================================================
# STRUKTUR KINEMATIKA & PERBAIKAN PLOT GABUNGAN TITIK TUNGGAL (TAB 5)
# =====================================================================

def calculate_kinematics_from_conjugate(shear_set1, shear_set2, fault_strike, fault_dip):
    """Menghitung sumbu tegasan dari kekar gerus konjugasi sesuai lembar panduan."""
    n1 = strike_dip_to_pole(shear_set1[0], shear_set1[1])
    n2 = strike_dip_to_pole(shear_set2[0], shear_set2[1])
    
    sigma2_vec = np.cross(n1, n2)
    sigma2_vec /= np.linalg.norm(sigma2_vec)
    s2_tr, s2_pl = vector_to_trend_plunge(sigma2_vec)
    
    b1 = n1 + n2
    b2 = n1 - n2
    b1 /= np.linalg.norm(b1)
    b2 /= np.linalg.norm(b2)
    
    ang = angle_between_vectors(n1, n2)
    if ang <= 90:
        sigma1_vec, sigma3_vec = b1, b2
    else:
        sigma1_vec, sigma3_vec = b2, b1
        
    s1_tr, s1_pl = vector_to_trend_plunge(sigma1_vec)
    s3_tr, s3_pl = vector_to_trend_plunge(sigma3_vec)
    
    # Bidang Bantu adalah polar dari Sigma 2
    aux_s, aux_d = normal_to_plane(sigma2_vec)
    
    # Net Slip adalah perpotongan Bidang Sesar dengan Bidang Bantu
    fault_normal = strike_dip_to_pole(fault_strike, fault_dip)
    ns = np.cross(fault_normal, sigma2_vec)
    if np.linalg.norm(ns) > 1e-9:
        ns /= np.linalg.norm(ns)
    else:
        ns = sigma1_vec - np.dot(sigma1_vec, fault_normal) * fault_normal
        ns /= np.linalg.norm(ns)
        
    ns_tr, ns_pl = vector_to_trend_plunge(ns)
    
    # Nilai pitch diukur dari jurus bidang sesar ke net slip
    st_r = np.radians(fault_strike)
    st_vec = np.array([np.cos(st_r), np.sin(st_r), 0.0])
    raw_ang = angle_between_vectors(ns, st_vec)
    pitch = raw_ang if raw_ang <= 90 else 180 - raw_ang
    
    return {
        "fault_strike": fault_strike, "fault_dip": fault_dip,
        "aux_strike": aux_s, "aux_dip": aux_d,
        "sigma1": (s1_tr, s1_pl), "sigma2": (s2_tr, s2_pl), "sigma3": (s3_tr, s3_pl),
        "netslip_trend": ns_tr, "netslip_plunge": ns_pl, "pitch": pitch,
    }


def plot_compilation_stereonet(gash_dom, shear_set1, shear_set2, kin_res):
    """
    PERBAIKAN KOREKSI PLOT: Menggunakan metode ekstraksi fungsi internal mplstereonet.line() 
    ke dalam objek dasar ax.plot() Matplotlib untuk memaksa penggambaran berupa TITIK TUNGGAL murni.
    """
    fig = plt.figure(figsize=(7, 7))
    if not HAS_MPLSTEREONET:
        ax = fig.add_subplot(111, projection='polar')
        return fig
        
    try:
        ax = fig.add_subplot(111, projection='stereonet')
        
        # 1. Menggambar Lingkaran Besar Bidang Sesar dan Bidang Bantu
        ax.plane(kin_res["fault_strike"], kin_res["fault_dip"], color="black", lw=2.2, label="Bidang Sesar")
        ax.plane(kin_res["aux_strike"], kin_res["aux_dip"], color="gray", lw=1.2, ls=":", label="Bidang Bantu")
        
        if gash_dom:
            ax.plane(gash_dom[0], gash_dom[1], color="blue", lw=1.2, ls="--", label="Bidang Gash")
        if shear_set1:
            ax.plane(shear_set1[0], shear_set1[1], color="red", lw=1.2, label="Bidang Shear 1")
        if shear_set2:
            ax.plane(shear_set2[0], shear_set2[1], color="green", lw=1.2, label="Bidang Shear 2")
            
        # 2. SOLUSI FIX: Mengonversi kedudukan trend/plunge titik ke koordinat kartesian proyeksi 
        #    stereonet lewat fungsi internal mplstereonet.line() untuk mengunci wujudnya menjadi TITIK.
        s1_trend, s1_plunge = kin_res["sigma1"]
        s2_trend, s2_plunge = kin_res["sigma2"]
        s3_trend, s3_plunge = kin_res["sigma3"]
        ns_trend, ns_plunge = kin_res["netslip_trend"], kin_res["netslip_plunge"]
        
        s1_lon, s1_lat = mplstereonet.line(s1_plunge, s1_trend)
        s2_lon, s2_lat = mplstereonet.line(s2_plunge, s2_trend)
        s3_lon, s3_lat = mplstereonet.line(s3_plunge, s3_trend)
        ns_lon, ns_lat = mplstereonet.line(ns_plunge, ns_trend)
        
        # Plot objek murni menggunakan ax.plot() standar koordinat proyeksi stereonet
        ax.plot(s1_lon, s1_lat, "s", color="red", markersize=9, label=f"σ1 ({s1_trend:.0f}/{s1_plunge:.0f})")
        ax.plot(s2_lon, s2_lat, "o", color="orange", markersize=9, label=f"σ2 ({s2_trend:.0f}/{s2_plunge:.0f})")
        ax.plot(s3_lon, s3_lat, "^", color="purple", markersize=9, label=f"σ3 ({s3_trend:.0f}/{s3_plunge:.0f})")
        ax.plot(ns_lon, ns_lat, "X", color="cyan", markersize=10, mec="black", label=f"Net Slip ({ns_trend:.0f}/{ns_plunge:.0f})")
        
        ax.grid(True, color="gainsboro", linestyle="-", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.35, 0.0))
        ax.set_title("Kompilasi Data Stereonet Akhir Gabungan", fontsize=11, fontweight='bold', pad=15)
        return fig
    except Exception as e:
        st.error(f"Gagal memplot kompilasi akhir: {str(e)}")
        return fig


# =====================================================================
# MAIN APPLICATION INTERFACE (TAMPILAN ASLI KESUKAAN ANDA)
# =====================================================================

def main():
    with st.sidebar:
        st.header("⚙️ Panel Kontrol Input")
        uploaded_file = st.file_uploader("Upload File Data Lapangan", type=["csv", "txt"])

    st.title("FAULTGRAM")
    st.subheader("Analisis Kinematik Sesar & Klasifikasi Sesar Rickard 1971")
    st.markdown("---")

    if uploaded_file is None:
        st.info("👈 Silakan upload file CSV/TXT data lapangan pada sidebar.")
        return

    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        st.error(f"Gagal membaca file: {str(e)}")
        return

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
        "📊 Data Viewer", "🔹 Gash Fracture", "🔸 Shear Fracture (Conjugate)", "🧭 Sumbu Tegasan & Kinematika", "🏷️ Hasil Analisis Kompilasi Akhir"
    ])

    with tab1:
        st.dataframe(df, use_container_width=True)

    gash_dom = None
    with tab2:
        st.subheader("🔹 Analisis Gash Fracture")
        if gash_strike_col != "(Tidak ada)" and gash_dip_col != "(Tidak ada)":
            sub = df[[gash_strike_col, gash_dip_col]].apply(pd.to_numeric, errors="coerce").dropna()
            sub.columns = ["strike", "dip"]
            if len(sub) > 0:
                dom_s, dom_d = find_dominant_plane(sub["strike"], sub["dip"])
                gash_dom = (dom_s, dom_d)
                st.metric("Gash Strike/Dip Harga Umum", f"{dom_s:.0f}°/{dom_d:.0f}°")
                fig = plot_fracture_analysis(sub["strike"], sub["dip"], dom_s, dom_d, "Gash Plane Analysis")
                if fig is not None:
                    st.pyplot(fig)

    shear_set1, shear_set2 = None, None
    with tab3:
        st.subheader("🔸 Analisis Shear Fracture Konjugasi (Poin 2 - Dua Set Harga Umum)")
        if shear_strike_col != "(Tidak ada)" and shear_dip_col != "(Tidak ada)":
            sub = df[[shear_strike_col, shear_dip_col]].apply(pd.to_numeric, errors="coerce").dropna()
            sub.columns = ["strike", "dip"]
            if len(sub) > 0:
                set1, set2, labels = find_conjugate_shear_planes(sub["strike"].values, sub["dip"].values)
                shear_set1, shear_set2 = set1, set2
                
                n1 = strike_dip_to_pole(set1[0], set1[1])
                n2 = strike_dip_to_pole(set2[0], set2[1])
                sudut_konj = angle_between_vectors(n1, n2)
                if sudut_konj > 90: sudut_konj = 180 - sudut_konj

                cc1, cc2, cc3 = st.columns(3)
                with cc1: st.metric("Strike/Dip Harga Umum Set 1", f"{set1[0]:.0f}° / {set1[1]:.0f}°")
                with cc2: st.metric("Strike/Dip Harga Umum Set 2", f"{set2[0]:.0f}° / {set2[1]:.0f}°")
                with cc3: st.metric("Sudut Konjugasi", f"{sudut_konj:.0f}°")

                fig_conj = plot_conjugate_shear_stereonet(sub["strike"], sub["dip"], set1, set2, labels)
                if fig_conj is not None:
                    st.pyplot(fig_conj)

    kinematics_result = None
    with tab4:
        st.subheader("🧭 Sumbu Tegasan & Kinematika")
        if shear_set1 and shear_set2 and fault_strike_col != "(Tidak ada)" and fault_dip_col != "(Tidak ada)":
            f_sub = df[[fault_strike_col, fault_dip_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(f_sub) > 0:
                f_strike = float(f_sub.iloc[0].iloc[0])
                f_dip = float(f_sub.iloc[0].iloc[1])
                
                kinematics_result = calculate_kinematics_from_conjugate(shear_set1, shear_set2, f_strike, f_dip)
                fig = plot_stress_stereonet(f_strike, f_dip, kinematics_result["aux_strike"], kinematics_result["aux_dip"],
                                            strike_dip_to_pole(kinematics_result["sigma1"][0], kinematics_result["sigma1"][1]), 
                                            strike_dip_to_pole(kinematics_result["sigma2"][0], kinematics_result["sigma2"][1]), 
                                            strike_dip_to_pole(kinematics_result["sigma3"][0], kinematics_result["sigma3"][1]), 
                                            "Stress Tensor Configuration")
                if fig is not None:
                    st.pyplot(fig)

    with tab5:
        st.subheader("🏷️ Hasil Akhir Analisis Kompilasi Gabungan")
        if kinematics_result is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Kedudukan Sesar Aktual:** {kinematics_result['fault_strike']:.0f}°/{kinematics_result['fault_dip']:.0f}°")
                st.info(f"**Nilai Pitch Net Slip:** {kinematics_result['pitch']:.1f}°")
            with col2:
                s1t, s1p = kinematics_result["sigma1"]
                ax_s2t, ax_s2p = kinematics_result["sigma2"]
                st.warning(f"**Kedudukan Sumbu σ1 (Lancip):** {s1t:.1f}°/{s1p:.1f}°")
                st.warning(f"**Kedudukan Sumbu σ2 (Perpotongan):** {ax_s2t:.1f}°/{ax_s2p:.1f}°")
                
            st.markdown("---")
            fig_compile = plot_compilation_stereonet(gash_dom, shear_set1, shear_set2, kinematics_result)
            if fig_compile is not None:
                st.pyplot(fig_compile)
        else:
            st.info("Lengkapi analisis pada Tab 'Sumbu Tegasan & Kinematika' terlebih dahulu.")


if __name__ == "__main__":
    main()
