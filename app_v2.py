import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------

c = 3e8
lambda0 = 1063.999e-9
f0 = c / lambda0


# --------------------------------------------------------------------
# Page configuration and session state
# --------------------------------------------------------------------

st.set_page_config(
    page_title="Filter Cavity Designer",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"], .stApp,
    .stMarkdown, .stMetric, .stButton, .stTextInput, .stNumberInput,
    .stSelectbox, .stSlider, .stRadio, .stCheckbox, .stCaption,
    .stAlert, .stExpander, .stDataFrame, .stTable, .stTabs,
    .stPlotlyChart, h1, h2, h3, h4, h5, h6, p, span, div, label, button,
    [data-testid="stMetric"], [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        font-family: "DejaVu Sans", "Bitstream Vera Sans", sans-serif !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        font-weight: 700 !important;
    }
    .block-container {
        padding-top: 0.4rem !important;
        padding-bottom: 0.4rem !important;
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
        height: 0rem;
        min-height: 0rem;
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "page" not in st.session_state:
    st.session_state.page = "landing"
if "geometry" not in st.session_state:
    st.session_state.geometry = None
if "two_mirror_params" not in st.session_state:
    st.session_state.two_mirror_params = None


def go_to(page_name):
    st.session_state.page = page_name


# --------------------------------------------------------------------
# Two-mirror cavity physics
# --------------------------------------------------------------------

def two_mirror_response(freqs, R1, R2, L, eps_loss):
    """
    Field-level reflection and transmission of a two-mirror Fabry-Perot cavity.

    Round-trip loss is included as an amplitude survival factor sqrt(1-eps_loss).
    """
    r1, r2 = np.sqrt(R1), np.sqrt(R2)
    t1, t2 = np.sqrt(1.0 - R1), np.sqrt(1.0 - R2)
    a = np.sqrt(1.0 - eps_loss)

    k = 2.0 * np.pi * freqs / c
    e_rt = np.exp(2j * k * L)
    e_one = np.exp(1j * k * L)

    denom = 1.0 - r1 * r2 * a * e_rt
    t_tot = t1 * t2 * np.sqrt(a) * e_one / denom
    r_tot = -r1 + (t1**2 * r2 * a * e_rt) / denom
    return t_tot, r_tot


def two_mirror_properties(R1, R2, L, Rc, eps_loss):
    FSR = c / (2.0 * L)
    gamma_hwhm = (c / (8.0 * np.pi * L)) * ((1.0 - R1) + (1.0 - R2) + eps_loss)
    linewidth_fwhm = 2.0 * gamma_hwhm
    finesse = FSR / linewidth_fwhm if linewidth_fwhm > 0 else np.inf

    g1 = 1.0
    g2 = 1.0 - L / Rc
    g_prod = g1 * g2
    is_stable = 0.0 < g_prod < 1.0

    if is_stable:
        tm_spacing = (FSR / np.pi) * np.arccos(np.sqrt(g_prod))
    else:
        tm_spacing = np.nan

    return {
        "FSR": FSR,
        "gamma_hwhm": gamma_hwhm,
        "linewidth_fwhm": linewidth_fwhm,
        "finesse": finesse,
        "g1": g1,
        "g2": g2,
        "g_prod": g_prod,
        "is_stable": is_stable,
        "tm_spacing": tm_spacing,
    }


# --------------------------------------------------------------------
# Landing page
# --------------------------------------------------------------------

def render_landing():
    st.title("Filter Cavity Designer")
    st.markdown(
        "Choose the cavity geometry you want to model. After selecting, "
        "you will be taken to the parameter input page, then to a results "
        "page with transmission, phase, and cavity property plots."
    )

    st.markdown("### Cavity geometry")
    geom = st.selectbox(
        "Select geometry",
        options=[
            "-- Select --",
            "Two-mirror cavity",
            "Etalon-mirror cavity",
        ],
        index=0,
        key="geometry_dropdown",
    )

    cont_disabled = geom == "-- Select --"
    if st.button("Continue", type="primary", disabled=cont_disabled):
        st.session_state.geometry = geom
        if geom == "Two-mirror cavity":
            go_to("two_mirror_params")
        else:
            go_to("etalon_placeholder")
        st.rerun()


# --------------------------------------------------------------------
# Two-mirror parameter page
# --------------------------------------------------------------------

def render_two_mirror_params():
    st.title("Two-mirror cavity — parameters")
    st.caption(
        "Plano–concave Fabry-Perot cavity with input mirror R₁ and concave "
        "end mirror R₂ separated by length L."
    )

    col_diag, col_input = st.columns([1, 1])

    with col_diag:
        st.markdown("### Cavity diagram")
        diagram_path = "twomirrorcavity.jpg"
        if os.path.exists(diagram_path):
            st.image(diagram_path, caption="Plano–concave two-mirror cavity")
        else:
            st.info("Cavity diagram image not found in working directory.")

    with col_input:
        st.markdown("### Input parameters")

        prev = st.session_state.two_mirror_params or {}

        R1 = st.number_input(
            "R₁ — input mirror reflectivity",
            min_value=0.0,
            max_value=0.999999,
            value=float(prev.get("R1", 0.99)),
            step=1e-4,
            format="%.6f",
        )
        R2 = st.number_input(
            "R₂ — end concave mirror reflectivity",
            min_value=0.0,
            max_value=0.9999999,
            value=float(prev.get("R2", 0.998)),
            step=1e-5,
            format="%.7f",
        )
        L = st.number_input(
            "L — cavity length [m]",
            min_value=0.001,
            max_value=10.0,
            value=float(prev.get("L", 0.15)),
            step=0.001,
            format="%.4f",
        )
        Rc = st.number_input(
            "ROC of R₂ — radius of curvature of end mirror [m]",
            min_value=0.001,
            max_value=1.0e5,
            value=float(prev.get("Rc", 0.2)),
            step=0.001,
            format="%.4f",
        )
        eps = st.number_input(
            "ε — round-trip cavity loss",
            min_value=0.0,
            max_value=0.1,
            value=float(prev.get("eps", 5e-4)),
            format="%.2e",
            help="Effective scatter / absorption loss per round trip.",
        )

    st.markdown("---")
    cnav1, cnav2 = st.columns([1, 1])
    with cnav1:
        if st.button("← Back"):
            go_to("landing")
            st.rerun()
    with cnav2:
        if st.button("Run analysis →", type="primary"):
            st.session_state.two_mirror_params = {
                "R1": R1,
                "R2": R2,
                "L": L,
                "Rc": Rc,
                "eps": eps,
            }
            go_to("two_mirror_results")
            st.rerun()


# --------------------------------------------------------------------
# Two-mirror results page
# --------------------------------------------------------------------

def render_two_mirror_results():
    p = st.session_state.two_mirror_params
    if p is None:
        st.warning("No parameters found. Returning to the input page.")
        go_to("two_mirror_params")
        st.rerun()
        return

    control_col, plot_col, info_col = st.columns([0.7, 3.2, 1.0], gap="medium")

    with control_col:
        R1 = st.slider(
            "R₁ input mirror",
            min_value=0.0,
            max_value=0.999999,
            value=float(p["R1"]),
            step=1e-4,
            format="%.6f",
        )
        R2 = st.slider(
            "R₂ end mirror",
            min_value=0.0,
            max_value=0.999999,
            value=float(min(p["R2"], 0.999999)),
            step=1e-5,
            format="%.6f",
        )
        L = st.slider(
            "L cavity length [m]",
            min_value=0.001,
            max_value=3.0,
            value=float(min(max(p["L"], 0.001), 3.0)),
            step=0.001,
            format="%.4f",
        )
        Rc = st.slider(
            "ROC of R₂ [m]",
            min_value=0.001,
            max_value=5.0,
            value=float(min(max(p["Rc"], 0.001), 5.0)),
            step=0.001,
            format="%.4f",
        )
        eps = st.slider(
            "ε round-trip loss",
            min_value=0.0,
            max_value=0.01,
            value=float(min(max(p["eps"], 0.0), 0.01)),
            step=1e-5,
            format="%.5f",
        )

        x_axis_mode = st.selectbox(
            "Transmission / reflection x-axis",
            options=["Laser detuning Δν", "Cavity length L"],
            index=0,
        )

        st.session_state.two_mirror_params = {
            "R1": R1,
            "R2": R2,
            "L": L,
            "Rc": Rc,
            "eps": eps,
        }

        st.markdown("---")
        if st.button("Edit full parameter page"):
            go_to("two_mirror_params")
            st.rerun()
        if st.button("Start over"):
            go_to("landing")
            st.rerun()

    # Frequency grid
    FSR_Hz = c / (2.0 * L)
    span = 2.0 * FSR_Hz
    N = 4001
    dnu = np.linspace(-span / 2.0, span / 2.0, N)
    freqs = f0 + dnu

    t_tot, r_tot = two_mirror_response(freqs, R1, R2, L, eps)
    T_power = np.abs(t_tot) ** 2
    R_power = np.abs(r_tot) ** 2

    # Length-scan equivalent: span ±λ0/2 around L gives the same round-trip
    # phase range as the detuning scan over 2 FSR at fixed wavelength.
    L_span = lambda0
    L_scan = np.linspace(L - L_span / 2.0, L + L_span / 2.0, N)
    k0 = 2.0 * np.pi * f0 / c
    e_rt_L = np.exp(2j * k0 * L_scan)
    e_one_L = np.exp(1j * k0 * L_scan)
    r1_a, r2_a = np.sqrt(R1), np.sqrt(R2)
    t1_a, t2_a = np.sqrt(1.0 - R1), np.sqrt(1.0 - R2)
    a_loss = np.sqrt(1.0 - eps)
    denom_L = 1.0 - r1_a * r2_a * a_loss * e_rt_L
    t_lenscan = t1_a * t2_a * np.sqrt(a_loss) * e_one_L / denom_L
    r_lenscan = -r1_a + (t1_a**2 * r2_a * a_loss * e_rt_L) / denom_L
    T_lenscan_power = np.abs(t_lenscan) ** 2
    R_lenscan_power = np.abs(r_lenscan) ** 2

    props = two_mirror_properties(R1, R2, L, Rc, eps)

    x_MHz = dnu * 1e-6
    phase_r = np.unwrap(np.angle(r_tot))
    omega = 2.0 * np.pi * freqs
    tau_g = -np.gradient(phase_r, omega)

    use_length_x = x_axis_mode == "Cavity length L"
    if use_length_x:
        top_x = (L_scan - L) * 1e9
        top_T = T_lenscan_power
        top_R = R_lenscan_power
        top_x_title = "ΔL [nm]"
    else:
        top_x = x_MHz
        top_T = T_power
        top_R = R_power
        top_x_title = "Δν [MHz]"

    with plot_col:
        fig_response = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"Power transmission vs {('cavity length' if use_length_x else 'detuning')}",
                f"Power reflection vs {('cavity length' if use_length_x else 'detuning')}",
                "Reflection phase vs detuning",
                "Group delay vs detuning",
            ),
            horizontal_spacing=0.12,
            vertical_spacing=0.22,
        )
        fig_response.add_trace(
            go.Scatter(x=top_x, y=top_T, mode="lines", name="Transmission"),
            row=1,
            col=1,
        )
        fig_response.add_trace(
            go.Scatter(x=top_x, y=top_R, mode="lines", name="Reflection"),
            row=1,
            col=2,
        )
        fig_response.add_trace(
            go.Scatter(x=x_MHz, y=phase_r, mode="lines", name="Phase"),
            row=2,
            col=1,
        )
        fig_response.add_trace(
            go.Scatter(x=x_MHz, y=tau_g, mode="lines", name="Group delay"),
            row=2,
            col=2,
        )
        fig_response.update_xaxes(title_text=top_x_title, row=1, col=1)
        fig_response.update_xaxes(title_text=top_x_title, row=1, col=2)
        fig_response.update_xaxes(title_text="Δν [MHz]", row=2, col=1)
        fig_response.update_xaxes(title_text="Δν [MHz]", row=2, col=2)
        fig_response.update_yaxes(title_text="Power", type="log", row=1, col=1, fixedrange=True)
        fig_response.update_yaxes(title_text="Power", row=1, col=2, fixedrange=True)
        fig_response.update_yaxes(title_text="rad", row=2, col=1, fixedrange=True)
        fig_response.update_yaxes(title_text="s", row=2, col=2, fixedrange=True)
        fig_response.update_layout(
            height=650,
            dragmode="zoom",
            showlegend=False,
            margin=dict(l=35, r=20, t=65, b=35),
            font=dict(family="DejaVu Sans"),
        )
        st.plotly_chart(fig_response, use_container_width=True)

    with info_col:
        st.metric("FSR", f"{props['FSR']/1e6:.3f} MHz")
        st.metric("Linewidth (FWHM)", f"{props['linewidth_fwhm']/1e3:.3f} kHz")
        st.metric("Finesse", f"{props['finesse']:.1f}")
        if np.isfinite(props["tm_spacing"]):
            st.metric("Transverse spacing", f"{props['tm_spacing']/1e6:.3f} MHz")
        else:
            st.metric("Transverse spacing", "—")

        if props["is_stable"]:
            st.success("Stable: 0 < g₁·g₂ < 1")
        else:
            st.error("Unstable: g₁·g₂ outside (0, 1)")

        if 0.0 < L < Rc:
            zR = float(np.sqrt(L * (Rc - L)))
            w0 = float(np.sqrt(lambda0 * zR / np.pi))
            st.markdown(f"**w₀** = {w0*1e6:.2f} µm")
            st.markdown(
                f"**z<sub>R</sub>** = {zR*1e3:.2f} mm",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("**w₀** = —")
            st.markdown("**z<sub>R</sub>** = —", unsafe_allow_html=True)

        if np.isfinite(props["tm_spacing"]):
            ratio = props["tm_spacing"] / props["linewidth_fwhm"]
            if ratio >= 5.0:
                st.success(f"HOM spacing / linewidth = {ratio:.2f} (≥ 5)")
            else:
                st.warning(f"HOM spacing / linewidth = {ratio:.2f} (< 5)")
        else:
            st.warning("HOM spacing check unavailable.")

    with st.expander("Derivations and definitions", expanded=False):
        st.subheader("Derivations and definitions")

        st.markdown("#### Field model")
        st.latex(r"k = \frac{2\pi f}{c}, \qquad \theta = 2 k L")
        st.latex(
            r"""
t_{\mathrm{tot}}(f) = \frac{t_1 t_2 \sqrt{1-\epsilon}\; e^{i k L}}
{1 - r_1 r_2 \sqrt{1-\epsilon}\; e^{i\theta}}
"""
        )
        st.latex(
            r"""
r_{\mathrm{tot}}(f) = -r_1 + \frac{t_1^2 r_2 \sqrt{1-\epsilon}\; e^{i\theta}}
{1 - r_1 r_2 \sqrt{1-\epsilon}\; e^{i\theta}}
"""
        )
        st.markdown(
            "All $r_i$, $t_i$ are amplitude reflectivities and transmissivities, "
            "with $R_i = r_i^2$ and $1-R_i = t_i^2$."
        )

        st.markdown("#### Free spectral range")
        st.latex(r"\mathrm{FSR} = \frac{c}{2 L}")

        st.markdown("#### Cavity pole and linewidth")
        st.latex(
            r"""
\gamma = \frac{c}{8\pi L}\left[(1-R_1) + (1-R_2) + \epsilon\right]
"""
        )
        st.latex(r"\Delta\nu_{\mathrm{cav}} = 2\gamma")
        st.latex(r"\mathcal{F} = \frac{\mathrm{FSR}}{\Delta\nu_{\mathrm{cav}}}")

        st.markdown("#### Geometric stability (g-factors)")
        st.latex(
            r"""
g_1 = 1, \qquad g_2 = 1 - \frac{L}{R_c}, \qquad 0 < g_1 g_2 < 1
"""
        )

        st.markdown("#### Transverse mode spacing")
        st.latex(
            r"""
\Delta\nu_{\perp} = \frac{\mathrm{FSR}}{\pi}\,\cos^{-1}\!\left(\sqrt{g_1 g_2}\right)
"""
        )

        st.markdown("#### Higher-order mode separation criterion")
        st.latex(r"\Delta\nu_{\perp} \ge 5\,\Delta\nu_{\mathrm{cav}}")

        st.markdown("#### Reflection phase and group delay")
        st.latex(r"\phi(f) = \arg\,r_{\mathrm{tot}}(f)")
        st.latex(r"\tau_g(f) = -\frac{d\phi}{d\omega}")


# --------------------------------------------------------------------
# Etalon placeholder page
# --------------------------------------------------------------------

def render_etalon_placeholder():
    st.title("Etalon-mirror cavity")
    st.info(
        "Etalon-mirror cavity dynamics will be implemented in the next step. "
        "Use the back button to return to the landing page."
    )
    if st.button("← Back"):
        go_to("landing")
        st.rerun()


# --------------------------------------------------------------------
# Page router
# --------------------------------------------------------------------

page = st.session_state.page
if page == "landing":
    render_landing()
elif page == "two_mirror_params":
    render_two_mirror_params()
elif page == "two_mirror_results":
    render_two_mirror_results()
elif page == "etalon_placeholder":
    render_etalon_placeholder()
else:
    st.error(f"Unknown page: {page}")
    if st.button("Reset"):
        go_to("landing")
        st.rerun()
