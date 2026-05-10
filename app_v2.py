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
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0.4rem !important;
        padding-left: 1.0rem !important;
        padding-right: 1.0rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
        height: 0rem;
        min-height: 0rem;
        background: transparent;
    }
    /* Compact slider + number_input rows for the dashboard controls */
    [data-testid="stSlider"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-bottom: -0.6rem !important;
    }
    [data-testid="stSlider"] > label {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stNumberInput"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-bottom: -0.4rem !important;
    }
    [data-testid="stNumberInput"] input {
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
        font-size: 0.85rem !important;
    }
    .synced-spacer { height: 1.45rem; }
    /* Mini-controls used in the right info column for the thermal scan */
    .mini-controls [data-testid="stNumberInput"] input {
        font-size: 0.72rem !important;
        padding-top: 0.1rem !important;
        padding-bottom: 0.1rem !important;
    }
    .mini-controls [data-testid="stNumberInput"] label,
    .mini-controls label {
        font-size: 0.72rem !important;
        margin-bottom: 0 !important;
    }
    .mini-controls h6, .mini-controls .mini-title {
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.15rem 0 !important;
    }
    /* Compact selectbox used inline above the plots */
    .inline-xaxis [data-testid="stSelectbox"] {
        margin-bottom: -0.2rem !important;
    }
    .inline-xaxis [data-testid="stSelectbox"] label {
        font-size: 0.78rem !important;
        margin-bottom: 0 !important;
    }
    .inline-xaxis [data-baseweb="select"] {
        font-size: 0.8rem !important;
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
if "etalon_params" not in st.session_state:
    st.session_state.etalon_params = None


def go_to(page_name):
    st.session_state.page = page_name


def synced_input(
    label,
    min_value,
    max_value,
    default,
    step,
    key_base,
    fmt=None,
    integer=False,
    slider_ratio=3,
    num_ratio=2,
):
    """
    Render a slider and a number-input that share a value.

    Both widgets stay in sync via on_change callbacks; the function returns
    the current value as float (or int when ``integer=True``).
    """
    slider_key = f"{key_base}_slider"
    num_key = f"{key_base}_num"

    if integer:
        cast = int
    else:
        cast = float

    if slider_key not in st.session_state:
        st.session_state[slider_key] = cast(default)
    if num_key not in st.session_state:
        st.session_state[num_key] = cast(default)

    def _on_slider():
        st.session_state[num_key] = st.session_state[slider_key]

    def _on_num():
        v = cast(st.session_state[num_key])
        v = max(cast(min_value), min(cast(max_value), v))
        st.session_state[slider_key] = v

    cs, cn = st.columns([slider_ratio, num_ratio], gap="small")
    with cs:
        if integer:
            st.slider(
                label,
                min_value=int(min_value),
                max_value=int(max_value),
                step=int(step),
                key=slider_key,
                on_change=_on_slider,
            )
        else:
            st.slider(
                label,
                min_value=float(min_value),
                max_value=float(max_value),
                step=float(step),
                format=fmt,
                key=slider_key,
                on_change=_on_slider,
            )
    with cn:
        st.markdown(
            '<div class="synced-spacer"></div>', unsafe_allow_html=True
        )
        if integer:
            st.number_input(
                label,
                min_value=int(min_value),
                max_value=int(max_value),
                step=int(step),
                key=num_key,
                label_visibility="collapsed",
                on_change=_on_num,
            )
        else:
            st.number_input(
                label,
                min_value=float(min_value),
                max_value=float(max_value),
                step=float(step),
                format=fmt,
                key=num_key,
                label_visibility="collapsed",
                on_change=_on_num,
            )

    return st.session_state[slider_key]


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
# Etalon-mirror cavity physics
# --------------------------------------------------------------------

def etalon_coeffs(r1, r2, t1, t2, k_val, n_substrate, d):
    """Field-level etalon transmission and reflections."""
    delta = 2.0 * k_val * n_substrate * d
    exp_i_delta = np.exp(1j * delta)
    denom = 1.0 - r1 * r2 * exp_i_delta
    t_et = t1 * t2 * np.exp(1j * delta / 2.0) / denom
    r_et_L = -r1 + (t1**2 * r2 * exp_i_delta) / denom
    r_et_R = r2 + (t2**2 * r1 * exp_i_delta) / denom
    return t_et, r_et_L, r_et_R


def three_surface_response(freqs, R1, R2, R3, L1, L2, n_substrate):
    """
    Field-level reflection and transmission of an etalon (R1, R2, n, L1) +
    long cavity (L2) + concave end mirror (R3).
    """
    r1, r2, r3 = np.sqrt(R1), np.sqrt(R2), np.sqrt(R3)
    t1, t2, t3 = np.sqrt(1.0 - R1), np.sqrt(1.0 - R2), np.sqrt(1.0 - R3)

    k_vals = 2.0 * np.pi * freqs / c
    theta = 2.0 * k_vals * L2

    t_et, r_et_L, r_et_R = etalon_coeffs(
        r1, r2, t1, t2, k_vals, n_substrate, L1
    )

    denom = 1.0 - r_et_R * r3 * np.exp(1j * theta)
    r_tot = r_et_L + (t_et**2 * r3 * np.exp(1j * theta)) / denom
    t_tot = (t_et * t3 * np.exp(1j * k_vals * L2)) / denom
    return r_tot, t_tot


def Teff_etalon(phi, R1, R2):
    sqrtR = np.sqrt(R1 * R2)
    num = (1.0 - R1) * (1.0 - R2)
    den = 1.0 - 2.0 * sqrtR * np.cos(phi) + R1 * R2
    return num / den


def dTeff_dphi(phi, R1, R2):
    sqrtR = np.sqrt(R1 * R2)
    num = (1.0 - R1) * (1.0 - R2)
    den = 1.0 - 2.0 * sqrtR * np.cos(phi) + R1 * R2
    return -(num * (2.0 * sqrtR * np.sin(phi))) / (den**2)


def etalon_phi(L1, n_substrate):
    return 4.0 * np.pi * n_substrate * L1 / lambda0


def etalon_pole_gamma(phi, R1, R2, R3, L2, eps_loss):
    return (c / (8.0 * np.pi * L2)) * (
        Teff_etalon(phi, R1, R2) + (1.0 - R3) + eps_loss
    )


def etalon_dphi_dT(L1, n_substrate, alpha_ppm, dn_dT_val):
    alpha = alpha_ppm * 1e-6
    return (4.0 * np.pi * L1 / lambda0) * (n_substrate * alpha + dn_dT_val)


def etalon_tunability_dgamma_dT(
    phi, L1, R1, R2, L2, n_substrate, alpha_ppm, dn_dT_val
):
    return (
        (c / (8.0 * np.pi * L2))
        * dTeff_dphi(phi, R1, R2)
        * etalon_dphi_dT(L1, n_substrate, alpha_ppm, dn_dT_val)
    )


def etalon_properties(
    L1, L2, R1, R2, R3, Rc, eps_loss, n_substrate, alpha_ppm, dn_dT_val
):
    phi = etalon_phi(L1, n_substrate)
    Teff_val = Teff_etalon(phi, R1, R2)
    gamma_hwhm = etalon_pole_gamma(phi, R1, R2, R3, L2, eps_loss)
    linewidth_fwhm = 2.0 * gamma_hwhm
    FSR = c / (2.0 * L2)
    finesse = FSR / linewidth_fwhm if linewidth_fwhm > 0 else np.inf

    L_eff = L2 + n_substrate * L1
    g1 = 1.0
    g2 = 1.0 - L_eff / Rc
    g_prod = g1 * g2
    is_stable = 0.0 < g_prod < 1.0
    tm_spacing = (
        (FSR / np.pi) * np.arccos(np.sqrt(g_prod)) if is_stable else np.nan
    )

    dgamma_dT = etalon_tunability_dgamma_dT(
        phi, L1, R1, R2, L2, n_substrate, alpha_ppm, dn_dT_val
    )

    return {
        "phi": phi,
        "Teff": Teff_val,
        "FSR": FSR,
        "gamma_hwhm": gamma_hwhm,
        "linewidth_fwhm": linewidth_fwhm,
        "finesse": finesse,
        "L_eff": L_eff,
        "g1": g1,
        "g2": g2,
        "g_prod": g_prod,
        "is_stable": is_stable,
        "tm_spacing": tm_spacing,
        "dgamma_dT": dgamma_dT,
    }


def three_surface_length_scan(L_scan, R1, R2, R3, L1, n_substrate):
    """Reflection and transmission with the long cavity length swept at f0."""
    r1, r2, r3 = np.sqrt(R1), np.sqrt(R2), np.sqrt(R3)
    t1, t2, t3 = np.sqrt(1.0 - R1), np.sqrt(1.0 - R2), np.sqrt(1.0 - R3)

    k0 = 2.0 * np.pi * f0 / c
    delta = 2.0 * k0 * n_substrate * L1
    exp_i_delta = np.exp(1j * delta)
    denom_et = 1.0 - r1 * r2 * exp_i_delta
    t_et = t1 * t2 * np.exp(1j * delta / 2.0) / denom_et
    r_et_L = r1 + (t1**2 * r2 * exp_i_delta) / denom_et
    r_et_R = r2 + (t2**2 * r1 * exp_i_delta) / denom_et

    theta = 2.0 * k0 * L_scan
    denom_arr = 1.0 - r_et_R * r3 * np.exp(1j * theta)
    r_tot = r_et_L + (t_et**2 * r3 * np.exp(1j * theta)) / denom_arr
    t_tot = (t_et * t3 * np.exp(1j * k0 * L_scan)) / denom_arr
    return r_tot, t_tot


# --------------------------------------------------------------------
# Landing page
# --------------------------------------------------------------------

def render_landing():
    st.title("Filter Cavity Designer")

    st.markdown("### Cavity model")
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
            go_to("etalon_params")
        st.rerun()


# --------------------------------------------------------------------
# Two-mirror parameter page
# --------------------------------------------------------------------

def render_two_mirror_params():
    st.caption(
        "Plano–concave Fabry-Perot cavity with input mirror R₁ and concave "
        "end mirror R₂ separated by length L."
    )

    col_diag, col_input = st.columns([1, 1])

    with col_diag:
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

    control_col, plot_col, info_col = st.columns([1.0, 3.0, 1.0], gap="medium")

    with control_col:
        R1 = synced_input(
            "R₁ input mirror",
            0.0, 0.999999, float(p["R1"]),
            1e-4, "tm_R1", fmt="%.6f",
        )
        R2 = synced_input(
            "R₂ end mirror",
            0.0, 0.999999, float(min(p["R2"], 0.999999)),
            1e-5, "tm_R2", fmt="%.6f",
        )
        L = synced_input(
            "L cavity length [m]",
            0.001, 3.0, float(min(max(p["L"], 0.001), 3.0)),
            0.001, "tm_L", fmt="%.4f",
        )
        Rc = synced_input(
            "ROC of R₂ [m]",
            0.001, 5.0, float(min(max(p["Rc"], 0.001), 5.0)),
            0.001, "tm_Rc", fmt="%.4f",
        )
        eps = synced_input(
            "ε round-trip loss",
            0.0, 0.01, float(min(max(p["eps"], 0.0), 0.01)),
            1e-5, "tm_eps", fmt="%.5f",
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
# Etalon parameter page
# --------------------------------------------------------------------

ETALON_DEFAULTS = {
    "L1_mm": 6.0,
    "L2": 0.15,
    "R1": 0.90,
    "R2": 0.90,
    "R3": 0.998,
    "Rc": 0.20,
    "eps": 5e-4,
    "n": 1.45,
    "alpha_ppm": 0.55,
    "dn_dT": 8.62e-6,
    "dT_max": 10.0,
    "N_samples": 5,
}


def _ep(key):
    """Etalon parameter getter with default fallback."""
    prev = st.session_state.etalon_params or {}
    return prev.get(key, ETALON_DEFAULTS[key])


def render_etalon_params():
    st.title("Etalon-mirror cavity — parameters")
    st.caption(
        "Three-surface filter cavity: solid-substrate etalon (R₁, R₂, n, L₁) + "
        "concave end mirror (R₃) separated by long cavity length L₂."
    )

    col_diag, col_input = st.columns([1, 1])

    with col_diag:
        diagram_path = "etalonmirrorcavity.jpg"
        if os.path.exists(diagram_path):
            st.image(diagram_path, caption="Etalon + concave end-mirror cavity")
        else:
            st.info("Cavity diagram image not found in working directory.")

    with col_input:
        st.markdown("### Optical parameters")
        opt_a, opt_b = st.columns(2, gap="small")
        with opt_a:
            L1_mm = st.number_input(
                "L₁ — substrate thickness [mm]",
                min_value=0.001,
                max_value=50.0,
                value=float(_ep("L1_mm")),
                step=0.01,
                format="%.4f",
            )
            R1 = st.number_input(
                "R₁ — etalon front",
                min_value=0.0,
                max_value=0.999999,
                value=float(_ep("R1")),
                step=1e-4,
                format="%.6f",
            )
            R3 = st.number_input(
                "R₃ — end mirror",
                min_value=0.5,
                max_value=0.9999999,
                value=float(_ep("R3")),
                step=1e-5,
                format="%.7f",
            )
            eps = st.number_input(
                "ε — round-trip loss",
                min_value=0.0,
                max_value=0.1,
                value=float(_ep("eps")),
                format="%.2e",
            )
        with opt_b:
            L2 = st.number_input(
                "L₂ — long cavity length [m]",
                min_value=0.001,
                max_value=10.0,
                value=float(_ep("L2")),
                step=0.001,
                format="%.4f",
            )
            R2 = st.number_input(
                "R₂ — etalon back",
                min_value=0.0,
                max_value=0.999999,
                value=float(_ep("R2")),
                step=1e-4,
                format="%.6f",
            )
            Rc = st.number_input(
                "ROC of R₃ [m]",
                min_value=0.001,
                max_value=1.0e5,
                value=float(_ep("Rc")),
                step=0.001,
                format="%.4f",
            )

        st.markdown("### Etalon substrate properties")
        sub_a, sub_b = st.columns(2, gap="small")
        with sub_a:
            n_substrate = st.number_input(
                "n — refractive index",
                min_value=1.0,
                max_value=4.0,
                value=float(_ep("n")),
                step=0.01,
            )
            dn_dT_val = st.number_input(
                "dn/dT [1/K]",
                value=float(_ep("dn_dT")),
                format="%.2e",
            )
        with sub_b:
            alpha_ppm = st.number_input(
                "dL₁/dT [ppm/K]",
                min_value=0.0,
                max_value=100.0,
                value=float(_ep("alpha_ppm")),
                step=0.01,
            )

        st.markdown("### Thermal scan")
        ts_a, ts_b = st.columns(2, gap="small")
        with ts_a:
            dT_max = st.number_input(
                "ΔTₘₐₓ [°C]",
                min_value=0.0,
                max_value=1000.0,
                value=float(_ep("dT_max")),
                step=0.5,
            )
        with ts_b:
            N_samples = st.number_input(
                "# T samples",
                min_value=1,
                max_value=30,
                value=int(_ep("N_samples")),
                step=1,
            )

    st.markdown("---")
    cnav1, cnav2 = st.columns([1, 1])
    with cnav1:
        if st.button("← Back"):
            go_to("landing")
            st.rerun()
    with cnav2:
        if st.button("Run analysis →", type="primary"):
            st.session_state.etalon_params = {
                "L1_mm": L1_mm,
                "L2": L2,
                "R1": R1,
                "R2": R2,
                "R3": R3,
                "Rc": Rc,
                "eps": eps,
                "n": n_substrate,
                "alpha_ppm": alpha_ppm,
                "dn_dT": dn_dT_val,
                "dT_max": dT_max,
                "N_samples": int(N_samples),
            }
            go_to("etalon_results")
            st.rerun()


# --------------------------------------------------------------------
# Etalon results dashboard
# --------------------------------------------------------------------

def render_etalon_results():
    p = st.session_state.etalon_params
    if p is None:
        st.warning("No parameters found. Returning to the input page.")
        go_to("etalon_params")
        st.rerun()
        return

    control_col, plot_col, info_col = st.columns([1.0, 3.0, 1.0], gap="small")

    with control_col:
        st.markdown("##### Optical")
        L1_mm = synced_input(
            "L₁ [mm]",
            0.1, 50.0, float(min(max(p["L1_mm"], 0.1), 50.0)),
            0.01, "et_L1mm", fmt="%.4f",
        )
        L2 = synced_input(
            "L₂ [m]",
            0.01, 3.0, float(min(max(p["L2"], 0.01), 3.0)),
            0.001, "et_L2", fmt="%.4f",
        )
        R1 = synced_input(
            "R₁ etalon front",
            0.0, 0.999999, float(min(max(p["R1"], 0.0), 0.999999)),
            1e-4, "et_R1", fmt="%.6f",
        )
        R2 = synced_input(
            "R₂ etalon back",
            0.0, 0.999999, float(min(max(p["R2"], 0.0), 0.999999)),
            1e-4, "et_R2", fmt="%.6f",
        )
        R3 = synced_input(
            "R₃ end mirror",
            0.5, 0.9999999, float(min(max(p["R3"], 0.5), 0.9999999)),
            1e-5, "et_R3", fmt="%.7f",
        )
        Rc = synced_input(
            "ROC of R₃ [m]",
            0.01, 5.0, float(min(max(p["Rc"], 0.01), 5.0)),
            0.001, "et_Rc", fmt="%.4f",
        )
        eps = synced_input(
            "ε round-trip loss",
            0.0, 0.01, float(min(max(p["eps"], 0.0), 0.01)),
            1e-5, "et_eps", fmt="%.5f",
        )

        st.markdown("##### Substrate")
        n_substrate = synced_input(
            "n",
            1.0, 4.0, float(min(max(p["n"], 1.0), 4.0)),
            0.01, "et_n", fmt="%.4f",
        )
        alpha_ppm = synced_input(
            "dL₁/dT [ppm/K]",
            0.0, 100.0, float(min(max(p["alpha_ppm"], 0.0), 100.0)),
            0.01, "et_alpha", fmt="%.4f",
        )
        dn_dT_val = st.number_input(
            "dn/dT [1/K]",
            value=float(p["dn_dT"]),
            format="%.2e",
        )

        st.markdown("---")
        if st.button("Edit full parameter page"):
            go_to("etalon_params")
            st.rerun()
        if st.button("Start over"):
            go_to("landing")
            st.rerun()

    with info_col:
        st.markdown(
            '<div class="mini-controls">'
            '<div class="mini-title">Thermal scan</div>',
            unsafe_allow_html=True,
        )
        col_dT, col_N = st.columns(2, gap="small")
        with col_dT:
            dT_max = st.number_input(
                "ΔTₘₐₓ [°C]",
                min_value=0.0,
                max_value=1000.0,
                value=float(p["dT_max"]),
                step=0.5,
                format="%.2f",
                key="et_dTmax_mini",
            )
        with col_N:
            N_samples = st.number_input(
                "# T samples",
                min_value=1,
                max_value=30,
                value=int(p["N_samples"]),
                step=1,
                key="et_Nsamples_mini",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with plot_col:
        st.markdown('<div class="inline-xaxis">', unsafe_allow_html=True)
        x_axis_mode = st.selectbox(
            "Transmission / reflection x-axis",
            options=["Laser detuning Δν", "Cavity length L₂"],
            index=0,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.etalon_params = {
        "L1_mm": L1_mm,
        "L2": L2,
        "R1": R1,
        "R2": R2,
        "R3": R3,
        "Rc": Rc,
        "eps": eps,
        "n": n_substrate,
        "alpha_ppm": alpha_ppm,
        "dn_dT": dn_dT_val,
        "dT_max": dT_max,
        "N_samples": int(N_samples),
    }

    L1 = L1_mm * 1e-3
    alpha = alpha_ppm * 1e-6
    dL1_dT_geom = alpha * L1

    FSR_Hz = c / (2.0 * L2)
    span = 2.0 * FSR_Hz
    N = 4001
    dnu = np.linspace(-span / 2.0, span / 2.0, N)
    freqs = f0 + dnu

    if N_samples > 1:
        delta_T_arr = np.linspace(0.0, dT_max, int(N_samples))
    else:
        delta_T_arr = np.array([0.0])

    temp_curves = []
    for dT in delta_T_arr:
        L1_T = L1 + dL1_dT_geom * dT
        n_T = n_substrate + dn_dT_val * dT
        r_tot_T, t_tot_T = three_surface_response(
            freqs, R1, R2, R3, L1_T, L2, n_T
        )
        T_T = np.abs(t_tot_T) ** 2
        R_T = np.abs(r_tot_T) ** 2
        phase_T = np.unwrap(np.angle(r_tot_T))
        tau_g_T = -np.gradient(phase_T, 2.0 * np.pi * freqs)
        temp_curves.append((dT, T_T, R_T, phase_T, tau_g_T))

    L_span = lambda0
    L_scan = np.linspace(L2 - L_span / 2.0, L2 + L_span / 2.0, N)
    len_curves = []
    for dT in delta_T_arr:
        L1_T = L1 + dL1_dT_geom * dT
        n_T = n_substrate + dn_dT_val * dT
        r_lenscan_T, t_lenscan_T = three_surface_length_scan(
            L_scan, R1, R2, R3, L1_T, n_T
        )
        T_lenscan_T = np.abs(t_lenscan_T) ** 2
        R_lenscan_T = np.abs(r_lenscan_T) ** 2
        len_curves.append((dT, T_lenscan_T, R_lenscan_T))

    props = etalon_properties(
        L1, L2, R1, R2, R3, Rc, eps, n_substrate, alpha_ppm, dn_dT_val
    )

    use_length_x = x_axis_mode == "Cavity length L₂"
    x_MHz = dnu * 1e-6

    if use_length_x:
        top_x = (L_scan - L2) * 1e9
        top_x_title = "ΔL₂ [nm]"
    else:
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

        if use_length_x:
            for dT, T_lenscan_T, R_lenscan_T in len_curves:
                fig_response.add_trace(
                    go.Scatter(
                        x=top_x,
                        y=T_lenscan_T,
                        mode="lines",
                        name=f"ΔT={dT:.1f} °C",
                    ),
                    row=1,
                    col=1,
                )
                fig_response.add_trace(
                    go.Scatter(
                        x=top_x,
                        y=R_lenscan_T,
                        mode="lines",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
        else:
            for dT, T_T, R_T, _, _ in temp_curves:
                fig_response.add_trace(
                    go.Scatter(
                        x=x_MHz,
                        y=T_T,
                        mode="lines",
                        name=f"ΔT={dT:.1f} °C",
                    ),
                    row=1,
                    col=1,
                )
                fig_response.add_trace(
                    go.Scatter(
                        x=x_MHz,
                        y=R_T,
                        mode="lines",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

        for dT, _, _, phase_T, tau_g_T in temp_curves:
            fig_response.add_trace(
                go.Scatter(x=x_MHz, y=phase_T, mode="lines", showlegend=False),
                row=2,
                col=1,
            )
            fig_response.add_trace(
                go.Scatter(x=x_MHz, y=tau_g_T, mode="lines", showlegend=False),
                row=2,
                col=2,
            )

        fig_response.update_xaxes(title_text=top_x_title, row=1, col=1)
        fig_response.update_xaxes(title_text=top_x_title, row=1, col=2)
        fig_response.update_xaxes(title_text="Δν [MHz]", row=2, col=1)
        fig_response.update_xaxes(title_text="Δν [MHz]", row=2, col=2)
        fig_response.update_yaxes(
            title_text="Power", type="log", row=1, col=1, fixedrange=True
        )
        fig_response.update_yaxes(title_text="Power", row=1, col=2, fixedrange=True)
        fig_response.update_yaxes(title_text="rad", row=2, col=1, fixedrange=True)
        fig_response.update_yaxes(title_text="s", row=2, col=2, fixedrange=True)
        fig_response.update_layout(
            height=650,
            dragmode="zoom",
            showlegend=True,
            legend=dict(orientation="v", x=1.02, y=1.0, font=dict(size=10)),
            margin=dict(l=35, r=20, t=65, b=35),
        )
        st.plotly_chart(fig_response, use_container_width=True)

    with info_col:
        st.metric("FSR", f"{props['FSR']/1e6:.3f} MHz")
        st.metric(
            "Linewidth (FWHM)", f"{props['linewidth_fwhm']/1e3:.3f} kHz"
        )
        st.metric("Finesse", f"{props['finesse']:.1f}")
        st.metric("T_eff", f"{props['Teff']*1e6:.1f} ppm")
        st.metric("dγ/dT", f"{props['dgamma_dT']/1e3:.2f} kHz/°C")
        if np.isfinite(props["tm_spacing"]):
            st.metric(
                "Transverse spacing", f"{props['tm_spacing']/1e6:.3f} MHz"
            )
        else:
            st.metric("Transverse spacing", "—")

        if props["is_stable"]:
            st.success("Stable: 0 < g₁·g₂ < 1")
        else:
            st.error("Unstable: g₁·g₂ outside (0, 1)")

        L_eff = props["L_eff"]
        if 0.0 < L_eff < Rc:
            zR = float(np.sqrt(L_eff * (Rc - L_eff)))
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
        st.subheader("Etalon-mirror cavity")

        st.markdown("#### Etalon round-trip phase")
        st.latex(r"\phi = \frac{4\pi n L_1}{\lambda_0}")

        st.markdown("#### Effective etalon transmission")
        st.latex(
            r"""
T_{\mathrm{eff}}(\phi)
= \frac{(1-R_1)(1-R_2)}{1 - 2\sqrt{R_1 R_2}\cos\phi + R_1 R_2}
"""
        )

        st.markdown("#### Three-surface field model")
        st.latex(r"k = \frac{2\pi f}{c}, \quad \delta = 2 k n L_1, \quad \theta = 2 k L_2")
        st.latex(
            r"""
t_{\mathrm{et}} = \frac{t_1 t_2 e^{i\delta/2}}{1 - r_1 r_2 e^{i\delta}},\quad
r_{\mathrm{et,L}} = r_1 + \frac{t_1^2 r_2 e^{i\delta}}{1 - r_1 r_2 e^{i\delta}},\quad
r_{\mathrm{et,R}} = r_2 + \frac{t_2^2 r_1 e^{i\delta}}{1 - r_1 r_2 e^{i\delta}}
"""
        )
        st.latex(
            r"""
r_{\mathrm{tot}}(f) = r_{\mathrm{et,L}} + \frac{t_{\mathrm{et}}^2 r_3 e^{i\theta}}{1 - r_{\mathrm{et,R}} r_3 e^{i\theta}}
"""
        )
        st.latex(
            r"""
t_{\mathrm{tot}}(f) = \frac{t_{\mathrm{et}} t_3 e^{i k L_2}}{1 - r_{\mathrm{et,R}} r_3 e^{i\theta}}
"""
        )

        st.markdown("#### Cavity pole and linewidth")
        st.latex(
            r"""
\gamma = \frac{c}{8\pi L_2}\left[T_{\mathrm{eff}}(\phi) + (1-R_3) + \epsilon\right]
"""
        )
        st.latex(r"\Delta\nu_{\mathrm{cav}} = 2\gamma, \qquad \mathcal{F} = \frac{\mathrm{FSR}}{\Delta\nu_{\mathrm{cav}}}")

        st.markdown("#### Thermal tuning")
        st.latex(
            r"""
\frac{d\phi}{dT} = \frac{4\pi L_1}{\lambda_0}\left(n\alpha + \frac{dn}{dT}\right)
"""
        )
        st.latex(
            r"""
\frac{d\gamma}{dT} = \frac{c}{8\pi L_2}\,\frac{dT_{\mathrm{eff}}}{d\phi}\,\frac{d\phi}{dT}
"""
        )
        st.latex(
            r"""
\frac{dT_{\mathrm{eff}}}{d\phi} = -\frac{(1-R_1)(1-R_2)\,2\sqrt{R_1R_2}\sin\phi}{\left(1-2\sqrt{R_1R_2}\cos\phi + R_1R_2\right)^2}
"""
        )

        st.markdown("#### Geometric stability and transverse modes")
        st.latex(
            r"""
L_{\mathrm{eff}} = L_2 + n L_1, \qquad
g_1 = 1, \qquad g_2 = 1 - \frac{L_{\mathrm{eff}}}{R_c}
"""
        )
        st.latex(
            r"""
\Delta\nu_{\perp} = \frac{\mathrm{FSR}}{\pi}\cos^{-1}\!\left(\sqrt{g_1 g_2}\right)
"""
        )
        st.latex(r"\Delta\nu_{\perp} \ge 5\,\Delta\nu_{\mathrm{cav}}")

        st.markdown("#### Beam waist")
        st.latex(
            r"""
z_R = \sqrt{L_{\mathrm{eff}}(R_c - L_{\mathrm{eff}})}, \qquad
w_0 = \sqrt{\frac{\lambda_0\, z_R}{\pi}}
"""
        )


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
elif page == "etalon_params":
    render_etalon_params()
elif page == "etalon_results":
    render_etalon_results()
else:
    st.error(f"Unknown page: {page}")
    if st.button("Reset"):
        go_to("landing")
        st.rerun()
