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
        diagram_path = "planoconcavecavity.png"
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

    R1, R2, L, Rc, eps = p["R1"], p["R2"], p["L"], p["Rc"], p["eps"]

    st.title("Two-mirror cavity — results")
    st.caption(
        f"R₁ = {R1:.6f}   R₂ = {R2:.7f}   L = {L:.4f} m   "
        f"ROC = {Rc:.4f} m   ε = {eps:.2e}"
    )

    # Frequency grid
    FSR_Hz = c / (2.0 * L)
    span = 2.0 * FSR_Hz
    N = 4001
    dnu = np.linspace(-span / 2.0, span / 2.0, N)
    freqs = f0 + dnu

    t_tot, r_tot = two_mirror_response(freqs, R1, R2, L, eps)
    T_power = np.abs(t_tot) ** 2
    R_power = np.abs(r_tot) ** 2

    props = two_mirror_properties(R1, R2, L, Rc, eps)

    tab_T, tab_phase, tab_props, tab_math = st.tabs(
        ["Transmission", "Phase response", "Cavity properties", "Derivations"]
    )

    # ---------- Transmission tab ----------
    with tab_T:
        st.subheader("Power transmission")
        x_MHz = dnu * 1e-6

        fig_T = go.Figure()
        fig_T.add_trace(
            go.Scatter(x=x_MHz, y=T_power, mode="lines", name="|t_tot|²")
        )
        fig_T.update_layout(
            xaxis_title="Frequency detuning Δν [MHz]",
            yaxis_title="Transmitted power [arb.]",
            yaxis_type="log",
            dragmode="zoom",
            yaxis=dict(fixedrange=True),
        )
        st.plotly_chart(fig_T, use_container_width=True)

        st.subheader("Power reflection")
        fig_R = go.Figure()
        fig_R.add_trace(
            go.Scatter(x=x_MHz, y=R_power, mode="lines", name="|r_tot|²")
        )
        fig_R.update_layout(
            xaxis_title="Frequency detuning Δν [MHz]",
            yaxis_title="Reflected power [arb.]",
            dragmode="zoom",
            yaxis=dict(fixedrange=True),
        )
        st.plotly_chart(fig_R, use_container_width=True)

    # ---------- Phase tab ----------
    with tab_phase:
        st.subheader("Reflection phase and group delay")
        x_MHz = dnu * 1e-6
        phase_r = np.unwrap(np.angle(r_tot))
        omega = 2.0 * np.pi * freqs
        tau_g = -np.gradient(phase_r, omega)

        fig_phase = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Reflection phase vs frequency detuning",
                "Group delay vs frequency detuning",
            ),
            horizontal_spacing=0.12,
        )
        fig_phase.add_trace(
            go.Scatter(x=x_MHz, y=phase_r, mode="lines", name="arg r_tot"),
            row=1,
            col=1,
        )
        fig_phase.add_trace(
            go.Scatter(x=x_MHz, y=tau_g, mode="lines", name="τ_g"),
            row=1,
            col=2,
        )
        fig_phase.update_xaxes(title_text="Δν [MHz]", row=1, col=1)
        fig_phase.update_xaxes(title_text="Δν [MHz]", row=1, col=2)
        fig_phase.update_yaxes(title_text="Phase [rad]", row=1, col=1, fixedrange=True)
        fig_phase.update_yaxes(title_text="Group delay [s]", row=1, col=2, fixedrange=True)
        fig_phase.update_layout(height=460, dragmode="zoom", showlegend=False)
        st.plotly_chart(fig_phase, use_container_width=True)

    # ---------- Cavity properties tab ----------
    with tab_props:
        st.subheader("Cavity properties")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Free spectral range (FSR)", f"{props['FSR']/1e6:.3f} MHz")
            st.metric("Cavity pole γ (HWHM)", f"{props['gamma_hwhm']/1e3:.3f} kHz")
            st.metric("Cavity linewidth (FWHM)", f"{props['linewidth_fwhm']/1e3:.3f} kHz")
            st.metric("Finesse", f"{props['finesse']:.1f}")
        with col_b:
            st.metric("g₁", f"{props['g1']:.4f}")
            st.metric("g₂", f"{props['g2']:.4f}")
            st.metric("g₁·g₂", f"{props['g_prod']:.4f}")
            if np.isfinite(props["tm_spacing"]):
                st.metric(
                    "Transverse mode spacing Δν⊥",
                    f"{props['tm_spacing']/1e6:.3f} MHz",
                )
            else:
                st.metric("Transverse mode spacing Δν⊥", "—")

        st.markdown("---")

        if props["is_stable"]:
            st.success("Cavity is geometrically stable (0 < g₁·g₂ < 1).")
        else:
            st.error("Cavity is not geometrically stable (g₁·g₂ outside (0, 1)).")

        if np.isfinite(props["tm_spacing"]):
            ratio = props["tm_spacing"] / props["linewidth_fwhm"]
            if ratio >= 5.0:
                st.success(
                    f"Transverse mode spacing / linewidth = **{ratio:.2f}** "
                    "(≥ 5, well separated)."
                )
            else:
                st.warning(
                    f"Transverse mode spacing / linewidth = **{ratio:.2f}** "
                    "(< 5, transverse modes not well separated)."
                )
        else:
            st.warning(
                "Transverse mode spacing is not real-valued; skipping HOM-to-linewidth check."
            )

    # ---------- Derivations tab ----------
    with tab_math:
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

    # ---------- Navigation ----------
    st.markdown("---")
    cnav1, cnav2 = st.columns([1, 1])
    with cnav1:
        if st.button("← Edit parameters"):
            go_to("two_mirror_params")
            st.rerun()
    with cnav2:
        if st.button("⌂ Start over"):
            go_to("landing")
            st.rerun()


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
