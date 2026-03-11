import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from tunablefc_design import BW_tunability, T_3_mirror, T_effective_2_mirror, lambda0


# --------------------------------------------------------------------
# Basic configuration
# --------------------------------------------------------------------

st.set_page_config(
    page_title="Thermally Tunable Filter Cavity",
    layout="wide",
)

st.title("Thermally Tunable Fabry-Perot Filter Cavity")
st.markdown(
    """
This webapp models a **plano–concave filter cavity** whose input coupler
is a **thermally tunable etalon** (two partially reflective surfaces separated by a solid substrate of refractive index n)
and whose end mirror is a highly reflective concave mirror. 
The purpose of this model is to demonstrate the effect of modifying the input coupler transmissivity via temperature tuning on the filter cavity response.

- Adjust the **geometry**, **mirror reflectivities**, and **thermal tunability**
  from the sidebar.
- Use the tabs to model **power transmission**, **phase response**, and
  **stability criteria**.
"""
)


# --------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------

st.sidebar.header("Input parameters")

# Lengths (slider + textbox)
st.sidebar.markdown("**Etalon spacing L₁**")
col_L1_slider, col_L1_input = st.sidebar.columns([2, 1])
with col_L1_slider:
    L1_mm_slider = st.slider(
        "L₁ [mm] (slider)",
        min_value=1.0,
        max_value=50.0,
        value=6.0,
        step=0.01,
        label_visibility="collapsed",
    )
with col_L1_input:
    L1_mm = st.number_input(
        "L₁ [mm]",
        min_value=0.1,
        max_value=50.0,
        value=L1_mm_slider,
        step=0.001,
    )

st.sidebar.markdown("**Long cavity length L₂**")
col_L2_slider, col_L2_input = st.sidebar.columns([2, 1])
with col_L2_slider:
    L2_m_slider = st.slider(
        "L₂ [m] (slider)",
        min_value=0.05,
        max_value=3.0,
        value=0.15,
        step=0.01,
        label_visibility="collapsed",
    )
with col_L2_input:
    L2_m = st.number_input(
        "L₂ [m]",
        min_value=0.01,
        max_value=3.0,
        value=L2_m_slider,
        step=0.001,
    )

# Reflectivities (slider + textbox)
st.sidebar.markdown("**Reflectivities**")
R1 = st.sidebar.number_input(
    "R₁ (etalon mirror 1)",
    min_value=0.5,
    max_value=0.999999,
    value=0.90,
    step=1e-4,
    format="%.6f",
)
R2 = st.sidebar.number_input(
    "R₂ (etalon mirror 2)",
    min_value=0.5,
    max_value=0.999999,
    value=0.90,
    step=1e-4,
    format="%.6f",
)
R3 = st.sidebar.number_input(
    "R₃ (end mirror)",
    min_value=0.9,
    max_value=0.9999999,
    value=0.9980,
    step=1e-5,
    format="%.7f",
)

# Thermal tuning
st.sidebar.markdown("**Etalon substrate properties**")

n_etalon = st.sidebar.number_input(
    "Etalon refractive index n",
    min_value=1.0,
    max_value=4.0,
    value=1.45,
    step=0.01
)

alpha = st.sidebar.number_input(
    "dL₁/dT (ppm/K) – effective tunability",
    min_value=0.0,
    max_value=100.0,
    value=0.55,
    step=0.01,
)

dn_dT = st.sidebar.number_input(
    "Thermo-optic coefficient dn/dT (1/K)",
    value=1.0e-5,
    format="%.2e",
    help="Temperature derivative of the refractive index."
)

max_delta_T = st.sidebar.number_input(
    "Maximum temperature excursion ΔTₘₐₓ [°C]",
    min_value=0.0,
    max_value=1000.0,
    value=10.0,
    step=0.5,
)
n_temperatures = st.sidebar.number_input(
    "Number of temperature samples",
    min_value=2,
    max_value=200,
    value=10,
    step=1,
)

# cavity loss parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Cavity loss")
eps_loss = st.sidebar.number_input(
    "Round-trip cavity loss ε",
    min_value=0.0,
    max_value=0.1,
    value=5e-4,
    format="%.2e",
    help="Effective round-trip loss of the cavity (scatter, absorption, etc.)."
)

# Stability / geometry parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Stability model")
R_c = st.sidebar.number_input(
    "End mirror radius of curvature R_c [m]",
    min_value=0.01,
    max_value=1e5,
    value=1.0,
    step=0.01,
    format="%.3f",
    help="Used for simple plano–concave stability estimate.",
)
use_effective_length = st.sidebar.checkbox(
    "Include etalon thickness in effective cavity length",
    value=True,
    help="If checked, the effective cavity length is L₂ + n·L₁.",
)


# --------------------------------------------------------------------
# Helper functions (field-level three-surface model)
# --------------------------------------------------------------------

c = 3e8  # speed of light [m/s]
f0 = c / lambda0


def etalon_coeffs(r1, r2, t1, t2, k_val, n, d):
    """Return complex etalon transmission and reflections for given k."""
    delta = 2 * k_val * n * d
    exp_i_delta = np.exp(1j * delta)
    denom = 1 - r1 * r2 * exp_i_delta

    t_et = t1 * t2 * np.exp(1j * delta / 2) / denom
    r_et_L = r1 + (t1**2 * r2 * exp_i_delta) / denom
    r_et_R = r2 + (t2**2 * r1 * exp_i_delta) / denom

    return t_et, r_et_L, r_et_R


def three_surface_response(
    freqs,
    R1,
    R2,
    R3,
    L1,
    L2,
    n_substrate=n_etalon,
):
    """
    Compute complex reflection and transmission of the three-surface system
    (etalon + long cavity + end mirror) as in CavityPhaseResponse.ipynb.
    """
    r1, r2, r3 = np.sqrt(R1), np.sqrt(R2), np.sqrt(R3)
    t1, t2, t3 = np.sqrt(1 - R1), np.sqrt(1 - R2), np.sqrt(1 - R3)

    k_vals = 2 * np.pi * freqs / c
    theta = 2 * k_vals * L2

    t_et, r_et_L, r_et_R = etalon_coeffs(
        r1=r1,
        r2=r2,
        t1=t1,
        t2=t2,
        k_val=k_vals,
        n=n_substrate,
        d=L1,
    )

    denom = 1 - r_et_R * r3 * np.exp(1j * theta)

    # Total reflection and transmission fields
    r_tot = r_et_L + (t_et**2 * r3 * np.exp(1j * theta)) / denom
    t_tot = (t_et * t3 * np.exp(1j * k_vals * L2)) / denom

    return r_tot, t_tot


def simple_stability(L_eff, R_c):
    """
    Simple plano–concave cavity stability via g-factors:
    g1 = 1 (plane), g2 = 1 - L_eff / R_c, stable if 0 < g1*g2 < 1.
    """
    g1 = 1.0
    g2 = 1.0 - L_eff / R_c
    g_prod = g1 * g2
    stable = 0 < g_prod < 1
    return g1, g2, g_prod, stable


def cavity_waist(lambda0, L2, Rc):
    """
    Waist size and Rayleigh range for a plano–concave cavity.
    Matches the formulas used in pdhlocking.ipynb.
    """
    if L2 <= 0 or Rc <= L2:
        raise ValueError("Cavity must satisfy 0 < L2 < Rc for stability.")

    zR = np.sqrt(L2 * (Rc - L2))
    w0 = np.sqrt((lambda0 / np.pi) * zR)
    return w0, zR

def Teff(phi, R1, R2):
    sqrtR = np.sqrt(R1 * R2)
    num = (1 - R1) * (1 - R2)
    den = 1 - 2 * sqrtR * np.cos(phi) + R1 * R2
    return num / den

def dTeff_dphi(phi, R1, R2):
    sqrtR = np.sqrt(R1 * R2)
    num = (1 - R1) * (1 - R2)
    den = 1 - 2 * sqrtR * np.cos(phi) + R1 * R2
    return -(num * (2 * sqrtR * np.sin(phi))) / (den**2)

def phi_from_L1(L1, n=n_etalon, lambda0=lambda0):
    return 4 * np.pi * n * L1 / lambda0

def pole_gamma(phi, R1, R2, L2, eps_loss=1e-6):
    # gamma in Hz
    return (c / (8 * np.pi * L2)) * (eps_loss + Teff(phi, R1, R2))

def dphi_dT(L1, n=n_etalon, alpha=alpha, dn_dT=dn_dT, lambda0=lambda0):
    # total phase tuning from expansion + thermo-optic
    return (4 * np.pi * L1 / lambda0) * (n * alpha + dn_dT)

def tunability_dgamma_dT(phi, L1, R1, R2, L2, eps_loss=1e-6):
    # dgamma/dT in Hz/K
    return (c / (8 * np.pi * L2)) * dTeff_dphi(phi, R1, R2) * dphi_dT(L1)

# --------------------------------------------------------------------
# Derived quantities shared across tabs
# --------------------------------------------------------------------

L1 = L1_mm * 1e-3  # convert to metres

# Use BW_tunability with explicit tunability value in ppm/K
baseline_bw, bw_expansion, dL1_dT, dgamma_dL1 = BW_tunability(
    L1, L2_m, R1, R2, R3, tunability=alpha, n=n_etalon, dn_dT=dn_dT
)

# Geometric dL1/dT from the thermal expansion coefficient
dL1_dT_geom = alpha * L1

delta_T_values = np.linspace(0.0, max_delta_T, int(n_temperatures))

# Frequency grid: span ~2 FSR around resonance, recomputed for each L2
FSR_L2_Hz = c / (2 * L2_m)
span_Hz = 2.0 * FSR_L2_Hz  # covers ~2 resonances
N_points = 2001
dnu_arr = np.linspace(-span_Hz / 2.0, span_Hz / 2.0, N_points)
freqs = f0 + dnu_arr

L_eff = L2_m + (use_effective_length * 1.0) * n_etalon * L1
g1, g2, g_prod, is_stable = simple_stability(L_eff=L_eff, R_c=R_c)


# --------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------

tab_tr, tab_phase, tab_stab, tab_eq = st.tabs(
    ["Power transmission", "Phase response", "Stability", "Equations / derivations"]
)


# --------------------------------------------------------------------
# Tab 1: Power transmission and bandwidth tunability
# --------------------------------------------------------------------

with tab_tr:
    st.subheader("Power transmission vs frequency detuning")

    # Use Plotly for interactive, zoomable plotting
    fig = go.Figure()

    # Local k for this detuning range
    k_vals = 2 * np.pi * freqs / c
    x_MHz = dnu_arr * 1e-6

    # Temperature-tuned curves (using effective dL1/dT from the bandwidth model)
    for dT in delta_T_values:
        L1_T = L1 + dL1_dT * dT
        T3 = T_3_mirror(k_vals, L1_T, L2_m, R1, R2, R3)
        bw_T = baseline_bw + bw_expansion * dT
        fig.add_trace(
            go.Scatter(
                x=x_MHz,
                y=T3,
                mode="lines",
                name=f"L₁={L1_T*1e3:.6f} mm, γ≈{bw_T/1e3:.1f} kHz",
            )
        )

    # Effective two-mirror comparison
    T2 = T_effective_2_mirror(k_vals, L2_m, R1, R2, R3)
    fig.add_trace(
        go.Scatter(
            x=x_MHz,
            y=T2,
            mode="lines",
            name="Effective two-mirror cavity",
            line=dict(color="black", dash="dash"),
        )
    )

    # FSR and annotation similar to your reference plot
    FSR_L2_MHz = FSR_L2_Hz * 1e-6

    fig.update_layout(
        xaxis_title="Frequency detuning Δν [MHz]",
        yaxis_title="Transmitted Power [arb. units]",
        yaxis_type="log",
        dragmode="zoom",
        yaxis=dict(fixedrange=True),
        title="Power Transmission with Etalon Tuning",
        legend=dict(orientation="v"),
        annotations=[
            dict(
                x=0.02,
                y=0.05,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                text=(
                    f"FSR(L2) ≈ {FSR_L2_MHz:.3f} MHz<br>"
                    f"Pole: {baseline_bw/1e3:.3f} kHz<br>"
                    f"Tunability: {bw_expansion/1e3:.3f} kHz/°C"
                ),
                bordercolor="gray",
                borderwidth=1,
                bgcolor="white",
                opacity=0.8,
                font=dict(size=10),
            )
        ],
    )

    st.plotly_chart(fig, use_container_width=True)
    st.latex(
        rf"R_1={R1:.4f},\; R_2={R2:.4f},\; R_3={R3:.4f},\; L_2={L2_m*100:.1f}\,\text{{ cm}}"
    )

    st.markdown(
        f"""
**Summary**

- Baseline pole (bandwidth): **{baseline_bw/1e3:.2f} kHz**
- Bandwidth tunability: **{bw_expansion/1e3:.2f} kHz / °C**
- dL₁/dT: **{dL1_dT*1e9:.3f} nm / °C**
"""
    )

    c = 299792458

    # Free spectral range
    FSR = c / (2 * L2_m)

    # Finesse
    finesse = FSR/(2*baseline_bw)

    st.markdown("### Additional cavity parameters")

    st.latex(
        r"""
\mathrm{FSR} = \frac{c}{2L_2}
"""
    )

    st.latex(
        r"""
\mathcal{F} = \frac{\mathrm{FSR}}{\mathrm{FWHM}} = \frac{\mathrm{FSR}}{2\gamma}
"""
    )

    st.markdown(
        f"""
- Free spectral range (FSR): **{FSR/1e6:.2f} MHz**  
- Cavity finesse: **{finesse:.1f}**
"""
    )


    st.markdown("### Power transmission model")
    st.markdown(
        "The three-mirror transmission plotted above is modeled as an effective "
        "Fabry–Perot cavity whose input coupler is the etalon. The effective "
        "power transmissivity of the compound input coupler is"
    )

    st.latex(
        r"""
    T_{\mathrm{eff}}(\phi)
    =
    \frac{(1-R_1)(1-R_2)}
    {1-2\sqrt{R_1R_2}\cos\phi+R_1R_2}
    """
    )

    st.markdown("and the resulting filter-cavity pole is")

    st.latex(
        r"""
    \gamma(\phi)
    =
    \frac{c}{8\pi L_2}
    \left[
    T_{\mathrm{eff}}(\phi) + T_3 + \epsilon
    \right]
    """
    )

    st.markdown(
        "Here $T_{\\mathrm{eff}}$ is the effective transmissivity of the etalon "
        "input coupler, $T_3$ is the end mirror fixed transmissivity, $\\epsilon$ represents the round-trip loss, and "
        "$\\gamma$ is the cavity pole or half-width at half-maximum. These "
        "quantities set the Lorentzian resonance width of the cavity."
    )

    st.markdown("### Thermal tunability of the cavity pole")

    st.markdown(
        "Thermal tuning of the cavity bandwidth arises because the etalon phase "
        "depends on temperature through both thermal expansion and the "
        "thermo-optic effect. The etalon round-trip phase is"
    )

    st.latex(
        r"""
    \phi
    =
    \frac{4\pi n L_1}{\lambda_0}
    """
    )

    st.markdown(
        "Taking the temperature derivative gives"
    )

    st.latex(
        r"""
    \frac{d\phi}{dT}
    =
    \frac{4\pi L_1}{\lambda_0}
    \left(
    n\alpha + \frac{dn}{dT}
    \right)
    """
    )

    st.markdown(
        "Because the cavity pole depends on the etalon phase through "
        "$T_{\\mathrm{eff}}(\\phi)$, the thermal tunability is"
    )

    st.latex(
        r"""
    \frac{d\gamma}{dT}
    =
    \frac{c}{8\pi L_2}
    \frac{dT_{\mathrm{eff}}}{d\phi}
    \frac{d\phi}{dT}
    """
    )

    st.markdown(
        "**Note**: See https://lightmachinery.com/optical-design-center/etalon-temperature-tuning/ for thermo-optic properties of different substrates."
    )

    st.markdown(
        "The phase derivative of the effective transmissivity is"
    )

    st.latex(
        r"""
    \frac{dT_{\mathrm{eff}}}{d\phi}
    =
    -\frac{(1-R_1)(1-R_2)\,2\sqrt{R_1R_2}\sin\phi}
    {\left(1-2\sqrt{R_1R_2}\cos\phi+R_1R_2\right)^2}
    """
    )

    st.markdown(
        "Combining these expressions gives the thermal tuning of the cavity pole:"
    )

    st.latex(
        r"""
    \frac{d\gamma}{dT}
    =
    \frac{c}{8\pi L_2}
    \left[
    -\frac{(1-R_1)(1-R_2)\,2\sqrt{R_1R_2}\sin\phi}
    {\left(1-2\sqrt{R_1R_2}\cos\phi+R_1R_2\right)^2}
    \right]
    \left[
    \frac{4\pi L_1}{\lambda_0}
    \left(
    n\alpha + \frac{dn}{dT}
    \right)
    \right]
    """
    )

    st.markdown(
    "In practice this mechanism allows the cavity bandwidth to be tuned "
    "thermally by modifying the internal etalon phase. For typical tabletop parameters "
    "we want to enable bandwidth shifts of ~10\% of the baseline pole over temperature changes "
    "of order 10 °C."
    )
    # ---------- Plot versus phase around quadrature ----------
    phi0 = np.linspace(-np.pi, np.pi, 4000) + (np.pi / 2)  # centered near quadrature

    gamma_vals = pole_gamma(phi0, R1, R2, L2_m, eps_loss=eps_loss)
    tune_vals = tunability_dgamma_dT(phi0, L1, R1, R2, L2_m, eps_loss=eps_loss)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4), sharex=False)

    # ---------------- Panel 1: full phase scan ----------------
    axes[0].plot(phi0, gamma_vals/1e6, label="Pole frequency $\\gamma$ (MHz)")
    axes[0].plot(phi0, tune_vals/1e6, label="Tunability $d\\gamma/dT$ (MHz/K)")
    axes[0].axvline(np.pi/2, linestyle="--", linewidth=1, label="Quadrature $\\phi=\\pi/2$")
    axes[0].set_xlabel("Etalon phase $\\phi$")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Filter cavity pole and tunability vs etalon phase")
    axes[0].grid(True)
    axes[0].legend()

    # ---------------- Panel 2: zoom near quadrature ----------------
    phi_q = np.pi / 2
    zoom_width = 1.0

    mask = (phi0 > phi_q - zoom_width) & (phi0 < phi_q + zoom_width)

    axes[1].plot(phi0[mask], gamma_vals[mask]/1e6, label="$\\gamma$ (MHz)")
    axes[1].plot(phi0[mask], tune_vals[mask]/1e6, label="$d\\gamma/dT$ (MHz/K)")
    axes[1].axvline(phi_q, linestyle="--", linewidth=1, label="Quadrature")
    axes[1].set_xlabel("Etalon phase $\\phi$")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("Zoom near quadrature")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

# --------------------------------------------------------------------
# Tab 2: Phase response and group delay
# --------------------------------------------------------------------

with tab_phase:

    st.subheader("Reflection phase and group delay with temperature tuning")

    from plotly.subplots import make_subplots

    x_MHz = dnu_arr * 1e-6

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Reflection phase vs frequency detuning",
            "Group delay vs frequency detuning",
        ),
        horizontal_spacing=0.12,
    )

    for dT in delta_T_values:
        # Temperature-dependent etalon parameters
        n_T = n_etalon + dn_dT * dT
        L1_T = L1 + dL1_dT * dT

        # Round-trip etalon phase
        phi_T = 4 * np.pi * n_T * L1_T / lambda0
        phi_T_wrapped = np.mod(phi_T, 2 * np.pi)

        # Full three-surface response with temperature-dependent substrate index
        r_tot, t_tot = three_surface_response(
            freqs=freqs,
            R1=R1,
            R2=R2,
            R3=R3,
            L1=L1_T,
            L2=L2_m - dL1_dT * dT,
            n_substrate=n_etalon,
        )

        phase = np.unwrap(np.angle(r_tot))
        omega = 2 * np.pi * freqs

        # Group delay: tau_g = -d(arg r)/d omega
        tau_g = -np.gradient(phase, omega)

        fig.add_trace(
            go.Scatter(
                x=x_MHz,
                y=phase,
                mode="lines",
                name=f"ΔT = {dT:.2f} °C, φ mod 2π = {phi_T_wrapped:.2f}",
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_MHz,
                y=tau_g,
                mode="lines",
                showlegend=False,
                line=dict(width=2),
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Frequency detuning Δν [MHz]", row=1, col=1)
    fig.update_xaxes(title_text="Frequency detuning Δν [MHz]", row=1, col=2)

    fig.update_yaxes(title_text="Reflection phase [rad]", row=1, col=1, fixedrange=True)
    fig.update_yaxes(title_text="Group delay [s]", row=1, col=2, fixedrange=True)

    fig.update_layout(
        dragmode="zoom",
        height=500,
        legend_title_text="Temperature tuning",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "The curves above show how thermal tuning of the etalon modifies the "
        "complex reflection phase and associated group delay of the three-surface cavity. "
        "Here the etalon phase is computed from the temperature-dependent optical path length, "
        "including both thermal expansion and the thermo-optic contribution."
    )

    st.latex(r"\phi(T) = \frac{4\pi\,n(T)\,L_1(T)}{\lambda_0}")
    st.latex(r"\tau_g = -\frac{d}{d\omega}\arg\!\left(r_{\mathrm{tot}}\right)")

    st.markdown(
        "This model describes the complex field reflection and transmission profile of the planar etalon-concave cavity. All "
        "quantities are treated at the **field level**, with frequency dependence "
        "retained explicitly."
    )

    st.markdown("#### Definitions")
    st.latex(r"k = \frac{2\pi f}{c}")
    st.latex(r"\delta = 2 k n d")
    st.latex(r"\theta = 2 k L")
    st.markdown("All rᵢ and tᵢ are **amplitude** reflectivities and transmissivities.")

    st.image("tunablefc_drawing.png")

    st.markdown("#### Etalon transmission")
    st.latex(
        r"""
t_{\mathrm{et}}(f)
=
\frac{
t_1 t_2 e^{i\delta/2}
}{
1 - r_1 r_2 e^{i\delta}
}
"""
    )
    st.markdown(
        "This corresponds to the coherent sum of all multiple internal "
        "reflections inside the solid spacer."
    )

    st.markdown("#### Etalon reflection from the left")
    st.latex(
        r"""
r_{\mathrm{et,L}}(f)
=
r_1
+
\frac{
t_1^2 r_2 e^{i\delta}
}{
1 - r_1 r_2 e^{i\delta}
}
"""
    )
    st.markdown(
        "This includes the prompt reflection from surface 1 and all internal "
        "etalon contributions."
    )

    st.markdown("#### Etalon reflection from the right")
    st.latex(
        r"""
r_{\mathrm{et,R}}(f)
=
r_2
+
\frac{
t_2^2 r_1 e^{i\delta}
}{
1 - r_1 r_2 e^{i\delta}
}
"""
    )
    st.markdown(
        "This quantity acts as the **input coupler** for the long cavity."
    )

    st.markdown("#### Total reflection and transmission of three‑surface system")
    st.latex(
        r"""
r_{\mathrm{tot}}(f)
=
r_{\mathrm{et,L}}
+
\frac{
t_{\mathrm{et}}^2 r_3 e^{i\theta}
}{
1 - r_{\mathrm{et,R}} r_3 e^{i\theta}
}
"""
    )
    st.markdown(
        "This expression describes an effective Fabry–Perot cavity whose input "
        "coupler is the etalon itself. The reflection phase plotted above is:"
    )
    st.latex(r"\phi(f) = \arg\big(r_{\mathrm{tot}}(f)\big)")
    st.latex(
        r"""
t_{\mathrm{tot}}(f)
=
\frac{
t_{\mathrm{et}}\, t_3\, e^{i k L}
}{
1 - r_{\mathrm{et,R}} r_3 e^{i\theta}
}
"""
    )
    st.latex(
        r"""
T_{\mathrm{tot}}(f)
=
\left| t_{\mathrm{tot}}(f) \right|^2
"""
    )


# --------------------------------------------------------------------
# Tab 3: Stability criteria
# --------------------------------------------------------------------

with tab_stab:
    st.image("planoconcavecavity.png")
    st.subheader("Geometric stability (plano–concave model)")

    st.markdown(
        "For a simple plano–concave cavity with effective length L2 and end mirror radius of curvature R_c, "
        "the standard **g‑factors** are"
    )
    st.latex(
        r"""
g_1 = 1, \qquad g_2 = 1 - \frac{L_{\mathrm{2}}}{R_c},
"""
    )
    st.markdown("and the cavity is **geometrically stable** if")
    st.latex(
        r"""
0 < g_1 g_2 < 1.
"""
    )

    st.latex(
        r"""
L_{\mathrm{eff}} = L_2 + n L_1, \qquad g_1 = 1, \qquad g_2 = 1 - \frac{L_{\mathrm{eff}}}{R_c}
"""
    )

    if is_stable:
        st.success("The cavity is **geometrically stable** (0 < g₁g₂ < 1).")
    else:
        st.error("The cavity is **not** geometrically stable (g₁g₂ outside (0, 1)).")

    st.caption(
        "This is a first-pass stability estimate; more detailed models can "
        "include the etalon as an effective mirror with complex phase and loss."
    )

    st.markdown("### Cavity mode parameters (waist and Rayleigh range)")

    try:
        w0, zR = cavity_waist(lambda0, L2_m, R_c)
        w2 = w0 * np.sqrt(1 + (L2_m / zR)**2)
        st.success(f"""
        Waist at flat mirror: **w₀ = {w0*1e6:.2f} µm**  
        Rayleigh range: **z_R = {zR:.4f} m**  
        Spot size at end mirror: **w₂ = {w2*1e6:.2f} µm**
        """)

        st.latex(
            r"""
z_R = \sqrt{L_2(R_c - L_2)}, \qquad
w_0 = \sqrt{\frac{\lambda_0}{\pi} z_R}
"""
        )
    except ValueError as e:
        st.warning(
            "Mode parameters are only defined for a stable cavity with "
            "0 < L₂ < R_c. Adjust L₂ or R_c to satisfy the stability condition."
        )

    st.markdown("### Stability from ABCD matrix formalism")
    st.markdown(
        "The same condition arises from the paraxial **ABCD matrix** for one "
        "round trip of the cavity. For propagation over distance \(L\) and "
        "reflection from a spherical mirror of radius \(R\),"
    )
    st.latex(
        r"""
M_{\text{prop}} =
\begin{pmatrix}
1 & L \\
0 & 1
\end{pmatrix},
\qquad
M_{\text{mirror}} =
\begin{pmatrix}
1 & 0 \\
-2/R & 1
\end{pmatrix}.
"""
    )
    st.markdown(
        "For a plano–concave cavity the round‑trip matrix can be written in "
        "terms of the g‑factors, and the stability condition is"
    )
    st.latex(
        r"""
\left|\frac{A + D}{2}\right| < 1
\quad\Longleftrightarrow\quad
0 < g_1 g_2 < 1,
"""
    )
    st.markdown(
        "which is the criterion implemented above to decide whether the chosen "
        "geometry is stable."
    )

# --------------------------------------------------------------------
# Tab 4: Equations / derivations
# --------------------------------------------------------------------

with tab_eq:
    st.subheader("Cavity response derivations")

    st.markdown("### Effective transmissivity of compound mirror")
    st.latex(
        r"""
T_{\mathrm{eff}}(\phi)
=
\frac{(1 - R_1)(1 - R_2)}
{1 - 2\sqrt{R_1 R_2}\cos\phi + R_1 R_2}
"""
    )

    st.latex(
        r"""
\gamma
=
\frac{c}{8\pi L_2}
\left(T_{\mathrm{eff}} + (1 - R_3)\right)
"""
    )

    st.markdown("### Phase tuning with temperature")
    st.latex(
        r"""
\phi = \frac{4\pi n L_1}{\lambda_0}
"""
    )
    st.latex(
        r"""
\frac{d\phi}{dT}
=
\frac{4\pi L_1}{\lambda_0}
\left(
n\alpha + \frac{dn}{dT}
\right)
"""
    )
    st.latex(
        r"""
\frac{d\gamma}{dT}
=
\frac{c}{8\pi L_2}
\frac{dT_{\mathrm{eff}}}{d\phi}
\frac{d\phi}{dT}
"""
    )
    st.latex(
        r"""
\frac{dT_{\mathrm{eff}}}{d\phi}
=
-\frac{(1-R_1)(1-R_2)\,2\sqrt{R_1R_2}\sin\phi}
{\left(1 - 2\sqrt{R_1R_2}\cos\phi + R_1R_2\right)^2}
"""
    )

    st.markdown("Maximum linear tunability occurs near quadrature:")
    st.latex(r"\phi \approx \pi/2, \qquad \sin\phi \ \text{maximized}")
    st.markdown("In this regime the cavity pole varies approximately linearly with temperature.")

    st.markdown("### Three-surface field model")
    st.latex(
        r"""
\delta = 2kn d, \qquad \theta = 2kL
"""
    )
    st.latex(
        r"""
t_{\mathrm{et}} =
\frac{t_1 t_2 e^{i\delta/2}}{1 - r_1 r_2 e^{i\delta}}
"""
    )
    st.latex(
        r"""
r_{\mathrm{et,L}} =
r_1 +
\frac{t_1^2 r_2 e^{i\delta}}{1 - r_1 r_2 e^{i\delta}},
\qquad
r_{\mathrm{et,R}} =
r_2 +
\frac{t_2^2 r_1 e^{i\delta}}{1 - r_1 r_2 e^{i\delta}}
"""
    )
    st.latex(
        r"""
r_{\mathrm{tot}}(f)
=
r_{\mathrm{et,L}}
+
\frac{t_{\mathrm{et}}^2 r_3 e^{i\theta}}
{1 - r_{\mathrm{et,R}} r_3 e^{i\theta}}
"""
    )
    st.latex(
        r"""
t_{\mathrm{tot}}(f)
=
\frac{t_{\mathrm{et}} t_3 e^{ikL}}
{1 - r_{\mathrm{et,R}} r_3 e^{i\theta}}
"""
    )

    st.markdown(
        "These expressions are the basis for the **power transmission**, "
        "**phase response**, and **group delay** plotted in the other tabs."
    )

