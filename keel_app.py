import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import trimesh
import plotly.graph_objects as go

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False



LEAD_DENSITY_DEFAULT = 11340.0  # kg/m3
SEAWATER_DENSITY = 1025.0      # kg/m3
SEAWATER_NU = 1.05e-6          # m2/s approx

# Common stations used in both 3D and 2D views so the two visualisations align.
DRAWING_STATIONS = (0.00, 0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 1.00)


def naca_00xx_thickness(x, t, le_radius_factor=1.0):
    """Closed NACA 00xx-style half-thickness distribution.

    The standard leading-edge radius is approximately:
        r/c = 1.1019 * t^2
    where t is the full thickness/chord ratio.

    le_radius_factor = 1.0 gives the standard NACA nose. Values above or
    below 1.0 make the nose fuller or finer. The distribution is normalised
    so the selected maximum thickness is retained.
    """
    x = np.asarray(x, dtype=float)
    xclip = np.clip(x, 0.0, 1.0)

    # LE radius is proportional to the square of the sqrt(x) coefficient.
    a0 = 0.2969 * math.sqrt(max(le_radius_factor, 1e-6))
    shape = (
        a0*np.sqrt(xclip)
        - 0.1260*xclip
        - 0.3516*xclip**2
        + 0.2843*xclip**3
        - 0.1036*xclip**4
    )

    # Keep the requested maximum thickness unchanged when LE radius is varied.
    grid = np.linspace(0.0, 1.0, 4001)
    gshape = (
        a0*np.sqrt(grid)
        - 0.1260*grid
        - 0.3516*grid**2
        + 0.2843*grid**3
        - 0.1036*grid**4
    )
    gmax = max(float(np.max(gshape)), 1e-12)
    return (t / 2.0) * shape / gmax


def standard_le_radius_ratio(t):
    """Standard NACA leading-edge radius as a fraction of chord."""
    return 1.1019 * t * t


def _hermite_segment(x, x0, x1, y0, y1, m0, m1):
    """Cubic Hermite segment used for a smooth, monotonic chord remap."""
    h = x1 - x0
    u = (x - x0) / h
    h00 = 2*u**3 - 3*u**2 + 1
    h10 = u**3 - 2*u**2 + u
    h01 = -2*u**3 + 3*u**2
    h11 = u**3 - u**2
    return h00*y0 + h10*h*m0 + h01*y1 + h11*h*m1


def _series_chord_map(x, target):
    """Move maximum thickness aft without changing the leading-edge radius.

    The first 10% of chord is left completely unchanged (x maps to x), so the
    NACA sqrt(x) nose and its calculated LE radius remain rounded for every
    selected family. Farther aft the chord is smoothly remapped so the base
    profile's ~30% maximum thickness occurs at the requested target station.
    """
    x = np.asarray(x, dtype=float)
    xc = np.clip(x, 0.0, 1.0)
    if abs(target - 0.30) < 1e-12:
        return xc

    x0 = 0.10
    y0 = 0.10
    yt = 0.30

    # Positive matching slopes keep the mapping smooth and monotonic.
    s_left = (yt - y0) / (target - x0)
    s_right = (1.0 - yt) / (1.0 - target)
    mt = 2.0 * s_left * s_right / (s_left + s_right)

    out = np.empty_like(xc)
    a = xc <= x0
    b = (xc > x0) & (xc <= target)
    c = xc > target
    out[a] = xc[a]  # preserve the exact nose and LE radius
    out[b] = _hermite_segment(xc[b], x0, target, y0, yt, 1.0, mt)
    out[c] = _hermite_segment(xc[c], target, 1.0, yt, 1.0, mt, 1.0)
    return out


def naca_series_thickness(x, t, series, le_radius_factor=1.0):
    """Symmetrical longitudinal thickness envelope.

    00 and 63 currently use the base rounded profile. 64 and 65 move the
    maximum-thickness station aft while preserving the same calculated NACA
    leading-edge radius. Exact NACA 6-series ordinates can replace these
    prototype families later without changing the rest of the bulb model.
    """
    x = np.asarray(x, dtype=float)
    target = {'00': 0.30, '63': 0.30, '64': 0.40, '65': 0.50}[series]
    xr = _series_chord_map(x, target)
    return naca_00xx_thickness(xr, t, le_radius_factor)





SECTION_SERIES = {
    "00": {"max_pos": 0.30},
    "63": {"max_pos": 0.35},
    "64": {"max_pos": 0.40},
    "65": {"max_pos": 0.40},
    "66": {"max_pos": 0.45},
}

# Published reference basic-thickness ordinates for the NACA 65-series.
# The curve is normalized; the app's thickness inputs still set final thickness.
_NACA6_REFERENCE = {
    "63": (
        np.array([0, 0.5, 0.75, 1.25, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], dtype=float)/100.0,
        np.array([0, 0.9838, 1.1938, 1.5181, 2.0979, 2.9256, 3.5432, 4.0386, 4.8003, 5.3431, 5.7115, 5.9304, 6.0005, 5.922, 5.7051, 5.3698, 4.9354, 4.4206, 3.8398, 3.2112, 2.5567, 1.9002, 1.2724, 0.7076, 0.2521, 0], dtype=float)/100.0
    ),
    "64": (
        np.array([0, 0.5, 0.75, 1.25, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], dtype=float)/100.0,
        np.array([0, 0.9755, 1.1802, 1.4904, 2.0331, 2.8075, 3.3914, 3.8668, 4.6159, 5.1704, 5.5721, 5.8381, 5.9789, 5.9752, 5.7972, 5.4788, 5.0545, 4.5462, 3.9721, 3.3488, 2.6938, 2.0289, 1.3815, 0.7862, 0.2883, 0], dtype=float)/100.0
    ),
    "65": (
        np.array([0, 0.5, 0.75, 1.25, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], dtype=float)/100.0,
        np.array([0, 0.9146, 1.1031, 1.3824, 1.8792, 2.6091, 3.1756, 3.649, 4.4035, 4.9758, 5.407, 5.7172, 5.9135, 5.999, 5.9512, 5.7573, 5.4137, 4.9445, 4.3806, 3.7427, 3.0576, 2.3433, 1.6282, 0.9461, 0.3534, 0], dtype=float)/100.0
    ),
    "66": (
        np.array([0, 0.5, 0.75, 1.25, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], dtype=float)/100.0,
        np.array([0, 0.9039, 1.0858, 1.3525, 1.8067, 2.4936, 3.0378, 3.4933, 4.234, 4.7996, 5.2364, 5.5656, 5.7987, 5.9424, 5.9981, 5.9642, 5.8339, 5.5857, 5.1477, 4.5122, 3.7677, 2.9432, 2.0807, 1.2346, 0.4738, 0], dtype=float)/100.0
    ),
}

def _reference_interp(x, xp, yp):
    """Shape-preserving interpolation through reference ordinate stations."""
    x = np.asarray(x, dtype=float)
    try:
        from scipy.interpolate import PchipInterpolator
        return np.maximum(PchipInterpolator(xp, yp)(x), 0.0)
    except Exception:
        return np.maximum(np.interp(x, xp, yp), 0.0)

def section_series_curve(x, series, le_radius_factor=1.0):
    """Selected base section normalized to unit peak half-thickness."""
    if series in _NACA6_REFERENCE:
        xp, yp = _NACA6_REFERENCE[series]
        y = _reference_interp(np.asarray(x, dtype=float), xp, yp)
        ymax = float(np.max(y))
        return y / ymax if ymax > 1e-12 else y

    # 00 retains the existing smooth NACA-style section.
    return bulb_thickness(x, 1.0, SECTION_SERIES["00"]["max_pos"], le_radius_factor)



def keel_section_xy(chord_mm, thickness_pct, series, n=301):
    x = np.linspace(0.0, 1.0, n)
    unit = section_series_curve(x, series, 1.0)
    unit = unit / max(float(np.max(unit)), 1e-12)
    half_t = chord_mm * (thickness_pct / 100.0) / 2.0
    y = unit * half_t
    return x * chord_mm, y


def keel_planform_figure(top_setout, bot_setout, top_chord, bot_chord, top_depth, span):
    """Keel side elevation using the bow setout line as the fore/aft datum.

    Setout dimensions run from the bow setout line to the FORWARD edge of each
    keel section. The long setout distance is compressed/not to scale so the
    keel remains large on screen. The keel chords and vertical dimensions are
    shown proportionally.
    """
    bot_depth = top_depth + span

    # Local drawing coordinates. Bow/forward is to the RIGHT.
    # Put the common aft edge at x=0. With the default geometry:
    # 4300 + 550 = 4850 and 4370 + 480 = 4850, so the aft edge is vertical.
    top_aft = 0.0
    bot_aft = (top_setout + top_chord) - (bot_setout + bot_chord)
    top_fwd = top_aft + top_chord
    bot_fwd = bot_aft + bot_chord

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[top_aft, top_fwd, bot_fwd, bot_aft, top_aft],
        y=[top_depth, top_depth, bot_depth, bot_depth, top_depth],
        mode='lines', name='Keel', line=dict(width=3)
    ))

    chord_max = max(top_chord, bot_chord)
    keel_forward = max(top_fwd, bot_fwd)
    bow_x = keel_forward + chord_max * 0.85
    x_left = min(top_aft, bot_aft) - chord_max * 0.35
    x_right = bow_x + chord_max * 0.20

    # LWL and bow setout datum.
    fig.add_shape(type='line', x0=x_left, x1=x_right, y0=0, y1=0, line=dict(width=2))
    fig.add_annotation(x=x_left + 0.10*(x_right-x_left), y=0, text='LWL', showarrow=False, yshift=18)
    fig.add_shape(type='line', x0=bow_x, x1=bow_x, y0=0, y1=bot_depth + 140, line=dict(width=2))
    fig.add_annotation(x=bow_x, y=(top_depth+bot_depth)/2, text='Bow setout line',
                       showarrow=False, textangle=-90, xshift=22)

    # Fore/aft setout dimensions: bow line to FORWARD edge. NTS.
    top_dim_y = max(80.0, top_depth - 90.0)
    bot_dim_y = bot_depth + 90.0
    for yd, xfwd, value, label, shift in [
        (top_dim_y, top_fwd, top_setout, 'Top', -14),
        (bot_dim_y, bot_fwd, bot_setout, 'Bottom', 14),
    ]:
        fig.add_shape(type='line', x0=xfwd, x1=bow_x, y0=yd, y1=yd, line=dict(width=1))
        fig.add_shape(type='line', x0=xfwd, x1=xfwd, y0=yd-22, y1=yd+22, line=dict(width=1))
        fig.add_shape(type='line', x0=bow_x, x1=bow_x, y0=yd-22, y1=yd+22, line=dict(width=1))
        fig.add_annotation(x=(xfwd+bow_x)/2, y=yd,
                           text=f'{label} setout {value:.0f} mm (NTS)',
                           showarrow=False, yshift=shift)

    # Chord labels.
    fig.add_annotation(x=(top_aft+top_fwd)/2, y=top_depth,
                       text=f'Top chord {top_chord:.0f} mm', showarrow=False, yshift=20)
    fig.add_annotation(x=(bot_aft+bot_fwd)/2, y=bot_depth,
                       text=f'Bottom chord {bot_chord:.0f} mm', showarrow=False, yshift=-20)

    # Vertical height dimensions on the left of the keel.
    dim_x1 = x_left + chord_max*0.08
    dim_x2 = dim_x1 - chord_max*0.16
    # 280 mm: LWL to top section.
    fig.add_shape(type='line', x0=dim_x1, x1=dim_x1, y0=0, y1=top_depth, line=dict(width=1))
    for yy in (0, top_depth):
        fig.add_shape(type='line', x0=dim_x1-22, x1=dim_x1+22, y0=yy, y1=yy, line=dict(width=1))
    fig.add_annotation(x=dim_x1, y=top_depth/2, text=f'{top_depth:.0f} mm',
                       showarrow=False, textangle=-90, xshift=-14)

    # Span: top to bottom.
    fig.add_shape(type='line', x0=dim_x2, x1=dim_x2, y0=top_depth, y1=bot_depth, line=dict(width=1))
    for yy in (top_depth, bot_depth):
        fig.add_shape(type='line', x0=dim_x2-22, x1=dim_x2+22, y0=yy, y1=yy, line=dict(width=1))
    fig.add_annotation(x=dim_x2, y=(top_depth+bot_depth)/2, text=f'Keel span {span:.0f} mm',
                       showarrow=False, textangle=-90, xshift=-14)

    # Overall depth from LWL to bottom section.
    overall_x = dim_x2 - chord_max*0.17
    fig.add_shape(type='line', x0=overall_x, x1=overall_x, y0=0, y1=bot_depth, line=dict(width=1))
    for yy in (0, bot_depth):
        fig.add_shape(type='line', x0=overall_x-22, x1=overall_x+22, y0=yy, y1=yy, line=dict(width=1))
    fig.add_annotation(x=overall_x, y=bot_depth/2, text=f'Overall depth {bot_depth:.0f} mm',
                       showarrow=False, textangle=-90, xshift=-14)

    fig.update_xaxes(range=[overall_x-chord_max*0.12, x_right], visible=False)
    fig.update_yaxes(range=[bot_depth + 250, -120], visible=False, scaleanchor='x', scaleratio=1)
    fig.update_layout(
        title='Keel side elevation — bow to right (setout distance not to scale)',
        height=720, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=30, r=70, t=70, b=30), showlegend=False
    )
    return fig

def section_figure(top_chord, top_t, top_series, bot_chord, bot_t, bot_series):
    fig=go.Figure()
    for chord,t,series,name,offset in [(top_chord,top_t,top_series,'Top',0),(bot_chord,bot_t,bot_series,'Bottom',-max(top_chord,bot_chord)*0.22)]:
        x,y=keel_section_xy(chord,t,series)
        fig.add_trace(go.Scatter(x=x-chord/2,y=y+offset,mode='lines',name=f'{name} upper'))
        fig.add_trace(go.Scatter(x=x-chord/2,y=-y+offset,mode='lines',name=f'{name} lower',showlegend=False))
    fig.update_yaxes(scaleanchor='x',scaleratio=1,title='Thickness (mm)')
    fig.update_xaxes(title='Chord from mid-point (mm)', autorange='reversed')
    fig.update_layout(title='Top and bottom keel sections',height=500,margin=dict(l=30,r=30,t=60,b=30))
    return fig



def keel_3d_figure(top_setout, bot_setout, top_chord, bot_chord, top_depth, span,
                   top_t, bot_t, top_series, bot_series, n_span=31, n_chord=101):
    """3D linear keel surface generated from the selected top and bottom foil sections."""
    eta = np.linspace(0.0, 1.0, n_span)
    xi = np.linspace(0.0, 1.0, n_chord)  # 0 = leading edge, 1 = trailing edge

    # Use the top aft position as the local longitudinal datum. Positive X is forward,
    # so the bow is to the right when the model is viewed from the side.
    aft_ref = top_setout + top_chord
    X = np.zeros((n_span, n_chord))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)

    for i, e in enumerate(eta):
        chord = top_chord + e * (bot_chord - top_chord)
        setout = top_setout + e * (bot_setout - top_setout)
        thick_pct = top_t + e * (bot_t - top_t)

        # Blend the actual selected top and bottom thickness envelopes linearly.
        ut = section_series_curve(xi, top_series, 1.0)
        ub = section_series_curve(xi, bot_series, 1.0)
        ut = ut / max(float(np.max(ut)), 1e-12)
        ub = ub / max(float(np.max(ub)), 1e-12)
        unit = (1.0 - e) * ut + e * ub
        unit = unit / max(float(np.max(unit)), 1e-12)
        half_thickness = chord * (thick_pct / 100.0) * 0.5 * unit

        forward_local = aft_ref - setout
        X[i, :] = forward_local - xi * chord
        Y[i, :] = half_thickness
        Z[i, :] = -(top_depth + e * span)

    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.96, name='Starboard surface'))
    fig.add_trace(go.Surface(x=X, y=-Y, z=Z, showscale=False, opacity=0.96, name='Port surface'))

    # Intermediate section rings make the linear transition easy to inspect.
    for e in np.linspace(0.0, 1.0, 6):
        i = int(round(e * (n_span - 1)))
        xr = np.concatenate([X[i], X[i, ::-1]])
        yr = np.concatenate([Y[i], -Y[i, ::-1]])
        zr = np.full_like(xr, Z[i, 0])
        fig.add_trace(go.Scatter3d(x=xr, y=yr, z=zr, mode='lines',
                                   line=dict(width=3), showlegend=False, hoverinfo='skip'))

    fig.update_layout(
        title='3D keel — linear transition between top and bottom sections',
        height=720, margin=dict(l=0, r=0, t=55, b=0),
        scene=dict(
            xaxis_title='Fore / aft (mm) — forward →',
            yaxis_title='Thickness (mm)',
            zaxis_title='Below LWL (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.55, y=1.25, z=0.75))
        )
    )
    return fig

def section_perimeter_ratio(thickness_pct, series, n=1001):
    """Closed foil perimeter divided by chord for the selected symmetrical section."""
    x = np.linspace(0.0, 1.0, n)
    unit = section_series_curve(x, series, 1.0)
    unit = unit / max(float(np.max(unit)), 1e-12)
    y = unit * (thickness_pct / 100.0) / 2.0
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return 2.0 * float(np.sum(ds))


def hydro_results(top_chord, bot_chord, top_t, bot_t, top_series, bot_series,
                  span, speed_kn, alpha_deg, rho=SEAWATER_DENSITY, nu=SEAWATER_NU,
                  efficiency=0.90):
    """Basic finite-wing keel lift and drag estimate.

    Intended for comparative design work, not as a replacement for CFD or tank testing.
    """
    ct = top_chord / 1000.0
    cb = bot_chord / 1000.0
    b = span / 1000.0
    sref = 0.5 * (ct + cb) * b
    cmean = 0.5 * (ct + cb)
    ar = b*b / max(sref, 1e-12)
    v = speed_kn * 0.514444
    q = 0.5 * rho * v*v

    # Wetted area: integrate local chord * local closed-section perimeter/chord.
    eta = np.linspace(0.0, 1.0, 101)
    chords = ct + (cb-ct)*eta
    pr_top = section_perimeter_ratio(top_t, top_series)
    pr_bot = section_perimeter_ratio(bot_t, bot_series)
    pr = pr_top + (pr_bot-pr_top)*eta
    swet = float(np.trapz(chords*pr, eta) * b)

    # Finite-wing lift slope, incompressible flow.
    a0 = 2.0*math.pi
    a = a0 / (1.0 + a0/(math.pi*efficiency*max(ar, 1e-9)))
    cl = a * math.radians(alpha_deg)
    lift_n = q*sref*cl

    # ITTC-1957 friction line plus a simple foil form factor.
    re = v*cmean/max(nu, 1e-12)
    if re > 1e4:
        cf = 0.075/(math.log10(re)-2.0)**2
    else:
        cf = 0.0
    tmean = 0.5*(top_t+bot_t)/100.0
    form_factor = 1.0 + 2.0*tmean + 60.0*tmean**4
    cd_profile = cf*form_factor*swet/max(sref, 1e-12)
    cd_induced = cl*cl/(math.pi*efficiency*max(ar, 1e-12))
    profile_drag_n = q*sref*cd_profile
    induced_drag_n = q*sref*cd_induced
    total_drag_n = profile_drag_n + induced_drag_n

    return {
        'planform_area_m2': sref, 'wetted_area_m2': swet, 'mean_chord_m': cmean,
        'aspect_ratio': ar, 'speed_ms': v, 'reynolds': re, 'cf': cf,
        'form_factor': form_factor, 'cl': cl, 'cd_profile': cd_profile,
        'cd_induced': cd_induced, 'lift_n': lift_n,
        'profile_drag_n': profile_drag_n, 'induced_drag_n': induced_drag_n,
        'total_drag_n': total_drag_n,
        'lift_kgf': lift_n/9.80665, 'profile_drag_kgf': profile_drag_n/9.80665,
        'induced_drag_kgf': induced_drag_n/9.80665,
        'total_drag_kgf': total_drag_n/9.80665,
        'ld_ratio': abs(lift_n)/total_drag_n if total_drag_n > 1e-12 else 0.0,
        'efficiency': efficiency,
    }


def make_pdf_report(top_setout, bot_setout, top_chord, bot_chord, top_depth, span,
                    top_t, bot_t, top_series, bot_series, speed_kn, alpha_deg, h):
    """Create a simple two-page A4 PDF report as bytes."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1 - geometry and side elevation.
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.10, 0.35, 0.82, 0.52])
        bot_depth = top_depth + span
        top_aft = 0.0
        bot_aft = (top_setout + top_chord) - (bot_setout + bot_chord)
        top_fwd = top_aft + top_chord
        bot_fwd = bot_aft + bot_chord
        bow_x = max(top_fwd, bot_fwd) + max(top_chord, bot_chord)*0.85
        ax.plot([top_aft, top_fwd, bot_fwd, bot_aft, top_aft],
                [top_depth, top_depth, bot_depth, bot_depth, top_depth], linewidth=2.2)
        ax.axhline(0, linewidth=1.2)
        ax.axvline(bow_x, linewidth=1.2)
        ax.text(bow_x, (top_depth+bot_depth)/2, '  Bow setout line', rotation=90, va='center')
        ax.text(top_fwd, top_depth-35, f'Top: {top_setout:.0f} + {top_chord:.0f} mm', ha='right')
        ax.text(bot_fwd, bot_depth+65, f'Bottom: {bot_setout:.0f} + {bot_chord:.0f} mm', ha='right')
        ax.text(top_aft-70, top_depth/2, f'{top_depth:.0f} mm', rotation=90, va='center')
        ax.text(top_aft-150, (top_depth+bot_depth)/2, f'Span {span:.0f} mm', rotation=90, va='center')
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis(); ax.axis('off')
        fig.suptitle('Tboats Keel Designer - Design Report', fontsize=18, fontweight='bold', y=0.96)
        fig.text(0.10, 0.91, 'Keel side elevation - bow to right; bow setout distance shown not to scale', fontsize=10)
        geom = (
            f'Top section: NACA {top_series}, chord {top_chord:.0f} mm, thickness {top_t:.1f}% ({top_chord*top_t/100:.1f} mm)\n'
            f'Bottom section: NACA {bot_series}, chord {bot_chord:.0f} mm, thickness {bot_t:.1f}% ({bot_chord*bot_t/100:.1f} mm)\n'
            f'Top below LWL: {top_depth:.0f} mm     Keel span: {span:.0f} mm     Bottom below LWL: {bot_depth:.0f} mm\n'
            f'Planform area: {h["planform_area_m2"]:.3f} m2     Wetted area: {h["wetted_area_m2"]:.3f} m2     Aspect ratio: {h["aspect_ratio"]:.2f}'
        )
        fig.text(0.10, 0.24, geom, fontsize=10, linespacing=1.7)
        fig.text(0.10, 0.08, 'Prototype design calculation - dimensions to be independently checked before manufacture.', fontsize=8)
        pdf.savefig(fig); plt.close(fig)

        # Page 2 - sections and hydrodynamic results.
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle('Keel Sections and Hydrodynamic Results', fontsize=17, fontweight='bold', y=0.96)
        ax = fig.add_axes([0.10, 0.58, 0.82, 0.27])
        for chord,t,series,name,offset in [
            (top_chord,top_t,top_series,'Top',55), (bot_chord,bot_t,bot_series,'Bottom',-55)]:
            x,y=keel_section_xy(chord,t,series)
            ax.plot(x-chord/2, y+offset, linewidth=1.5)
            ax.plot(x-chord/2, -y+offset, linewidth=1.5)
            ax.text(-chord/2, offset+35, f'{name}: NACA {series}, {t:.1f}%', fontsize=9)
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(True, linewidth=0.3)
        ax.invert_xaxis()  # Forward / leading edge to RHS
        ax.set_xlabel('Chord from mid-point (mm) — forward to RHS'); ax.set_ylabel('Thickness / offset (mm)')
        results = (
            f'Boat speed: {speed_kn:.1f} kn     Angle of attack / leeway: {alpha_deg:.1f} deg\n\n'
            f'Lift coefficient CL: {h["cl"]:.3f}\n'
            f'Lift: {h["lift_n"]:.0f} N  ({h["lift_kgf"]:.1f} kgf)\n\n'
            f'Profile drag: {h["profile_drag_n"]:.1f} N  ({h["profile_drag_kgf"]:.2f} kgf)\n'
            f'Induced drag: {h["induced_drag_n"]:.1f} N  ({h["induced_drag_kgf"]:.2f} kgf)\n'
            f'Total drag: {h["total_drag_n"]:.1f} N  ({h["total_drag_kgf"]:.2f} kgf)\n'
            f'Lift / Drag: {h["ld_ratio"]:.1f}\n\n'
            f'Reynolds number: {h["reynolds"]:,.0f}     Skin-friction Cf: {h["cf"]:.5f}\n'
            f'Planform area: {h["planform_area_m2"]:.3f} m2     Wetted area: {h["wetted_area_m2"]:.3f} m2'
        )
        fig.text(0.10, 0.47, results, fontsize=10.5, linespacing=1.5, va='top')
        assumptions = (
            'Calculation assumptions\n'
            '- Seawater density 1025 kg/m3; kinematic viscosity 1.05e-6 m2/s.\n'
            f'- Finite-wing lift slope with span efficiency e = {h["efficiency"]:.2f}.\n'
            '- Profile drag uses the ITTC-1957 friction line with a simple foil form-factor correction.\n'
            '- Induced drag uses CDi = CL^2 / (pi e AR).\n'
            '- Keel-only calculation: hull and bulb end-plate effects are not included; these will be applied in combined keel/bulb analysis.\n'
            '- Hull/keel junction, free-surface, heel and ventilation effects are not included.\n'
            '- Intended for comparative preliminary design. CFD and/or validated VPP methods are appropriate for final performance predictions.'
        )
        fig.text(0.10, 0.22, assumptions, fontsize=8.5, linespacing=1.5, va='top')
        pdf.savefig(fig); plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

st.set_page_config(page_title='Tboats Keel Designer', layout='wide')
st.title('Tboats Keel Designer — Prototype')
st.caption('Linear keel geometry between the top and bottom sections. Top and bottom chord lines are parallel to the LWL.')

c1,c2=st.columns(2)
with c1:
    st.subheader('Top section')
    top_series=st.selectbox('Top NACA section series',['00','63','64','65','66'],index=1)
    top_chord=st.number_input('Top chord (mm)',100.0,3000.0,550.0,10.0)
    top_t=st.number_input('Top thickness (% chord)',5.0,30.0,13.0,0.5)
    top_setout=st.number_input('Top forward edge from bow setout (mm)',0.0,20000.0,4300.0,10.0)
    top_depth=st.number_input('Top section below LWL (mm)',0.0,5000.0,280.0,10.0)
with c2:
    st.subheader('Bottom section')
    bot_series=st.selectbox('Bottom NACA section series',['00','63','64','65','66'],index=1)
    bot_chord=st.number_input('Bottom chord (mm)',100.0,3000.0,480.0,10.0)
    bot_t=st.number_input('Bottom thickness (% chord)',5.0,30.0,12.5,0.5)
    bot_setout=st.number_input('Bottom forward edge from bow setout (mm)',0.0,20000.0,4370.0,10.0)
    span=st.number_input('Keel span — top to bottom (mm)',100.0,6000.0,2070.0,10.0)

# Share the current keel geometry with the combined Keel + Bulb page.
st.session_state['keel_design'] = {
    'top_series': top_series, 'top_chord': float(top_chord), 'top_t': float(top_t),
    'top_setout': float(top_setout), 'top_depth': float(top_depth),
    'bot_series': bot_series, 'bot_chord': float(bot_chord), 'bot_t': float(bot_t),
    'bot_setout': float(bot_setout), 'span': float(span),
}

bot_depth=top_depth+span
area_mm2=(top_chord+bot_chord)*0.5*span
ar=span**2/area_mm2
aft_top=top_setout+top_chord
aft_bottom=bot_setout+bot_chord
aft_delta=aft_bottom-aft_top

st.subheader('Geometry results')
r1,r2,r3,r4,r5=st.columns(5)
r1.metric('Bottom below LWL',f'{bot_depth:.0f} mm')
r2.metric('Keel span',f'{span:.0f} mm')
r3.metric('Planform area',f'{area_mm2/1e6:.3f} m²')
r4.metric('Aspect ratio',f'{ar:.2f}')
r5.metric('Aft-edge difference',f'{aft_delta:+.0f} mm')

if abs(aft_delta) < 0.5:
    st.success(f'Aft edge is parallel to bow setout line: top and bottom aft positions are both {aft_top:.0f} mm.')
else:
    st.warning(f'Aft edge is not parallel: top aft = {aft_top:.0f} mm, bottom aft = {aft_bottom:.0f} mm.')

st.plotly_chart(keel_planform_figure(top_setout,bot_setout,top_chord,bot_chord,top_depth,span),use_container_width=True)

st.subheader('Top and bottom foil sections')
st.plotly_chart(section_figure(top_chord,top_t,top_series,bot_chord,bot_t,bot_series),use_container_width=True)
st.write(f'**Top maximum thickness:** {top_chord*top_t/100:.1f} mm   |   **Bottom maximum thickness:** {bot_chord*bot_t/100:.1f} mm')

st.subheader('3D keel surface')
st.caption('Generated directly from the top and bottom NACA sections. Rotate, zoom and pan with the mouse. Intermediate section lines show the linear transition down the span.')
st.plotly_chart(keel_3d_figure(top_setout,bot_setout,top_chord,bot_chord,top_depth,span,
                              top_t,bot_t,top_series,bot_series),use_container_width=True)

st.divider()
st.subheader('Hydrodynamic estimate')
h1,h2=st.columns(2)
with h1:
    speed_kn=st.slider('Boat speed (knots)',1.0,20.0,8.0,0.5)
with h2:
    alpha_deg=st.slider('Angle of attack / leeway (deg)',-8.0,8.0,3.0,0.25)

with st.expander('Advanced settings'):
    efficiency=st.number_input('Span efficiency factor (e)',0.50,1.00,0.90,0.01,
                               help='Finite-span correction used for lift slope and induced drag. Default 0.90 for preliminary keel-only comparison.')
    st.caption('Keel-only mode: hull and bulb end-plate effects are not included yet.')

h=hydro_results(top_chord,bot_chord,top_t,bot_t,top_series,bot_series,
                span,speed_kn,alpha_deg,efficiency=efficiency)

m1,m2,m3,m4,m5=st.columns(5)
m1.metric('Planform area',f'{h["planform_area_m2"]:.3f} m²')
m2.metric('Wetted area',f'{h["wetted_area_m2"]:.3f} m²')
m3.metric('Aspect ratio',f'{h["aspect_ratio"]:.2f}')
m4.metric('Lift',f'{h["lift_kgf"]:.1f} kg')
m5.metric('Total drag',f'{h["total_drag_kgf"]:.2f} kg')

d1,d2,d3,d4=st.columns(4)
d1.metric('Lift coefficient CL',f'{h["cl"]:.3f}')
d2.metric('Profile drag',f'{h["profile_drag_kgf"]:.2f} kg')
d3.metric('Induced drag',f'{h["induced_drag_kgf"]:.2f} kg')
d4.metric('Lift / Drag',f'{h["ld_ratio"]:.1f}')

with st.expander('Calculation assumptions'):
    st.write('Seawater density: 1025 kg/m³. Kinematic viscosity: 1.05 × 10⁻⁶ m²/s.')
    st.write('Lift uses a finite-wing lift slope corrected for aspect ratio and the selected span-efficiency factor.')
    st.write('Profile drag uses the ITTC-1957 turbulent friction line plus a simple foil form-factor correction based on thickness.')
    st.write('Induced drag uses CDi = CL² / (π e AR).')
    st.write('End-plate effects: hull and bulb effects are not included in keel-only calculations. They will be applied in the combined keel/bulb analysis.')
    st.write('Hull/keel junction, free-surface, heel, ventilation and detailed 3D pressure effects are not included. Use these figures for preliminary comparison rather than final CFD-level prediction.')

st.subheader('PDF design report')
pdf_bytes=make_pdf_report(top_setout,bot_setout,top_chord,bot_chord,top_depth,span,
                          top_t,bot_t,top_series,bot_series,speed_kn,alpha_deg,h)
if pdf_bytes:
    st.download_button('Create / download PDF design report',data=pdf_bytes,
                       file_name='Tboats_Keel_Design_Report.pdf',mime='application/pdf')
else:
    st.warning('PDF creation needs matplotlib installed in this Python environment.')

st.info('Next development step: combine the approved keel geometry with the bulb designer and then introduce installed hull/bulb end-plate effects.')
