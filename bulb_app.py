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


st.set_page_config(page_title='Tboats Bulb Analyser', layout='wide')

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


def bulb_thickness(x, t, max_thickness_pos, le_radius_factor=1.0):
    """Internal bulb thickness law with user-selected maximum-thickness position.

    Uses the existing rounded NACA-style base profile, then smoothly remaps the
    chord so the maximum thickness occurs at max_thickness_pos (fraction of
    bulb length). This keeps the generating method inside the program while
    exposing only the useful design controls.
    """
    x = np.asarray(x, dtype=float)
    target = float(np.clip(max_thickness_pos, 0.20, 0.65))
    xr = _series_chord_map(x, target)
    return naca_00xx_thickness(xr, t, le_radius_factor)


def _tail_blend(xs, base, tail_value, max_pos, length_m):
    """Create the Beaver Tail with a fair side curve and a true tangent aft radius.

    The fairing:
      - leaves the original bulb unchanged through 50% length,
      - reaches the selected Beaver Tail half-beam at 95%,
      - continues aft on the same fair tangent,
      - then joins a circular end radius which is tangent to the fair line and
        passes through the 100% aft centreline point.

    All tangency calculations are done in physical metres, so the end radius
    remains geometrically correct for any bulb length.
    """
    xs = np.asarray(xs, dtype=float)
    base = np.asarray(base, dtype=float).copy()

    if tail_value <= 0.0:
        return base

    grid = np.linspace(0.0, 1.0, 4001)
    base_grid = np.interp(grid, xs, base)

    x0 = 0.50
    x95 = 0.95

    b50 = float(np.interp(x0, grid, base_grid))
    b95_orig = float(np.interp(x95, grid, base_grid))
    b95_target = float(tail_value)

    # Original tangent at 50% in dy/d(fraction).
    def slope_at(frac):
        k = int(round(frac * (len(grid) - 1)))
        i0 = max(0, k - 3)
        i1 = min(len(grid) - 1, k + 3)
        dx = grid[i1] - grid[i0]
        return float((base_grid[i1] - base_grid[i0]) / dx) if dx > 0 else 0.0

    m50 = slope_at(x0)

    # A gentle aft slope at 95%.  This is deliberately modest so the fairing
    # does not develop a hollow before the end radius takes over.
    drop = max(b95_orig - b95_target, 0.0)
    m95 = -0.65 * drop / max(x95 - x0, 1e-9)

    # Cubic Hermite fair curve from 50% to 95%.
    h = x95 - x0

    def hermite_y(frac):
        u = (frac - x0) / h
        h00 = 2*u**3 - 3*u**2 + 1
        h10 = u**3 - 2*u**2 + u
        h01 = -2*u**3 + 3*u**2
        h11 = u**3 - u**2
        return h00*b50 + h10*h*m50 + h01*b95_target + h11*h*m95

    fair = base_grid.copy()
    mask = (grid >= x0) & (grid <= x95)
    fair[mask] = np.array([hermite_y(x) for x in grid[mask]])

    # Add controlled fullness around the 80% station to prevent a hollow.
    # The correction is zero at 50% and 95%, and peaks around 80%.
    # Its strength increases with Beaver Tail amount.
    tail_amount = np.clip((b95_orig - b95_target) / max(b95_orig, 1e-9), 0.0, 1.0)
    u = np.clip((grid - x0) / (x95 - x0), 0.0, 1.0)
    # Smooth bell shape centered near u=0.67 (~80% length).
    bump = (u**2) * ((1.0 - u)**2)
    if np.max(bump) > 0:
        bump /= np.max(bump)
    fullness = 0.22 * tail_amount * b50 * bump
    fair[mask] += fullness[mask]

    # Continue the 95% tangent aft as a construction line.
    # Work in PHYSICAL x coordinates for the circle geometry.
    m_phys = m95 / max(length_m, 1e-12)  # dy/dx in metres/metre

    # Construction-line value at the 100% station.
    y_at_100_line = b95_target + m95 * (1.0 - x95)

    # For a circle whose centre lies on the centreline and which passes through
    # the aft point (x=L, y=0), tangency to the construction line gives:
    # q = y_end_line / sqrt(1 + m^2), where q is the physical distance
    # forward from the aft point to the tangency station.
    s = math.sqrt(1.0 + m_phys*m_phys)
    q_m = max(y_at_100_line / max(s, 1e-12), 0.0)

    # Limit the tangent point to the final 5% region. If the selected tail is
    # very large, the natural circle could otherwise try to start forward of 95%.
    q_m = min(q_m, 0.05 * length_m)
    xt_m = length_m - q_m
    xt = xt_m / length_m

    # Height of the construction line at the tangency point.
    yt = b95_target + m95 * (xt - x95)
    yt = max(yt, 1e-9)

    # Circle centre on the centreline.  For a curve y(x), the radius to the
    # tangent point is normal to the curve: xc = xt + m*y.
    xc_m = xt_m + m_phys * yt
    radius = length_m - xc_m
    radius = max(radius, 1e-9)

    # Build the final curve.
    out = fair.copy()

    # Between 95% and tangency, follow the fair construction line.
    between = (grid > x95) & (grid < xt)
    out[between] = b95_target + m95 * (grid[between] - x95)

    # From tangency to 100%, use the true circular arc.
    arc = grid >= xt
    x_phys = grid[arc] * length_m
    inside = np.maximum(radius*radius - (x_phys - xc_m)**2, 0.0)
    out[arc] = np.sqrt(inside)

    out[grid >= 1.0] = 0.0
    out[grid < x0] = base_grid[grid < x0]
    out = np.maximum(out, 0.0)

    return np.interp(xs, grid, out)


def _base_width_at_and_slope(frac, width_m, max_thickness_pos, le_radius_factor=1.0, section_series='63'):
    """Return original no-tail half-width and d(half-width)/d(x/c) at frac."""
    grid = np.linspace(0.0, 1.0, 4001)
    unit = section_series_curve(grid, section_series, le_radius_factor)
    ref = max(float(np.max(unit)), 1e-12)
    base_curve = (width_m / 2.0) * unit / ref
    y = float(np.interp(frac, grid, base_curve))
    k = int(np.argmin(np.abs(grid - frac)))
    i0 = max(0, k - 2)
    i1 = min(len(grid) - 1, k + 2)
    dx = grid[i1] - grid[i0]
    slope = float((base_curve[i1] - base_curve[i0]) / dx) if abs(dx) > 1e-12 else 0.0
    return y, slope




def beaver_tail_half_width_at(frac, length_m, width_m, max_thickness_pos, le_radius_factor, beaver_tail_pct, section_series='63'):
    """Return local asymmetric half-width using the same full Beaver Tail curve as the mesh."""
    frac = float(np.clip(frac, 0.0, 1.0))
    grid = np.linspace(0.0, 1.0, 4001)
    shape = bulb_thickness(grid, 1.0, max_thickness_pos, le_radius_factor)
    shape[0] = 0.0
    ref = max(float(np.max(shape)), 1e-12)
    base = (width_m / 2.0) * shape / ref

    if beaver_tail_pct > 0.0:
        tail_half_width = (width_m * (beaver_tail_pct / 100.0)) / 2.0
        curve = _tail_blend(grid, base, tail_half_width, max_thickness_pos, length_m)
    else:
        curve = base

    curve[-1] = 0.0
    return float(np.interp(frac, grid, curve))

def build_parametric_bulb(length_m, shape_mode, max_thickness_pos, thickness_pct=None,
                           top_pct=None, bottom_pct=None, width_m=None,
                           le_radius_factor=1.0, beaver_tail_pct=0.0, nx=121, ntheta=72):
    """Build a closed parametric bulb.

    Symmetric:
      - one thickness percentage controls both vertical depth and beam
      - every transverse section is circular
      - nose and aft end collapse to exact centre points

    Non-symmetric:
      - top and bottom use the same NACA family but separate thickness percentages
      - one overall width is shared by the upper and lower halves
      - transverse sections are two joined half-ellipses
    """
    xs = np.linspace(0.0, 1.0, nx)

    if shape_mode == 'Symmetric':
        t = thickness_pct / 100.0
        r = bulb_thickness(xs, t, max_thickness_pos, le_radius_factor) * length_m
        r[0] = 0.0
        r[-1] = 0.0
        top_r = r.copy()
        bottom_r = r.copy()
        half_width = r.copy()          # circular transverse sections
    else:
        tt = top_pct / 100.0
        tb = bottom_pct / 100.0
        top_r = bulb_thickness(xs, tt, max_thickness_pos, le_radius_factor) * length_m
        bottom_r = bulb_thickness(xs, tb, max_thickness_pos, le_radius_factor) * length_m
        top_r[0] = 0.0
        bottom_r[0] = 0.0

        # Because top and bottom use the same NACA family, a unit-thickness
        # profile gives the common fore/aft width distribution.
        shape = bulb_thickness(xs, 1.0, max_thickness_pos, le_radius_factor)
        shape[0] = 0.0
        ref = max(float(np.max(shape)), 1e-12)
        half_width = (width_m / 2.0) * shape / ref

        # Beaver Tail is the BEAM AT 95% OF BULB LENGTH, entered as % of maximum beam.
        # Its fairing begins at 50% length. A true tangent circular radius closes
        # the fair side curve into the aft centre point at 100%.
        tail_half_width = (width_m * (beaver_tail_pct / 100.0)) / 2.0
        if beaver_tail_pct > 0.0:
            half_width = _tail_blend(xs, half_width, tail_half_width, max_thickness_pos, length_m)

        top_r[-1] = 0.0
        bottom_r[-1] = 0.0
        half_width[-1] = 0.0

    verts = [[0.0, 0.0, 0.0]]
    ring_starts = []

    has_beaver_tail = False
    last_ring_index = nx - 1
    for i in range(1, last_ring_index):
        ring_starts.append(len(verts))
        for j in range(ntheta):
            th = 2.0 * math.pi * j / ntheta
            sn = math.sin(th)
            y = half_width[i] * math.cos(th)
            z = top_r[i] * sn if sn >= 0.0 else bottom_r[i] * sn
            verts.append([xs[i] * length_m, y, z])

    faces = []

    first = ring_starts[0]
    for j in range(ntheta):
        b = first + j
        c = first + (j + 1) % ntheta
        faces.append([0, c, b])

    for rix in range(len(ring_starts) - 1):
        a0 = ring_starts[rix]
        b0 = ring_starts[rix + 1]
        for j in range(ntheta):
            a = a0 + j
            b = a0 + (j + 1) % ntheta
            c = b0 + (j + 1) % ntheta
            d = b0 + j
            faces += [[a, b, c], [a, c, d]]

    last = ring_starts[-1]
    aft_tip = len(verts)
    verts.append([length_m, 0.0, 0.0])
    for j in range(ntheta):
        a = last + j
        b = last + (j + 1) % ntheta
        faces.append([a, b, aft_tip])

    mesh = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=True)
    mesh.remove_unreferenced_vertices()
    return mesh


def bulb_figure(mesh, length_m, shape_mode, max_thickness_pos, thickness_pct=None,
                top_pct=None, bottom_pct=None, width_m=None, le_radius_factor=1.0,
                beaver_tail_pct=0.0,
                station_fracs=DRAWING_STATIONS):
    v, f = mesh.vertices, mesh.faces
    fig = go.Figure(data=[go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        opacity=0.62, name='Bulb'
    )])

    theta = np.linspace(0.0, 2.0 * math.pi, 181)
    sn = np.sin(theta)

    for frac in station_fracs:
        # End stations collapse to a point/edge, so only draw full transverse
        # section loops at the internal stations. The same station set is used
        # by the 2D drawing below.
        if frac <= 0.0 or frac >= 1.0:
            continue
        if shape_mode == 'Symmetric':
            r = float(bulb_thickness(np.array([frac]), thickness_pct / 100.0, max_thickness_pos, le_radius_factor)[0] * length_m)
            ry = r
            z = r * sn
        else:
            rt = float(bulb_thickness(np.array([frac]), top_pct / 100.0, max_thickness_pos, le_radius_factor)[0] * length_m)
            rb = float(bulb_thickness(np.array([frac]), bottom_pct / 100.0, max_thickness_pos, le_radius_factor)[0] * length_m)
            ry = beaver_tail_half_width_at(
                frac, length_m, width_m, max_thickness_pos, le_radius_factor, beaver_tail_pct
            )

            z = np.where(sn >= 0.0, rt * sn, rb * sn)

        fig.add_trace(go.Scatter3d(
            x=np.full_like(theta, frac * length_m),
            y=ry * np.cos(theta), z=z,
            mode='lines', name=f'{int(frac * 100):02d}% section'
        ))

    if shape_mode == 'Symmetric':
        title = f'Symmetric bulb — {thickness_pct:.1f}% thickness, max at {max_thickness_pos*100:.0f}% length'
    else:
        title = f'Asymmetric bulb — max thickness at {max_thickness_pos*100:.0f}% length, Beaver Tail at 95% = {beaver_tail_pct:.0f}% beam'

    fig.update_layout(
        title=title,
        height=720,
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='Length', autorange='reversed'),
            yaxis_title='Width', zaxis_title='Vertical',
            camera=dict(eye=dict(x=1.55, y=-1.85, z=1.05))
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0),
        margin=dict(l=0, r=0, t=80, b=0)
    )
    return fig


def bulb_section_dimensions(frac, length_m, shape_mode, max_thickness_pos,
                            thickness_pct=None, top_pct=None, bottom_pct=None,
                            width_m=None, le_radius_factor=1.0, beaver_tail_pct=0.0):
    """Return local half-width, top radius and bottom radius at a station."""
    frac = float(np.clip(frac, 0.0, 1.0))
    if shape_mode == 'Symmetric':
        r = float(bulb_thickness(np.array([frac]), thickness_pct / 100.0,
                                 max_thickness_pos, le_radius_factor)[0] * length_m)
        return r, r, r

    rt = float(bulb_thickness(np.array([frac]), top_pct / 100.0,
                              max_thickness_pos, le_radius_factor)[0] * length_m)
    rb = float(bulb_thickness(np.array([frac]), bottom_pct / 100.0,
                              max_thickness_pos, le_radius_factor)[0] * length_m)
    ry = beaver_tail_half_width_at(
        frac, length_m, width_m, max_thickness_pos, le_radius_factor, beaver_tail_pct
    )

    return ry, rt, rb


def bulb_2d_figure(length_m, shape_mode, max_thickness_pos, thickness_pct=None,
                   top_pct=None, bottom_pct=None, width_m=None,
                   le_radius_factor=1.0, beaver_tail_pct=0.0,
                   station_fracs=DRAWING_STATIONS):
    """Engineering-style 2D plan/profile/section drawing using the same geometry."""
    xs = np.linspace(0.0, 1.0, 501)
    x_m = xs * length_m

    plan_half = []
    top_line = []
    bottom_line = []
    for frac in xs:
        ry, rt, rb = bulb_section_dimensions(
            frac, length_m, shape_mode, max_thickness_pos,
            thickness_pct, top_pct, bottom_pct, width_m,
            le_radius_factor, beaver_tail_pct
        )
        plan_half.append(ry)
        top_line.append(rt)
        bottom_line.append(-rb)

    plan_half = np.asarray(plan_half)
    top_line = np.asarray(top_line)
    bottom_line = np.asarray(bottom_line)

    max_half = max(float(np.max(plan_half)), float(np.max(top_line)),
                   float(np.max(-bottom_line)), length_m * 0.03)
    gap = max_half * 1.7
    plan_c = gap * 1.65
    section_c = 0.0
    profile_c = -gap * 1.75

    fig = go.Figure()

    # Plan view
    fig.add_trace(go.Scatter(x=x_m, y=plan_c + plan_half, mode='lines',
                             line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x_m, y=plan_c - plan_half, mode='lines',
                             line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
    # Profile view
    fig.add_trace(go.Scatter(x=x_m, y=profile_c + top_line, mode='lines',
                             line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x_m, y=profile_c + bottom_line, mode='lines',
                             line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))

    # Centre lines
    fig.add_trace(go.Scatter(x=[0, length_m], y=[plan_c, plan_c], mode='lines',
                             line=dict(color='red', width=1), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[0, length_m], y=[profile_c, profile_c], mode='lines',
                             line=dict(color='red', width=1), showlegend=False, hoverinfo='skip'))

    # Common station lines and transverse sections.
    th = np.linspace(0, 2*math.pi, 181)
    for frac in station_fracs:
        x = frac * length_m
        fig.add_vline(x=x, line_width=1, line_color='red', opacity=0.8)
        if 0.0 < frac < 1.0:
            ry, rt, rb = bulb_section_dimensions(
                frac, length_m, shape_mode, max_thickness_pos,
                thickness_pct, top_pct, bottom_pct, width_m,
                le_radius_factor, beaver_tail_pct
            )
            sn = np.sin(th)
            sec_y = ry * np.cos(th)
            sec_z = np.where(sn >= 0.0, rt * sn, rb * sn)
            fig.add_trace(go.Scatter(
                x=x + sec_y, y=section_c + sec_z, mode='lines',
                line=dict(color='black', width=1.5), showlegend=False, hoverinfo='skip'
            ))

    # Boat-design convention: 0% is the forward/nose station and is displayed on the RIGHT;
    # 100% is the aft end and is displayed on the LEFT.
    for frac in station_fracs:
        fig.add_annotation(x=frac*length_m, y=profile_c-gap*0.76,
                           text=f'{int(frac*100):02d}', showarrow=False,
                           font=dict(size=13, color='black'))
    fig.add_annotation(x=length_m*0.50, y=plan_c-gap*0.70, text='Plan',
                       showarrow=False, font=dict(size=20, color='black'))
    fig.add_annotation(x=length_m*0.50, y=profile_c-gap*1.02, text='Profile',
                       showarrow=False, font=dict(size=20, color='black'))
    fig.add_annotation(x=length_m*1.035, y=profile_c-gap*0.76, text='% length',
                       showarrow=False, xanchor='left', font=dict(size=13, color='black'))

    fig.update_xaxes(range=[length_m*1.08, -length_m*0.03], visible=False,
                     scaleanchor='y', scaleratio=1)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title='2D bulb drawing — plan, sections and profile',
        height=800,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=20, r=40, t=60, b=25)
    )
    return fig


def build_bulb_pdf(mesh, metrics, length_m, shape_mode, max_thickness_pos,
                   thickness_pct=None, top_pct=None, bottom_pct=None,
                   width_m=None, le_radius_factor=1.0, beaver_tail_pct=0.0):
    """Create a clean PDF design record. Returns bytes or None if matplotlib is unavailable."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
        ax.set_aspect('equal', adjustable='box')

        xs = np.linspace(0.0, 1.0, 501)
        x_mm = xs * length_m * 1000.0
        plan_half = []
        top_line = []
        bottom_line = []
        for frac in xs:
            ry, rt, rb = bulb_section_dimensions(
                frac, length_m, shape_mode, max_thickness_pos,
                thickness_pct, top_pct, bottom_pct, width_m,
                le_radius_factor, beaver_tail_pct
            )
            plan_half.append(ry*1000.0)
            top_line.append(rt*1000.0)
            bottom_line.append(-rb*1000.0)

        plan_half = np.asarray(plan_half)
        top_line = np.asarray(top_line)
        bottom_line = np.asarray(bottom_line)
        max_half = max(np.max(plan_half), np.max(top_line), np.max(-bottom_line), 1.0)
        gap = max_half * 1.7
        plan_c = gap * 1.65
        section_c = 0.0
        profile_c = -gap * 1.75

        ax.plot(x_mm, plan_c + plan_half, 'k-', lw=1.2)
        ax.plot(x_mm, plan_c - plan_half, 'k-', lw=1.2)
        ax.plot(x_mm, profile_c + top_line, 'k-', lw=1.2)
        ax.plot(x_mm, profile_c + bottom_line, 'k-', lw=1.2)
        ax.plot([0, length_m*1000], [plan_c, plan_c], 'r-', lw=0.5)
        ax.plot([0, length_m*1000], [profile_c, profile_c], 'r-', lw=0.5)

        th = np.linspace(0, 2*math.pi, 181)
        for frac in DRAWING_STATIONS:
            x = frac * length_m * 1000.0
            ax.axvline(x, color='red', lw=0.45, alpha=0.8)
            if 0.0 < frac < 1.0:
                ry, rt, rb = bulb_section_dimensions(
                    frac, length_m, shape_mode, max_thickness_pos,
                    thickness_pct, top_pct, bottom_pct, width_m,
                    le_radius_factor, beaver_tail_pct
                )
                sn = np.sin(th)
                sec_y = ry*1000.0*np.cos(th)
                sec_z = np.where(sn >= 0.0, rt*1000.0*sn, rb*1000.0*sn)
                ax.plot(x + sec_y, section_c + sec_z, 'k-', lw=0.8)
            ax.text(x, profile_c-gap*0.78, f'{int(frac*100):02d}', ha='center', va='top', fontsize=8)

        ax.text(length_m*500, plan_c-gap*0.72, 'Plan', ha='center', fontsize=14)
        ax.text(length_m*500, profile_c-gap*1.05, 'Profile', ha='center', fontsize=14)
        ax.text(length_m*1030, profile_c-gap*0.78, '% length', ha='left', va='top', fontsize=8)
        # Boat-design convention: bow/forward (0%) on the right, aft (100%) on the left.
        ax.set_xlim(length_m*1000*1.08, -length_m*1000*0.03)
        ax.set_ylim(profile_c-gap*1.15, plan_c+max_half*1.35)
        ax.axis('off')

        shape_text = 'Symmetric' if shape_mode == 'Symmetric' else 'Asymmetric'
        fig.suptitle(f'Tboats Bulb Analyser — {shape_text} bulb', fontsize=16, y=0.96)
        metric_text = (f"Length {metrics['Length (mm)']:.1f} mm    Width {metrics['Width (mm)']:.1f} mm    "
                       f"Depth {metrics['Depth (mm)']:.1f} mm    Lead weight {metrics['Lead weight (kg)']:.1f} kg")
        fig.text(0.06, 0.035, metric_text, fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

    buf.seek(0)
    return buf.getvalue()


def mesh_metrics(mesh, lead_density):
    bounds = mesh.bounds
    ext = bounds[1] - bounds[0]
    watertight = bool(mesh.is_watertight)
    volume = abs(mesh.volume) if watertight else np.nan
    area = mesh.area
    mass = volume * lead_density if watertight else np.nan
    centroid = mesh.center_mass if watertight else mesh.centroid
    return {
        'Watertight': watertight,
        'Length (mm)': ext[0]*1000,
        'Width (mm)': ext[1]*1000,
        'Depth (mm)': ext[2]*1000,
        'Surface area (m²)': area,
        'Volume (m³)': volume,
        'Lead weight (kg)': mass,
        'Centroid X (mm)': centroid[0]*1000,
    }


def projected_frontal_area(mesh):
    # Approximate projected area normal to X using convex hull of YZ projection
    pts = mesh.vertices[:,1:3]
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        return hull.volume  # 2D hull 'volume' is area
    except Exception:
        yspan = np.ptp(pts[:,0])
        zspan = np.ptp(pts[:,1])
        return math.pi*(yspan/2)*(zspan/2)


def zero_pitch_section_cl(section_series, top_pct, bottom_pct, max_thickness_pos, le_radius_factor=1.0):
    """Thin-airfoil estimate of CL at 0 deg from the longitudinal camber line.

    This is a comparison estimate for the asymmetric bulb section, not a CFD
    prediction of the complete 3-D bulb/keel system.
    """
    if top_pct is None or bottom_pct is None or abs(top_pct-bottom_pct) < 1e-12:
        return 0.0

    # Cosine-spaced chord points avoid the nose singularity.
    theta = np.linspace(1e-4, np.pi-1e-4, 1600)
    x = 0.5 * (1.0 - np.cos(theta))

    unit = section_series_curve(x, section_series, le_radius_factor)
    # section_series_curve is normalized to peak half-thickness = 1.
    # Top/bottom percentages are total section-depth percentages in the app.
    z_top = unit * (top_pct / 200.0)
    z_bottom = -unit * (bottom_pct / 200.0)
    z_camber = 0.5 * (z_top + z_bottom)

    dzdx = np.gradient(z_camber, x)

    # Standard thin-airfoil Fourier coefficients at alpha = 0.
    A0 = -(1.0/np.pi) * np.trapz(dzdx, theta)
    A1 =  (2.0/np.pi) * np.trapz(dzdx*np.cos(theta), theta)
    return float(2.0*np.pi*(A0 + 0.5*A1))


def add_zero_pitch_lift(dfd, shape_mode, section_series, top_pct, bottom_pct,
                        max_thickness_pos, length_m, max_width_m):
    """Simple provisional zero-pitch bulb lift comparison.

    For this stage of development the asymmetric reference bulb is calibrated
    to 10 kgf at 14 knots. Lift then varies with speed squared. A symmetric
    bulb returns zero at 0 degrees.

    Pitch/angle-of-attack effects will be added later with the keel model.
    """
    if shape_mode != 'Asymmetric' or abs(top_pct-bottom_pct) < 1e-12:
        lift_kgf = np.zeros(len(dfd), dtype=float)
    else:
        speeds = dfd['Speed (kn)'].to_numpy(dtype=float)
        reference_beam_m = 0.100
        beam_factor = max_width_m / reference_beam_m
        lift_kgf = 10.0 * beam_factor * (speeds / 14.0) ** 2

    out = dfd.copy()
    out['Estimated lift at 0° (N)'] = lift_kgf * 9.80665
    out['Estimated lift at 0° (kgf)'] = lift_kgf
    return out, np.nan


def drag_table(mesh, speeds_knots, form_factor=1.15):
    L = max(np.ptp(mesh.vertices[:,0]), 1e-6)
    S = mesh.area
    rows = []
    for kts in speeds_knots:
        v = kts * 0.514444
        Re = v * L / SEAWATER_NU
        cf = 0.075 / (max(math.log10(Re)-2, 1e-6)**2)
        rf = 0.5 * SEAWATER_DENSITY * v*v * S * cf
        total = rf * form_factor
        rows.append({
            'Speed (kn)': kts,
            'Reynolds number': Re,
            'Cf': cf,
            'Skin friction (N)': rf,
            'Estimated total drag (N)': total,
            'Estimated total drag (kgf)': total/9.80665,
        })
    return pd.DataFrame(rows)


def mesh_figure(mesh, title='Bulb geometry'):
    v = mesh.vertices
    f = mesh.faces
    fig = go.Figure(data=[go.Mesh3d(
        x=v[:,0], y=v[:,1], z=v[:,2],
        i=f[:,0], j=f[:,1], k=f[:,2],
        opacity=0.75
    )])
    fig.update_layout(title=title, scene_aspectmode='data', margin=dict(l=0,r=0,t=40,b=0))
    return fig


st.title('Tboats Bulb Analyser — Prototype')
st.caption('Geometry first. Hydrodynamic estimates are preliminary and intended for comparison, not as a replacement for CFD.')

mode = st.radio('Choose input method', ['Upload STL', 'Create bulb'], horizontal=True)
lead_density = st.number_input('Lead density (kg/m³)', min_value=9000.0, max_value=12000.0, value=LEAD_DENSITY_DEFAULT, step=10.0)
mesh = None

if mode == 'Upload STL':
    upl = st.file_uploader('Upload a closed STL bulb model', type=['stl'])
    if upl is not None:
        try:
            mesh = trimesh.load(io.BytesIO(upl.read()), file_type='stl', force='mesh')
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            mesh.process(validate=True)
        except Exception as e:
            st.error(f'Could not read STL: {e}')
else:
    st.subheader('Parametric bulb')
    st.caption('Set the bulb proportions directly. The generating curve stays inside the program.')

    c1, c2 = st.columns(2)
    with c1:
        shape_mode = st.radio('Bulb shape', ['Symmetric', 'Asymmetric'], horizontal=True)
    with c2:
        section_series = st.selectbox(
        'Section series',
        ['00', '63', '64', '65', '66'],
        index=1,
        help='Select the base section family. 63, 64, 65 and 66 use published 12%-thickness reference ordinate shapes, normalized to your selected thickness.'
    )
    if section_series == '00':
        max_thickness_pos_pct = st.number_input(
            'Position of maximum thickness (% from front)',
            min_value=20.0, max_value=60.0, value=30.0, step=1.0
        )
        max_thickness_pos = max_thickness_pos_pct / 100.0
    else:
        max_thickness_pos = SECTION_SERIES[section_series]['max_pos']
        st.caption(
            f"{section_series}-series uses the published reference ordinate shape, scaled to your selected thickness."
            if section_series in ["63", "64", "65", "66"]
            else "00-series uses the existing smooth NACA-style section."
        )

    length_mm = st.number_input('Bulb length (mm)', 200.0, 5000.0, 1900.0, 10.0)

    if shape_mode == 'Symmetric':
        thickness_pct = st.number_input('Thickness / diameter (% of bulb length)', 5.0, 30.0, 13.0, 0.5)
        max_diameter_mm = length_mm * thickness_pct / 100.0

        t_decimal = thickness_pct / 100.0
        standard_le_pct = standard_le_radius_ratio(t_decimal) * 100.0
        le_radius_factor = st.number_input(
            'LE radius factor (1.00 = standard NACA)',
            min_value=0.50, max_value=2.00, value=1.00, step=0.05
        )
        le_radius_pct = standard_le_pct * le_radius_factor
        le_radius_mm = length_mm * le_radius_pct / 100.0

        st.write(f'**Symmetric bulb — {thickness_pct:.1f}% thickness, maximum at {max_thickness_pos*100:.0f}% of length**')
        st.caption(
            f'Maximum height = maximum width = {max_diameter_mm:.1f} mm. '
            f'Standard LE radius = {standard_le_pct:.2f}% of length; '
            f'current LE radius = {le_radius_pct:.2f}% = {le_radius_mm:.1f} mm. '
            'Every transverse section is circular and the aft end closes to the exact centre point.'
        )
        top_pct = bottom_pct = width_mm = None
        beaver_tail_pct = 0.0
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            top_pct = st.number_input('Top thickness (%)', 2.0, 30.0, 13.0, 0.5)
        with c2:
            bottom_pct = st.number_input('Bottom thickness (%)', 2.0, 30.0, 13.0, 0.5)
        with c3:
            width_mm = st.number_input('Maximum width (mm)', 50.0, 1000.0, 247.0, 5.0)
        thickness_pct = None
        le_radius_factor = 1.0
        beaver_tail_pct = st.slider('Beaver Tail width at 95% length (% of maximum beam)', 0, 60, 0, 1)
        beaver_tail_mm = width_mm * beaver_tail_pct / 100.0
        st.write(f'**Asymmetric bulb — {section_series} section series**')
        st.caption(
            f'Top and bottom have separate depth controls. Requested Beaver Tail beam at 95% length = '
            f'{beaver_tail_pct}% of maximum beam = {beaver_tail_mm:.1f} mm. '
            'The 50% station stays on the original bulb. Extra fullness is added smoothly around the 80% station to avoid a hollow. From there the planform is faired aft as one '
            'continuous curve. The rounded aft end is then created as a tangent radius to that fair curve, '
            'so the radius join point is calculated automatically rather than being forced to start at 95%.'
        )

    # Share the current parametric bulb profile with the combined Keel + Bulb page.
    # Store the actual profile ordinates so the combined page uses exactly the
    # same bulb shape rather than recreating or approximating it.
    _cx = np.linspace(0.0, 1.0, 501)
    if shape_mode == 'Symmetric':
        _r = bulb_thickness(_cx, thickness_pct / 100.0, max_thickness_pos, le_radius_factor) * length_mm
        _top = _r.copy(); _bottom = _r.copy()
    else:
        _top = bulb_thickness(_cx, top_pct / 100.0, max_thickness_pos, le_radius_factor) * length_mm
        _bottom = bulb_thickness(_cx, bottom_pct / 100.0, max_thickness_pos, le_radius_factor) * length_mm
    _top[0] = _top[-1] = 0.0
    _bottom[0] = _bottom[-1] = 0.0
    st.session_state['bulb_design'] = {
        'shape_mode': shape_mode, 'section_series': section_series,
        'length_mm': float(length_mm), 'max_thickness_pos': float(max_thickness_pos),
        'thickness_pct': None if thickness_pct is None else float(thickness_pct),
        'top_pct': None if top_pct is None else float(top_pct),
        'bottom_pct': None if bottom_pct is None else float(bottom_pct),
        'width_mm': None if width_mm is None else float(width_mm),
        'beaver_tail_pct': float(beaver_tail_pct),
        'profile_x_mm': (_cx * length_mm).tolist(),
        'profile_top_mm': _top.tolist(), 'profile_bottom_mm': _bottom.tolist(),
    }

    try:
        if shape_mode == 'Symmetric':
            mesh = build_parametric_bulb(
                length_mm / 1000.0, shape_mode, max_thickness_pos, thickness_pct=thickness_pct,
                le_radius_factor=le_radius_factor
            )
        else:
            mesh = build_parametric_bulb(
                length_mm / 1000.0, shape_mode, max_thickness_pos,
                top_pct=top_pct, bottom_pct=bottom_pct, width_m=width_mm / 1000.0,
                le_radius_factor=le_radius_factor, beaver_tail_pct=beaver_tail_pct
            )
    except Exception as e:
        st.error(str(e))

if mesh is not None:
    m = mesh_metrics(mesh, lead_density)

    # Share exact mesh-derived bulb mass properties with the combined page.
    if mode == 'Create bulb' and mesh.is_watertight and 'bulb_design' in st.session_state:
        st.session_state['bulb_design']['volume_m3'] = float(abs(mesh.volume))
        st.session_state['bulb_design']['centroid_x_mm'] = float(mesh.center_mass[0] * 1000.0)
        st.session_state['bulb_design']['centroid_z_mm'] = float(mesh.center_mass[2] * 1000.0)
        st.session_state['bulb_design']['lead_density_kg_m3'] = float(lead_density)
        st.session_state['bulb_design']['lead_weight_kg'] = float(abs(mesh.volume) * lead_density)

    st.subheader('Bulb visualisation')
    view3d, view2d = st.tabs(['3D View', '2D Drawing'])

    with view3d:
        if mode == 'Create bulb':
            st.plotly_chart(
                bulb_figure(
                    mesh, length_mm / 1000.0, shape_mode, max_thickness_pos,
                    thickness_pct=thickness_pct,
                    top_pct=top_pct, bottom_pct=bottom_pct,
                    width_m=(width_mm / 1000.0) if width_mm is not None else None,
                    le_radius_factor=le_radius_factor, beaver_tail_pct=beaver_tail_pct,
                    station_fracs=DRAWING_STATIONS
                ),
                use_container_width=True
            )
        else:
            fig3d = mesh_figure(mesh)
            fig3d.update_layout(height=720)
            st.plotly_chart(fig3d, use_container_width=True)

    with view2d:
        if mode == 'Create bulb':
            st.plotly_chart(
                bulb_2d_figure(
                    length_mm / 1000.0, shape_mode, max_thickness_pos,
                    thickness_pct=thickness_pct,
                    top_pct=top_pct, bottom_pct=bottom_pct,
                    width_m=(width_mm / 1000.0) if width_mm is not None else None,
                    le_radius_factor=le_radius_factor,
                    beaver_tail_pct=beaver_tail_pct,
                    station_fracs=DRAWING_STATIONS
                ),
                use_container_width=True
            )
            st.caption('The 2D and 3D views use the same 0, 5, 20, 35, 50, 65, 80, 95 and 100% stations.')
        else:
            st.info('2D drawing is currently available for bulbs created in the app.')

    st.subheader('Geometry results')
    dfm = pd.DataFrame({'Parameter': list(m.keys()), 'Value': list(m.values())})
    st.dataframe(dfm, hide_index=True, use_container_width=True)
    if not m['Watertight']:
        st.warning('This mesh is not watertight. Surface dimensions and area are usable, but volume and lead weight are not reliable.')

    if mode == 'Create bulb':
        st.subheader('Export')
        e1, e2 = st.columns(2)
        with e1:
            export_mesh = mesh.copy()
            export_mesh.apply_scale(1000.0)  # STL coordinates in millimetres
            stl_bytes = export_mesh.export(file_type='stl')
            if isinstance(stl_bytes, str):
                stl_bytes = stl_bytes.encode('utf-8')
            st.download_button(
                'Export STL',
                data=stl_bytes,
                file_name='Tboats_bulb.stl',
                mime='model/stl',
                use_container_width=True
            )
            st.caption('Exports the current bulb as a closed STL with coordinates in millimetres.')

        with e2:
            pdf_bytes = build_bulb_pdf(
                mesh, m, length_mm / 1000.0, shape_mode, max_thickness_pos,
                thickness_pct=thickness_pct,
                top_pct=top_pct, bottom_pct=bottom_pct,
                width_m=(width_mm / 1000.0) if width_mm is not None else None,
                le_radius_factor=le_radius_factor, beaver_tail_pct=beaver_tail_pct
            )
            if pdf_bytes is not None:
                st.download_button(
                    'Print / Save PDF',
                    data=pdf_bytes,
                    file_name='Tboats_bulb_drawing.pdf',
                    mime='application/pdf',
                    use_container_width=True
                )
                st.caption('Creates an A4 landscape drawing using the same stations as the 2D and 3D views.')
            else:
                st.button('Print / Save PDF', disabled=True, use_container_width=True)
                st.caption('Install matplotlib to enable PDF export: python3 -m pip install matplotlib')

    st.subheader('Top and bottom overview')
    verts = mesh.vertices
    top = verts[verts[:,2] >= 0]
    bottom = verts[verts[:,2] < 0]
    top_area_approx = mesh.area * len(top)/max(len(verts),1)
    bottom_area_approx = mesh.area * len(bottom)/max(len(verts),1)
    st.write(f'Approx. vertex split: **{len(top):,} top / {len(bottom):,} bottom**. This will be replaced by a proper surface-area split in the next version.')

    st.subheader('Preliminary drag estimate')
    c1,c2 = st.columns(2)
    with c1:
        min_speed = st.number_input('Minimum speed (kn)', 1.0, 20.0, 4.0, 1.0)
        max_speed = st.number_input('Maximum speed (kn)', 2.0, 30.0, 14.0, 1.0)
    with c2:
        form_factor = st.slider('Form factor multiplier', 1.00, 2.00, 1.15, 0.05)
        npts = st.slider('Number of speed points', 3, 15, 6, 1)
    speeds = np.linspace(min_speed, max_speed, npts)
    dfd = drag_table(mesh, speeds, form_factor)

    # Add zero-pitch lift estimate to the existing output table.
    if mode == 'Create bulb':
        if shape_mode == 'Asymmetric':
            lift_width_m = width_mm / 1000.0
        else:
            lift_width_m = (length_mm / 1000.0) * (thickness_pct / 100.0)
        dfd, cl0 = add_zero_pitch_lift(
            dfd, shape_mode, section_series,
            top_pct, bottom_pct, max_thickness_pos,
            length_mm / 1000.0, lift_width_m
        )
    else:
        dfd['Estimated lift at 0° (N)'] = np.nan
        dfd['Estimated lift at 0° (kgf)'] = np.nan
        cl0 = np.nan

    st.dataframe(dfd.style.format({
        'Speed (kn)': '{:.1f}', 'Reynolds number': '{:.3e}', 'Cf': '{:.5f}',
        'Skin friction (N)': '{:.2f}', 'Estimated total drag (N)': '{:.2f}',
        'Estimated total drag (kgf)': '{:.2f}', 'Estimated lift at 0° (N)': '{:.2f}',
        'Estimated lift at 0° (kgf)': '{:.2f}'
    }), use_container_width=True)

    if mode == 'Create bulb':
        st.caption(
            'Provisional comparison model: asymmetric bulb calibrated to 10.0 kgf lift at 14 kn '
            'and scaled with speed². Symmetric bulb = 0 lift at 0°. '
            'Pitch and angle-of-attack effects will be added later with the keel model.'
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfd['Speed (kn)'], y=dfd['Estimated total drag (kgf)'], mode='lines+markers', name='Estimated drag'))
    fig.update_layout(xaxis_title='Speed (knots)', yaxis_title='Drag (kgf)', title='Estimated bulb drag vs speed')
    st.plotly_chart(fig, use_container_width=True)

    st.info('Next planned additions: keel geometry, combined keel + bulb analysis, angle-of-attack sweep, heel, leeway, and side-by-side comparison.')
