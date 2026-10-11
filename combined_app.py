import io
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

st.set_page_config(page_title='Tboats Keel + Bulb', layout='wide')
# File actions are displayed in the heading but populated after calculations.
advanced = st.session_state.get('app_access') == 'advanced'
heading, download_col, upload_col, pdf_col = st.columns([4.5, 1.2, 1.2, 0.9], vertical_alignment='center')
with heading:
    st.title('Tboat Keel + Bulb Designer')
with download_col:
    download_slot = st.empty()
with upload_col:
    if advanced:
        with st.popover('Upload', use_container_width=True):
            uploaded_combined = st.file_uploader('Upload a saved combined design', type=['json'], key='combined_file_upload')
    else:
        st.button('Upload', disabled=True, use_container_width=True)
        uploaded_combined = None
with pdf_col:
    pdf_slot = st.empty()

nav_a, nav_b, nav_c, guide_col = st.columns([1.5, 1.5, 1.5, 1.0], vertical_alignment='center')
with nav_a:
    st.page_link('keel_app.py', label='Keel Design', use_container_width=True)
with nav_b:
    st.page_link('bulb_app.py', label='Bulb Design', use_container_width=True)
with nav_c:
    st.page_link('combined_app.py', label='Keel + Bulb', use_container_width=True)
with guide_col:
    if MATPLOTLIB_AVAILABLE:
        guide_buffer = io.BytesIO()
        with PdfPages(guide_buffer) as guide_doc:
            page = plt.figure(figsize=(8.27, 11.69))
            page.suptitle('Tboat Keel + Bulb Designer — Guide', fontsize=17, y=.95)
            guide_text = (
                'PURPOSE\n'
                'Combine the current keel and bulb designs into one installation.\n'
                'A default design appears until you create or load your own components.\n\n'
                'WORKFLOW\n'
                '1. Set the keel geometry in Keel Design.\n'
                '2. Set the bulb geometry in Bulb Design.\n'
                '3. Return to Keel + Bulb to position the bulb beneath the keel.\n'
                '4. Adjust bulb bottom draft and bulb leading-edge setout.\n'
                '5. Check the dimensioned side elevation and 3D view.\n'
                '6. Compare lift, drag, centres of volume and righting moment.\n\n'
                'POSITIONING INPUTS\n'
                'Bulb bottom draft: vertical distance below the load waterline (mm).\n'
                'Bulb leading-edge setout: distance aft from bow setout datum (mm).\n'
                'Overlap and gap colouring is a positioning aid only.\n\n'
                'ADVANCED FILES\n'
                'Download: saves combined keel, bulb and positioning data as JSON.\n'
                'Upload: restores a previously downloaded combined JSON design.\n'
                'PDF: generates a three-page comparison report.\n'
                'Standard access permits on-screen design without file downloads.\n\n'
                'CALCULATION LIMITATIONS\n'
                'Hydrodynamic results are preliminary comparisons, not CFD.\n'
                'Righting moment is ballast-only; hull, rig and crew are excluded.\n'
                'Check all dimensions and engineering independently before manufacture.'
            )
            page.text(.09, .88, guide_text, va='top', fontsize=10, linespacing=1.55)
            guide_doc.savefig(page)
            plt.close(page)
        st.download_button('Guide', guide_buffer.getvalue(), file_name='Tboat_Keel_Bulb_Guide.pdf',
                           mime='application/pdf', use_container_width=True)
    else:
        with st.popover('Guide', use_container_width=True):
            st.write('Design the keel and bulb separately, then position them here. Check 2D/3D views, analysis and righting moment. Advanced users can save JSON and PDF reports.')

st.caption(f"Signed in: {st.session_state.get('customer_email', '')}  •  Access: {str(st.session_state.get('app_access', '')).title()}")
st.caption('Uses the current Keel Design and Bulb Design. A saved default keel and bulb are loaded automatically so the combined drawing is always available.')

# Saved default keel — same geometry as the Keel Designer defaults.
default_keel = {
    'top_series':'63','top_chord':550.0,'top_t':13.0,'top_setout':4300.0,'top_depth':280.0,
    'bot_series':'63','bot_chord':480.0,'bot_t':12.5,'bot_setout':4370.0,'span':2070.0,
}

# Saved default bulb — 1900 mm long, 13% symmetric 63-series style profile.
# This is only used until the user visits Bulb Design; thereafter the actual saved bulb profile is used.
def _default_bulb():
    L=1900.0
    x=np.linspace(0.0,1.0,501)
    # Smooth NACA 00xx thickness envelope, scaled to 13% total thickness.
    t=0.13
    yt=5*t*(0.2969*np.sqrt(np.maximum(x,0))-0.1260*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
    yt=np.maximum(yt,0.0)*L
    yt[0]=yt[-1]=0.0
    return {
        'shape_mode':'Symmetric','section_series':'63','length_mm':L,'max_thickness_pos':0.35,
        'thickness_pct':13.0,'top_pct':None,'bottom_pct':None,'width_mm':247.0,'beaver_tail_pct':0.0,
        'profile_x_mm':(x*L).tolist(),'profile_top_mm':yt.tolist(),'profile_bottom_mm':yt.tolist(),
    }

# Restore only a validated Tboat combined design. Preserve existing component data on failure.
if advanced and uploaded_combined is not None:
    raw = uploaded_combined.getvalue()
    signature = (uploaded_combined.name, raw)
    if st.session_state.get('_combined_loaded_file') != signature:
        try:
            saved = json.loads(raw.decode('utf-8'))
            if saved.get('type') != 'tboat_combined_design' or saved.get('version') != 1:
                raise ValueError('This is not a supported Tboat combined design file.')
            saved_keel, saved_bulb = saved['keel'], saved['bulb']
            keel_fields = ('top_series','top_chord','top_t','top_setout','top_depth','bot_series','bot_chord','bot_t','bot_setout','span')
            bulb_fields = ('length_mm','profile_x_mm','profile_top_mm','profile_bottom_mm')
            if not isinstance(saved_keel, dict) or not all(k in saved_keel for k in keel_fields):
                raise ValueError('Keel data is incomplete.')
            if not isinstance(saved_bulb, dict) or not all(k in saved_bulb for k in bulb_fields):
                raise ValueError('Bulb data is incomplete.')
            draft = float(saved['bulb_bottom_draft_mm'])
            setout = float(saved['bulb_le_setout_mm'])
            if not 500 <= draft <= 10000 or not 0 <= setout <= 20000:
                raise ValueError('Bulb positioning is outside the supported range.')
            for key in keel_fields:
                if key not in ('top_series','bot_series'):
                    float(saved_keel[key])
            for key in bulb_fields:
                if key != 'length_mm':
                    if not isinstance(saved_bulb[key], list) or len(saved_bulb[key]) < 2:
                        raise ValueError('Bulb profile is invalid.')
            st.session_state['keel_design'] = saved_keel
            st.session_state['bulb_design'] = saved_bulb
            st.session_state['combined_bulb_bottom_draft'] = draft
            st.session_state['combined_bulb_le_setout'] = setout
            st.session_state['_combined_loaded_file'] = signature
            st.success('Combined design loaded.')
        except (KeyError, ValueError, TypeError, UnicodeDecodeError, OverflowError) as exc:
            st.error(f'Unable to load combined design: {exc}')

if 'keel_design' not in st.session_state:
    st.session_state['keel_design']=default_keel.copy()
if 'bulb_design' not in st.session_state:
    st.session_state['bulb_design']=_default_bulb()

keel=st.session_state['keel_design']
bulb=st.session_state['bulb_design']

c1,c2=st.columns(2)
with c1:
    bulb_bottom_draft=st.number_input('Bulb bottom draft below LWL (mm)',500.0,10000.0,2597.0,1.0,
        key='combined_bulb_bottom_draft',
        help='Vertical distance from LWL to the lowest point of the bulb.')
with c2:
    bulb_le_setout=st.number_input('Bulb leading edge from bow setout (mm)',0.0,20000.0,3850.0,10.0,
        key='combined_bulb_le_setout',
        help='Horizontal distance aft from the bow setout line to the bulb leading edge.')

# Keel geometry
ktx=float(keel['top_setout']); ktc=float(keel['top_chord']); kty=float(keel['top_depth'])
kbx=float(keel['bot_setout']); kbc=float(keel['bot_chord']); kby=kty+float(keel['span'])

# Bulb profile translated into boat coordinates.
bx_local=np.asarray(bulb['profile_x_mm'],dtype=float)
btop_off=np.asarray(bulb['profile_top_mm'],dtype=float)
bbot_off=np.asarray(bulb['profile_bottom_mm'],dtype=float)
centre_depth=float(bulb_bottom_draft)-float(np.max(bbot_off))
bx=float(bulb_le_setout)+bx_local
btop=centre_depth-btop_off
bbot=centre_depth+bbot_off
bulb_te=float(bulb_le_setout)+float(bulb['length_mm'])

# Keel polygon: forward/top -> aft/top -> aft/bottom -> forward/bottom.
kx=[ktx,ktx+ktc,kbx+kbc,kbx,ktx]
ky=[kty,kty,kby,kby,kty]

fig=go.Figure()
fig.add_trace(go.Scatter(x=np.r_[bx,bx[::-1]],y=np.r_[btop,bbot[::-1]],fill='toself',mode='lines',
    name='Bulb',line=dict(width=2),opacity=0.55,hovertemplate='Setout %{x:.0f} mm<br>Depth %{y:.0f} mm<extra>Bulb</extra>'))
fig.add_trace(go.Scatter(x=kx,y=ky,fill='toself',mode='lines',name='Keel',line=dict(width=3),opacity=0.55,
    hovertemplate='Setout %{x:.0f} mm<br>Depth %{y:.0f} mm<extra>Keel</extra>'))

# Visual overlap/gap at the keel/bulb junction.
mask=(bx>=kbx)&(bx<=kbx+kbc)
if np.any(mask):
    xx=bx[mask]; tt=btop[mask]
    inside=tt<=kby
    if np.any(inside):
        fig.add_trace(go.Scatter(x=np.r_[xx[inside],xx[inside][::-1]],
            y=np.r_[tt[inside],np.full(np.sum(inside),kby)[::-1]],fill='toself',mode='lines',
            line=dict(width=0),name='Keel inside bulb',opacity=0.38,hoverinfo='skip'))
    if np.any(~inside):
        fig.add_trace(go.Scatter(x=np.r_[xx[~inside],xx[~inside][::-1]],
            y=np.r_[np.full(np.sum(~inside),kby),tt[~inside][::-1]],fill='toself',mode='lines',
            line=dict(width=0),name='Gap',opacity=0.45,hoverinfo='skip'))

# Drawing extents. Bow setout line is deliberately NTS and always on RHS.
xmin=min(float(np.min(bx)),ktx,kbx)-300
xmax=max(float(np.max(bx)),ktx+ktc,kbx+kbc)+450
bow_display=min(float(np.min(bx)),ktx,kbx)-520  # smaller x displays on RHS because x axis is reversed
max_depth=max(float(bulb_bottom_draft),kby)

# LWL and bow setout line.
fig.add_shape(type='line',x0=xmax,x1=bow_display,y0=0,y1=0,line=dict(width=2))
fig.add_annotation(x=bow_display+120,y=0,text='LWL',showarrow=False,yshift=14)
fig.add_shape(type='line',x0=bow_display,x1=bow_display,y0=0,y1=max_depth+110,line=dict(width=2))
fig.add_annotation(x=bow_display,y=max_depth*0.48,text='Bow setout line (NTS)',textangle=-90,showarrow=False,xshift=-28)

# Dimension helper functions. Horizontal dimensions use actual values in labels but NTS display where appropriate.
def hdim(x0,x1,y,label):
    fig.add_shape(type='line',x0=x0,x1=x1,y0=y,y1=y,line=dict(width=1))
    tick=28
    fig.add_shape(type='line',x0=x0,x1=x0,y0=y-tick,y1=y+tick,line=dict(width=1))
    fig.add_shape(type='line',x0=x1,x1=x1,y0=y-tick,y1=y+tick,line=dict(width=1))
    fig.add_annotation(x=(x0+x1)/2,y=y,text=label,showarrow=False,yshift=-15)

def vdim(x,y0,y1,label):
    fig.add_shape(type='line',x0=x,x1=x,y0=y0,y1=y1,line=dict(width=1))
    tick=28
    fig.add_shape(type='line',x0=x-tick,x1=x+tick,y0=y0,y1=y0,line=dict(width=1))
    fig.add_shape(type='line',x0=x-tick,x1=x+tick,y0=y1,y1=y1,line=dict(width=1))
    fig.add_annotation(x=x,y=(y0+y1)/2,text=label,showarrow=False,textangle=-90,xshift=18)

# Keel dimensions above/beside foil.
hdim(ktx,ktx+ktc,kty-105,f'Top chord {ktc:.0f} mm')
fig.add_annotation(x=ktx-120,y=kty-105,text=f'Top setout {ktx:.0f} mm (NTS)',showarrow=False,xanchor='right')
vdim(xmax-70,0,kty,f'{kty:.0f} mm')
vdim(xmax-70,kty,kby,f'{float(keel["span"]):.0f} mm')

# Bulb/keel chain dimensions below the bulb, matching the user's sketch.
dim_y=max_depth+125
hdim(bow_display,float(bulb_le_setout),dim_y,f'Bulb LE setout {bulb_le_setout:.0f} mm (NTS)')
if kbx>=float(bulb_le_setout):
    hdim(float(bulb_le_setout),kbx,dim_y+85,f'{kbx-float(bulb_le_setout):.0f} mm')
hdim(kbx,kbx+kbc,dim_y+85,f'Bottom chord {kbc:.0f} mm')
if bulb_te>=kbx+kbc:
    hdim(kbx+kbc,bulb_te,dim_y+85,f'{bulb_te-(kbx+kbc):.0f} mm')

# Overall draft and bulb depth.
vdim(xmax+110,0,float(bulb_bottom_draft),f'Overall draft {bulb_bottom_draft:.0f} mm')
bulb_total_depth=float(np.max(btop_off)+np.max(bbot_off))
vdim(xmax+210,float(bulb_bottom_draft)-bulb_total_depth,float(bulb_bottom_draft),f'Bulb depth {bulb_total_depth:.0f} mm')

fig.update_layout(
    title='Combined keel + bulb side elevation — forward to RHS',height=830,
    xaxis=dict(title='',autorange='reversed',showgrid=False,zeroline=False,showticklabels=False),
    yaxis=dict(title='Depth below LWL (mm)',autorange='reversed',scaleanchor='x',scaleratio=1,showgrid=False,zeroline=False),
    legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0),
    margin=dict(l=80,r=120,t=90,b=80)
)
st.plotly_chart(fig,use_container_width=True)

keel_bottom_mid=kbx+kbc/2
bulb_mid=float(bulb_le_setout)+float(bulb['length_mm'])/2
r1,r2,r3,r4=st.columns(4)
r1.metric('Keel bottom below LWL',f'{kby:.0f} mm')
r2.metric('Bulb bottom draft',f'{bulb_bottom_draft:.0f} mm')
r3.metric('Bulb LE / TE',f'{bulb_le_setout:.0f} / {bulb_te:.0f} mm')
r4.metric('Bulb vs keel midpoint',f'{bulb_mid-keel_bottom_mid:+.0f} mm')

st.caption('Overlap colouring is a visual positioning aid. Because the bulb top is curved and the keel bottom is straight, some coloured intersection is normal. Adjust bulb draft and bulb LE setout until the attachment looks appropriate.')
st.info('The combined drawing updates from the current Keel Design and Bulb Design. Next step after approving this drawing: combined interactive 3D geometry.')

# ---------------- Combined preliminary analysis ----------------
st.divider()
st.subheader('Combined keel + bulb analysis')
st.caption('Preliminary installed-system comparison. The hull and bulb end effects are represented by an installed span-efficiency setting; detailed junction interference still requires CFD or testing.')

import math

SEAWATER_DENSITY = 1025.0
SEAWATER_NU = 1.05e-6
G = 9.80665

ac1, ac2, ac3, ac4 = st.columns(4)
with ac1:
    speed_min = st.number_input('Minimum boat speed (knots)', 1.0, 20.0, 2.0, 0.5, key='combined_speed_min')
with ac2:
    speed_max = st.number_input('Maximum boat speed (knots)', 2.0, 30.0, 10.0, 0.5, key='combined_speed_max')
with ac3:
    speed_points = st.slider('Number of speed points', 3, 15, 9, 1, key='combined_speed_points')
with ac4:
    combined_alpha = st.slider('Keel angle of attack / leeway (deg)', 0.0, 10.0, 3.0, 0.25, key='combined_alpha')
if speed_max <= speed_min:
    st.warning('Maximum speed must exceed minimum speed.')
    st.stop()
combined_speeds = np.linspace(speed_min, speed_max, speed_points)

with st.expander('Combined analysis advanced settings'):
    installed_e = st.number_input(
        'Installed keel span efficiency factor (e)', 0.50, 1.00, 0.97, 0.01, key='combined_efficiency',
        help='Preliminary correction for the more favourable installed keel condition with hull/root and bulb end effects. This is not a CFD-derived junction correction.'
    )
    bulb_form_factor = st.number_input('Bulb drag form factor', 1.00, 2.00, 1.15, 0.01, key='combined_bulb_ff')

# Keel hydrodynamics using the same preliminary method as the standalone keel app.
ct = ktc / 1000.0
cb = kbc / 1000.0
span_m = float(keel['span']) / 1000.0
sref = 0.5 * (ct + cb) * span_m
cmean = 0.5 * (ct + cb)
ar = span_m**2 / max(sref, 1e-12)
# Approximate closed-section perimeter correction for the two selected thickness ratios.
def _foil_perimeter_ratio(t_pct):
    x = np.linspace(0.0, 1.0, 1001)
    t = t_pct / 100.0
    # Standard smooth NACA thickness envelope; sufficient for wetted-area correction here.
    y = 5*t*(0.2969*np.sqrt(np.maximum(x,0))-0.1260*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
    y = np.maximum(y,0.0)
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return 2.0*float(np.sum(ds))

eta = np.linspace(0.0, 1.0, 101)
chords = ct + (cb-ct)*eta
prs = _foil_perimeter_ratio(float(keel['top_t'])) + (_foil_perimeter_ratio(float(keel['bot_t']))-_foil_perimeter_ratio(float(keel['top_t'])))*eta
keel_swet = float(np.trapezoid(chords*prs, eta)*span_m)
a0 = 2.0*math.pi
lift_slope = a0/(1.0+a0/(math.pi*installed_e*max(ar,1e-9)))
keel_cl = lift_slope*math.radians(combined_alpha)
tmean = 0.5*(float(keel['top_t'])+float(keel['bot_t']))/100.0
keel_ff = 1.0+2.0*tmean+60.0*tmean**4

# Bulb surface-area estimate from the saved longitudinal profile and a transverse ellipse/circle.
# This intentionally remains a preliminary comparison model until the full bulb mesh is shared directly.
Lmm = float(bulb['length_mm'])
xx = np.asarray(bulb['profile_x_mm'],dtype=float)/1000.0
top_r = np.asarray(bulb['profile_top_mm'],dtype=float)/1000.0
bot_r = np.asarray(bulb['profile_bottom_mm'],dtype=float)/1000.0
shape_mode = bulb.get('shape_mode','Symmetric')
if shape_mode == 'Asymmetric' and bulb.get('width_mm'):
    max_half_width = float(bulb['width_mm'])/2000.0
    envelope = np.maximum(top_r,bot_r)
    half_width = max_half_width*envelope/max(float(np.max(envelope)),1e-12)
else:
    half_width = np.maximum(top_r,bot_r)

# Approximate lateral surface by integrating elliptical ring circumference along x.
a = np.maximum(half_width,1e-9)
b = np.maximum(0.5*(top_r+bot_r),1e-9)
h = ((a-b)**2)/np.maximum((a+b)**2,1e-18)
perim = math.pi*(a+b)*(1.0 + 3.0*h/(10.0+np.sqrt(np.maximum(4.0-3.0*h,1e-12))))
bulb_swet = float(np.trapezoid(perim,xx))
L_m = Lmm/1000.0
# Evaluate the existing hydrodynamic method at each requested boat speed.
def combined_result_at_speed(speed_kn):
    v = float(speed_kn) * 0.514444
    q = 0.5 * SEAWATER_DENSITY * v**2
    keel_lift_n = q * sref * keel_cl
    re_keel = v * cmean / max(SEAWATER_NU, 1e-12)
    cf_keel = 0.075 / (max(math.log10(max(re_keel, 1.0001))-2.0, 1e-6)**2) if re_keel > 1e4 else 0.0
    keel_profile_n = q * keel_swet * cf_keel * keel_ff
    keel_induced_n = q * sref * (keel_cl**2 / (math.pi * installed_e * max(ar, 1e-12)))
    re_bulb = v * L_m / max(SEAWATER_NU, 1e-12)
    cf_bulb = 0.075 / (max(math.log10(max(re_bulb, 1.0001))-2.0, 1e-6)**2) if re_bulb > 1e4 else 0.0
    bulb_drag_n = q * bulb_swet * cf_bulb * bulb_form_factor
    if (shape_mode == 'Asymmetric' and bulb.get('top_pct') is not None
            and bulb.get('bottom_pct') is not None
            and abs(float(bulb['top_pct'])-float(bulb['bottom_pct'])) > 1e-12):
        max_width_m = float(bulb.get('width_mm') or 100.0)/1000.0
        bulb_lift_kgf = 10.0*(max_width_m/0.100)*(speed_kn/14.0)**2
    else:
        bulb_lift_kgf = 0.0
    keel_lift_kgf = keel_lift_n/G
    keel_profile_kgf = keel_profile_n/G
    keel_induced_kgf = keel_induced_n/G
    bulb_drag_kgf = bulb_drag_n/G
    total_lift = keel_lift_kgf + bulb_lift_kgf
    total_drag = keel_profile_kgf + keel_induced_kgf + bulb_drag_kgf
    return {
        'Speed (kn)': float(speed_kn),
        'Keel lift (kgf)': keel_lift_kgf,
        'Bulb lift (kgf)': bulb_lift_kgf,
        'Combined lift (kgf)': total_lift,
        'Keel profile drag (kgf)': keel_profile_kgf,
        'Keel induced drag (kgf)': keel_induced_kgf,
        'Bulb drag (kgf)': bulb_drag_kgf,
        'Combined drag (kgf)': total_drag,
        'L/D': abs(total_lift)/total_drag if total_drag > 1e-12 else 0.0,
    }

combined_df = pd.DataFrame([combined_result_at_speed(speed) for speed in combined_speeds])
st.subheader('Combined performance by boat speed')
st.dataframe(combined_df.style.format({col: '{:.2f}' for col in combined_df.columns}),
             hide_index=True, use_container_width=True)

combined_graph = go.Figure()
for column in ('Keel lift (kgf)', 'Bulb lift (kgf)', 'Combined lift (kgf)',
               'Combined drag (kgf)'):
    combined_graph.add_trace(go.Scatter(
        x=combined_df['Speed (kn)'], y=combined_df[column],
        mode='lines+markers', name=column))
combined_graph.update_layout(
    title=f'Keel + bulb performance at {combined_alpha:.2f}° leeway',
    xaxis_title='Boat speed (knots)', yaxis_title='Lift / drag (kgf)',
    height=460)
st.plotly_chart(combined_graph, use_container_width=True)
ld_graph = go.Figure()
ld_graph.add_trace(go.Scatter(x=combined_df['Speed (kn)'],
                              y=combined_df['L/D'], mode='lines+markers', name='L/D'))
ld_graph.update_layout(title='Combined lift-to-drag ratio',
                       xaxis_title='Boat speed (knots)', yaxis_title='L/D', height=300)
st.plotly_chart(ld_graph, use_container_width=True)
st.caption(f'Keel wetted area {keel_swet:.3f} m²; bulb wetted area {bulb_swet:.3f} m². '
           'Drag comprises keel profile, keel induced and bulb drag. '
           'All results are preliminary comparisons, not CFD.')

with st.expander('Combined calculation assumptions'):
    st.write('Seawater density: 1025 kg/m³. Kinematic viscosity: 1.05 × 10⁻⁶ m²/s.')
    st.write('Keel lift uses the same finite-wing method as the Keel Designer, with a separate installed span-efficiency factor to represent the beneficial hull/root and bulb end effects.')
    st.write('The installed span-efficiency value is a preliminary design assumption, not a CFD-derived end-plate correction.')
    st.write('Bulb drag uses ITTC-1957 skin friction, estimated bulb wetted area and the selected bulb form factor.')
    st.write('A symmetric bulb produces zero lift at 0° in this preliminary model. The existing modest calibrated bulb-lift model is retained for asymmetric bulbs.')
    st.write('Junction interference drag, heel, free-surface effects, ventilation and detailed 3D pressure interaction are not yet included.')

# ---------------- Weight, centres of volume & righting moment ----------------
st.divider()
st.subheader('Keel + bulb righting moment')
st.caption('Preliminary ballast righting moment from the current 3D keel and bulb geometry. Calculated centres can be manually overridden.')


# Local keel section thickness function for RM centre-of-volume calculations.
# Kept self-contained because Streamlit pages execute as separate modules.
def _rm_section_curve(x, series):
    x=np.asarray(x,dtype=float)
    refs={
      '63':([0,.005,.0075,.0125,.025,.05,.075,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95,1], [0,.009838,.011938,.015181,.020979,.029256,.035432,.040386,.048003,.053431,.057115,.059304,.060005,.05922,.057051,.053698,.049354,.044206,.038398,.032112,.025567,.019002,.012724,.007076,.002521,0]),
      '64':([0,.005,.0075,.0125,.025,.05,.075,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95,1], [0,.009755,.011802,.014904,.020331,.028075,.033914,.038668,.046159,.051704,.055721,.058381,.059789,.059752,.057972,.054788,.050545,.045462,.039721,.033488,.026938,.020289,.013815,.007862,.002883,0]),
      '65':([0,.005,.0075,.0125,.025,.05,.075,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95,1], [0,.009146,.011031,.013824,.018792,.026091,.031756,.03649,.044035,.049758,.05407,.057172,.059135,.05999,.059512,.057573,.054137,.049445,.043806,.037427,.030576,.023433,.016282,.009461,.003534,0]),
      '66':([0,.005,.0075,.0125,.025,.05,.075,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90,.95,1], [0,.009039,.010858,.013525,.018067,.024936,.030378,.034933,.04234,.047996,.052364,.055656,.057987,.059424,.059981,.059642,.058339,.055857,.051477,.045122,.037677,.029433,.020807,.012346,.004738,0])}
    if series in refs:
        xp,yp=refs[series]; y=np.interp(x,np.asarray(xp),np.asarray(yp)); return y/max(float(np.max(y)),1e-12)
    # 00-series standard closed NACA thickness envelope, normalized to unit peak.
    y=0.2969*np.sqrt(np.clip(x,0,1))-0.1260*x-0.3516*x**2+0.2843*x**3-0.1036*x**4
    y=np.maximum(y,0.0); return y/max(float(np.max(y)),1e-12)

# Numerical centre of volume for the keel. The foil section area at each span
# station is integrated from the actual selected thickness envelope used by the app.
def _keel_volume_centroid(keel, n_span=301, n_chord=1001):
    eta = np.linspace(0.0, 1.0, n_span)
    xi = np.linspace(0.0, 1.0, n_chord)
    span_mm = float(keel['span'])
    top_depth_mm = float(keel['top_depth'])
    vols = []
    xs = []
    zs = []
    for e in eta:
        chord = float(keel['top_chord']) + e*(float(keel['bot_chord'])-float(keel['top_chord']))
        setout = float(keel['top_setout']) + e*(float(keel['bot_setout'])-float(keel['top_setout']))
        thick_pct = float(keel['top_t']) + e*(float(keel['bot_t'])-float(keel['top_t']))
        ut = _rm_section_curve(xi, keel['top_series'])
        ub = _rm_section_curve(xi, keel['bot_series'])
        ut = ut/max(float(np.max(ut)),1e-12); ub = ub/max(float(np.max(ub)),1e-12)
        unit=(1.0-e)*ut+e*ub; unit=unit/max(float(np.max(unit)),1e-12)
        half_t=chord*(thick_pct/100.0)*0.5*unit
        area_mm2=float(np.trapezoid(2.0*half_t, xi*chord))
        # Section volume centroid fore/aft from bow datum.
        if area_mm2 > 0:
            xsec=float(np.trapezoid((setout+xi*chord)*(2.0*half_t), xi*chord)/area_mm2)
        else:
            xsec=setout+0.5*chord
        vols.append(area_mm2)
        xs.append(xsec)
        zs.append(top_depth_mm+e*span_mm)
    vols=np.asarray(vols); xs=np.asarray(xs); zs=np.asarray(zs)
    # Integrate area along span. Ratios below give centroid coordinates.
    volume_mm3=float(np.trapezoid(vols, eta)*span_mm)
    xcg=float(np.trapezoid(vols*xs, eta)/max(np.trapezoid(vols,eta),1e-12))
    zcg=float(np.trapezoid(vols*zs, eta)/max(np.trapezoid(vols,eta),1e-12))
    return volume_mm3/1e9, xcg, zcg

# Bulb volume and centroid from the same longitudinal profile used on this page.
# Each transverse station is treated as two half ellipses sharing the current beam.
def _bulb_volume_centroid():
    x_mm=np.asarray(bulb['profile_x_mm'],dtype=float)
    rt=np.asarray(bulb['profile_top_mm'],dtype=float)
    rb=np.asarray(bulb['profile_bottom_mm'],dtype=float)
    hw=np.asarray(half_width,dtype=float)*1000.0
    # Full section area = pi/2 * half-width * (top radius + bottom radius)
    A=0.5*math.pi*hw*(rt+rb)
    vol_mm3=float(np.trapezoid(A,x_mm))
    if vol_mm3 <= 1e-9:
        return 0.0, float(bulb_le_setout)+Lmm/2.0, centre_depth
    x_local=float(np.trapezoid(A*x_mm,x_mm)/vol_mm3)
    # Centroid of joined upper/lower half ellipses relative to bulb centreline.
    # Positive depth is downward. Lower half centroid = +4 rb/(3pi), upper = -4 rt/(3pi).
    zoff=np.where((rt+rb)>1e-12, (4.0/(3.0*math.pi))*(rb*rb-rt*rt)/(rt+rb), 0.0)
    z_abs=centre_depth+zoff
    zcg=float(np.trapezoid(A*z_abs,x_mm)/vol_mm3)
    return vol_mm3/1e9, float(bulb_le_setout)+x_local, zcg

keel_volume_m3, keel_lcg_calc_mm, keel_vcg_calc_mm = _keel_volume_centroid(keel)
bulb_volume_m3, bulb_lcg_calc_mm, bulb_vcg_calc_mm = _bulb_volume_centroid()

rm1,rm2,rm3=st.columns(3)
with rm1:
    keel_base_kg_m2=st.number_input('Keel weight factor (kg/m² planform)', min_value=0.0, max_value=5000.0, value=250.0, step=10.0,
        help='Known keel weight factor per square metre of side planform area, as shown in the design method.')
    keel_weight_calc=sref*keel_base_kg_m2
    keel_weight_override=st.number_input('Keel weight override (kg, 0 = calculated)', min_value=0.0, max_value=100000.0, value=0.0, step=10.0)
    keel_weight=keel_weight_override if keel_weight_override>0 else keel_weight_calc
with rm2:
    keel_vcg_override=st.number_input('Keel VCG override below LWL (mm, 0 = calculated)', min_value=0.0, max_value=10000.0, value=0.0, step=10.0)
    keel_lcg_override=st.number_input('Keel LCG override from bow setout (mm, 0 = calculated)', min_value=0.0, max_value=20000.0, value=0.0, step=10.0)
    keel_vcg_mm=keel_vcg_override if keel_vcg_override>0 else keel_vcg_calc_mm
    keel_lcg_mm=keel_lcg_override if keel_lcg_override>0 else keel_lcg_calc_mm
with rm3:
    lead_density_rm=st.number_input('Bulb lead density (kg/m³)', min_value=9000.0, max_value=12000.0, value=float(bulb.get('lead_density_kg_m3',11340.0)), step=10.0, key='combined_rm_lead_density')
    bulb_volume_m3=float(bulb.get('volume_m3',bulb_volume_m3))
    bulb_lcg_calc_mm=float(bulb_le_setout)+float(bulb.get('centroid_x_mm',bulb_lcg_calc_mm-float(bulb_le_setout)))
    bulb_vcg_calc_mm=centre_depth+float(bulb.get('centroid_z_mm',bulb_vcg_calc_mm-centre_depth))
    bulb_weight_calc=bulb_volume_m3*lead_density_rm
    bulb_weight_override=st.number_input('Bulb weight override (kg, 0 = calculated)', min_value=0.0, max_value=100000.0, value=0.0, step=10.0)
    bulb_weight=bulb_weight_override if bulb_weight_override>0 else bulb_weight_calc

# Bulb CG overrides are included for consistency with keel and for imported/known bulbs.
bc1,bc2=st.columns(2)
with bc1:
    bulb_vcg_override=st.number_input('Bulb VCG override below LWL (mm, 0 = calculated)', min_value=0.0, max_value=10000.0, value=0.0, step=10.0)
with bc2:
    bulb_lcg_override=st.number_input('Bulb LCG override from bow setout (mm, 0 = calculated)', min_value=0.0, max_value=20000.0, value=0.0, step=10.0)
bulb_vcg_mm=bulb_vcg_override if bulb_vcg_override>0 else bulb_vcg_calc_mm
bulb_lcg_mm=bulb_lcg_override if bulb_lcg_override>0 else bulb_lcg_calc_mm

c1,c2,c3,c4=st.columns(4)
c1.metric('Keel weight used',f'{keel_weight:.1f} kg')
c2.metric('Keel VCG / LCG',f'{keel_vcg_mm:.0f} / {keel_lcg_mm:.0f} mm')
c3.metric('Bulb weight used',f'{bulb_weight:.1f} kg')
c4.metric('Bulb VCG / LCG',f'{bulb_vcg_mm:.0f} / {bulb_lcg_mm:.0f} mm')
st.caption(f'Calculated keel volume {keel_volume_m3:.4f} m³; keel planform area {sref:.3f} m²; calculated bulb volume {bulb_volume_m3:.4f} m³. VCG is depth below LWL; LCG is setout aft from the bow datum.')

heel_deg=np.arange(0.0,30.1,5.0)
heel_rad=np.radians(heel_deg)
keel_rm_kgm=keel_weight*(keel_vcg_mm/1000.0)*np.sin(heel_rad)
bulb_rm_kgm=bulb_weight*(bulb_vcg_mm/1000.0)*np.sin(heel_rad)
total_rm_kgm=keel_rm_kgm+bulb_rm_kgm
rm_df=pd.DataFrame({
    'Heel (deg)':heel_deg.astype(int),
    'Keel RM (kg·m)':keel_rm_kgm,
    'Bulb RM (kg·m)':bulb_rm_kgm,
    'Total RM (kg·m)':total_rm_kgm,
})
st.dataframe(rm_df.style.format({'Keel RM (kg·m)':'{:.1f}','Bulb RM (kg·m)':'{:.1f}','Total RM (kg·m)':'{:.1f}'}),hide_index=True,use_container_width=True)
rm_fig=go.Figure()
rm_fig.add_trace(go.Scatter(x=heel_deg,y=total_rm_kgm,mode='lines+markers',name='Total RM'))
rm_fig.add_trace(go.Scatter(x=heel_deg,y=bulb_rm_kgm,mode='lines+markers',name='Bulb RM'))
rm_fig.add_trace(go.Scatter(x=heel_deg,y=keel_rm_kgm,mode='lines+markers',name='Keel RM'))
rm_fig.update_layout(title='Keel + bulb righting moment',xaxis_title='Heel angle (deg)',yaxis_title='Righting moment (kg·m)',height=430)
st.plotly_chart(rm_fig,use_container_width=True)
st.caption('This is ballast-only righting moment about the LWL centre plane: weight × vertical CG depth × sin(heel). It does not include hull-form stability, crew, rig or other yacht weights.')

# ---------------- Combined interactive 3D view ----------------
st.divider()
st.subheader('Combined 3D keel + bulb')
st.caption('Dynamic first combined model. Rotate, zoom and pan with the mouse. Changes made in Keel Design or Bulb Design are reflected here.')

fig3 = go.Figure()

# Keel surfaces in boat coordinates: X = setout aft from bow, Y = transverse, Z = depth below LWL.
n_span=31; n_chord=81
etas=np.linspace(0.0,1.0,n_span); xis=np.linspace(0.0,1.0,n_chord)
KX=np.zeros((n_span,n_chord)); KY=np.zeros_like(KX); KZ=np.zeros_like(KX)
for i,e in enumerate(etas):
    chord=ktc+e*(kbc-ktc)
    setout=ktx+e*(kbx-ktx)
    tp=float(keel['top_t'])+e*(float(keel['bot_t'])-float(keel['top_t']))
    # Smooth NACA envelope for 3D display; side planform remains the exact approved geometry.
    u=5*(tp/100.0)*(0.2969*np.sqrt(np.maximum(xis,0))-0.1260*xis-0.3516*xis**2+0.2843*xis**3-0.1036*xis**4)
    u=np.maximum(u,0.0)
    KX[i,:]=setout+xis*chord
    KY[i,:]=u*chord
    KZ[i,:]=kty+e*float(keel['span'])
fig3.add_trace(go.Surface(x=KX,y=KY,z=KZ,showscale=False,opacity=0.82,name='Keel'))
fig3.add_trace(go.Surface(x=KX,y=-KY,z=KZ,showscale=False,opacity=0.82,name='Keel',showlegend=False))

# Bulb mesh from saved longitudinal profile with circular/elliptical transverse stations.
nt=49
theta=np.linspace(0.0,2.0*math.pi,nt)
BX=np.tile((float(bulb_le_setout)+np.asarray(bulb['profile_x_mm'],dtype=float))[:,None],(1,nt))
BY=np.zeros_like(BX); BZ=np.zeros_like(BX)
for i in range(BX.shape[0]):
    c=np.cos(theta); s=np.sin(theta)
    BY[i,:]=half_width[i]*1000.0*c
    # Positive sin is lower half in depth coordinates; negative sin is upper half.
    BZ[i,:]=centre_depth + np.where(s>=0.0, bbot_off[i]*s, btop_off[i]*s)
fig3.add_trace(go.Surface(x=BX,y=BY,z=BZ,showscale=False,opacity=0.72,name='Bulb'))

fig3.update_layout(
    height=760,margin=dict(l=0,r=0,t=50,b=0),
    scene=dict(
        xaxis_title='Setout aft from bow (mm)',yaxis_title='Transverse (mm)',zaxis_title='Depth below LWL (mm)',
        aspectmode='data',zaxis=dict(autorange='reversed'),xaxis=dict(autorange='reversed'),
        camera=dict(eye=dict(x=-1.45,y=1.25,z=0.70))
    )
)
st.plotly_chart(fig3,use_container_width=True)

st.caption('The interactive 3D model is for on-screen inspection only and is not included in the PDF.')


# ---------------- Combined PDF comparison report ----------------
st.divider()
st.subheader('Combined PDF comparison report')
design_name = st.text_input('Design / comparison name', value='Keel + Bulb Test A', key='combined_design_name')

def build_combined_pdf():
    if not MATPLOTLIB_AVAILABLE:
        return None
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: dimensioned side elevation
        f = plt.figure(figsize=(11.69, 8.27))
        ax = f.add_axes([0.07,0.16,0.86,0.72])
        ax.fill(np.r_[bx,bx[::-1]], np.r_[btop,bbot[::-1]], alpha=0.28, label='Bulb')
        ax.fill(kx,ky,alpha=0.35,label='Keel')
        ax.plot(bx,btop,lw=1.4); ax.plot(bx,bbot,lw=1.4)
        ax.plot(kx,ky,lw=1.8)
        ax.axhline(0,lw=1.1)
        # NTS bow datum placed at right of drawing.
        bow_pdf=min(float(np.min(bx)),ktx,kbx)-520
        ax.axvline(bow_pdf,lw=1.1)
        ax.text(bow_pdf,max_depth*0.48,' Bow setout line (NTS)',rotation=90,va='center',fontsize=8)
        ax.text(bow_pdf+80,-35,'LWL',fontsize=8)
        # Dimension helper
        def phdim(x0,x1,y,label):
            ax.plot([x0,x1],[y,y],lw=.7); ax.plot([x0,x0],[y-25,y+25],lw=.7); ax.plot([x1,x1],[y-25,y+25],lw=.7)
            ax.text((x0+x1)/2,y+35,label,ha='center',va='bottom',fontsize=7)
        def pvdim(x,y0,y1,label):
            ax.plot([x,x],[y0,y1],lw=.7); ax.plot([x-25,x+25],[y0,y0],lw=.7); ax.plot([x-25,x+25],[y1,y1],lw=.7)
            ax.text(x+35,(y0+y1)/2,label,rotation=90,ha='left',va='center',fontsize=7)
        phdim(ktx,ktx+ktc,kty-115,f'Top chord {ktc:.0f} mm')
        phdim(float(bulb_le_setout),kbx,max_depth+120,f'{kbx-float(bulb_le_setout):.0f} mm')
        phdim(kbx,kbx+kbc,max_depth+205,f'Bottom chord {kbc:.0f} mm')
        phdim(kbx+kbc,bulb_te,max_depth+205,f'{bulb_te-(kbx+kbc):.0f} mm')
        pvdim(xmax+80,0,kty,f'{kty:.0f} mm')
        pvdim(xmax+165,kty,kby,f'Keel span {float(keel["span"]):.0f} mm')
        pvdim(xmax+250,0,float(bulb_bottom_draft),f'Overall draft {bulb_bottom_draft:.0f} mm')
        pvdim(xmax+335,float(bulb_bottom_draft)-bulb_total_depth,float(bulb_bottom_draft),f'Bulb depth {bulb_total_depth:.0f} mm')
        ax.text((bow_pdf+float(bulb_le_setout))/2,max_depth+120,f'Bulb LE setout {bulb_le_setout:.0f} mm (NTS)',ha='center',fontsize=7)
        ax.text((bow_pdf+ktx)/2,kty-115,f'Top setout {ktx:.0f} mm (NTS)',ha='center',fontsize=7)
        ax.set_aspect('equal',adjustable='box'); ax.invert_xaxis(); ax.invert_yaxis(); ax.axis('off')
        f.suptitle(f'Tboats Foil Designer — {design_name}',fontsize=16,fontweight='bold',y=.96)
        f.text(.07,.91,'Combined keel + bulb — dimensioned side elevation, forward to RHS',fontsize=10)
        f.text(.07,.06,'Dimensions in mm. Bow setout distances are shown NTS. Prototype comparison report — independently check dimensions before manufacture.',fontsize=7.5)
        pdf.savefig(f); plt.close(f)

        # Page 2: geometry and analysis
        f=plt.figure(figsize=(8.27,11.69)); f.suptitle(f'{design_name} — Geometry & Analysis',fontsize=16,fontweight='bold',y=.965)
        keel_txt=(
            f'KEEL\nTop: NACA {keel["top_series"]}, chord {ktc:.0f} mm, thickness {float(keel["top_t"]):.1f}%\n'
            f'Bottom: NACA {keel["bot_series"]}, chord {kbc:.0f} mm, thickness {float(keel["bot_t"]):.1f}%\n'
            f'Top setout {ktx:.0f} mm; bottom setout {kbx:.0f} mm\nTop below LWL {kty:.0f} mm; span {float(keel["span"]):.0f} mm; bottom below LWL {kby:.0f} mm\n'
            f'Planform area {sref:.3f} m²; wetted area {keel_swet:.3f} m²; AR {ar:.2f}'
        )
        bulb_txt=(
            f'BULB\nShape: {shape_mode}; section series {bulb.get("section_series","-")}\nLength {Lmm:.0f} mm; depth {bulb_total_depth:.0f} mm\n'
            f'LE setout {bulb_le_setout:.0f} mm; TE setout {bulb_te:.0f} mm; bottom draft {bulb_bottom_draft:.0f} mm\n'
            f'Wetted area estimate {bulb_swet:.3f} m²'
        )
        if shape_mode=='Symmetric': bulb_txt += f'\nThickness / diameter {float(bulb.get("thickness_pct") or 0):.1f}%'
        else: bulb_txt += f'\nTop {float(bulb.get("top_pct") or 0):.1f}%; bottom {float(bulb.get("bottom_pct") or 0):.1f}%; width {float(bulb.get("width_mm") or 0):.0f} mm; Beaver tail {float(bulb.get("beaver_tail_pct") or 0):.0f}%'
        result_txt=(
            f'ANALYSIS SETTINGS\\nBoat speed {speed_min:.1f}–{speed_max:.1f} kn; '
            f'{speed_points} points; leeway {combined_alpha:.2f}°\\n'
            f'Installed span efficiency e = {installed_e:.2f}; '
            f'bulb form factor = {bulb_form_factor:.2f}\\n\\n'
            f'PERFORMANCE\\nSee speed sweep table and graphs on page 3.'
        )
        f.text(.09,.88,keel_txt,fontsize=10,va='top',linespacing=1.45)
        f.text(.09,.65,bulb_txt,fontsize=10,va='top',linespacing=1.45)
        f.text(.09,.43,result_txt,fontsize=10,va='top',linespacing=1.45)
        assumptions=('CALCULATION NOTES\n'
            'Seawater density 1025 kg/m³; kinematic viscosity 1.05e-6 m²/s.\n'
            'Keel lift uses a finite-wing model with the selected installed span-efficiency factor.\n'
            'Keel profile drag uses ITTC-1957 skin friction and a thickness-based form factor.\n'
            'Bulb drag uses estimated wetted area, ITTC-1957 skin friction and the selected bulb form factor.\n'
            'Symmetric bulb lift is zero in the present preliminary model; asymmetric lift uses the current calibrated comparison model.\n'
            'Detailed junction interference, heel, free-surface effects, ventilation and CFD pressure interaction are not included.\n'
            'Use this report for preliminary design comparison, not as a final performance prediction.')
        f.text(.09,.17,assumptions,fontsize=8.2,va='top',linespacing=1.4)
        pdf.savefig(f); plt.close(f)

        # Page 3: speed sweep table and graphs (no 3D illustration).
        f = plt.figure(figsize=(11.69, 8.27))
        f.suptitle(f'{design_name} — Combined Performance', fontsize=16, fontweight='bold', y=.96)
        f.text(.07, .905, f'Leeway {combined_alpha:.2f}°  |  Speed {speed_min:.1f}–{speed_max:.1f} kn'
               f'  |  Span efficiency {installed_e:.2f}  |  Bulb form factor {bulb_form_factor:.2f}',
               fontsize=9)
        ax = f.add_axes([.10, .52, .80, .32])
        for col, label in [('Keel lift (kgf)', 'Keel lift'), ('Bulb lift (kgf)', 'Bulb lift'),
                           ('Combined lift (kgf)', 'Combined lift'),
                           ('Combined drag (kgf)', 'Combined drag')]:
            ax.plot(combined_df['Speed (kn)'], combined_df[col], marker='o', markersize=3, label=label)
        ax.set_xlabel('Boat speed (knots)')
        ax.set_ylabel('Lift / drag (kgf)')
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=4, loc='upper left')
        cols = ['Speed (kn)', 'Keel lift (kgf)', 'Bulb lift (kgf)', 'Combined lift (kgf)',
                'Keel profile drag (kgf)', 'Keel induced drag (kgf)',
                'Bulb drag (kgf)', 'Combined drag (kgf)', 'L/D']
        headings = ['Speed', 'Keel lift', 'Bulb lift', 'Total lift', 'Profile drag',
                    'Induced drag', 'Bulb drag', 'Total drag', 'L/D']
        data = [[f'{float(row[c]):.2f}' for c in cols] for _, row in combined_df.iterrows()]
        tab_ax = f.add_axes([.04, .12, .92, .34])
        tab_ax.axis('off')
        table = tab_ax.table(cellText=data, colLabels=headings, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7.1)
        table.scale(1, 1.18)
        f.text(.07, .055, 'Preliminary finite-wing and friction estimates. '
               'Junction interference, heel and free-surface effects excluded.', fontsize=8)
        pdf.savefig(f)
        plt.close(f)

        # Page 4: L/D chart and ballast-only righting moment results.
        f = plt.figure(figsize=(11.69, 8.27))
        f.suptitle(f'{design_name} — Efficiency & Righting Moment',
                   fontsize=16, fontweight='bold', y=.96)
        ax1 = f.add_axes([.10, .55, .80, .30])
        ax1.plot(combined_df['Speed (kn)'], combined_df['L/D'], marker='o')
        ax1.set_xlabel('Boat speed (knots)')
        ax1.set_ylabel('Combined L/D')
        ax1.grid(alpha=.3)
        ax2 = f.add_axes([.10, .17, .80, .27])
        ax2.plot(heel_deg, total_rm_kgm, marker='o', label='Total RM')
        ax2.plot(heel_deg, bulb_rm_kgm, marker='o', label='Bulb RM')
        ax2.plot(heel_deg, keel_rm_kgm, marker='o', label='Keel RM')
        ax2.set_xlabel('Heel angle (degrees)')
        ax2.set_ylabel('Ballast RM (kg·m)')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=.3)
        f.text(.07, .075, f'Keel weight {keel_weight:.1f} kg, VCG {keel_vcg_mm:.0f} mm; '
               f'bulb weight {bulb_weight:.1f} kg, VCG {bulb_vcg_mm:.0f} mm.', fontsize=9)
        f.text(.07, .045, 'Ballast-only righting moment; excludes hull, crew, rig and other weights.', fontsize=8)
        pdf.savefig(f)
        plt.close(f)
    buf.seek(0); return buf.getvalue()

# Complete the header actions using the current live design and calculated report.
if advanced:
    saved_design = {
        'type': 'tboat_combined_design', 'version': 1,
        'keel': keel, 'bulb': bulb,
        'bulb_bottom_draft_mm': float(bulb_bottom_draft),
        'bulb_le_setout_mm': float(bulb_le_setout),
    }
    with download_slot.container():
        st.download_button('Download', data=json.dumps(saved_design, indent=2),
                           file_name='Tboat_Keel_Bulb_Design.json', mime='application/json',
                           use_container_width=True)
    report = build_combined_pdf()
    with pdf_slot.container():
        if report is not None:
            safe_name = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in design_name).strip('_') or 'Keel_Bulb'
            st.download_button('PDF', data=report, file_name=f'Tboat_{safe_name}.pdf',
                               mime='application/pdf', use_container_width=True)
        else:
            st.button('PDF', disabled=True, help='PDF creation requires matplotlib.', use_container_width=True)
else:
    with download_slot.container():
        st.button('Download', disabled=True, help='Available with Advanced access.', use_container_width=True)
    with pdf_slot.container():
        st.button('PDF', disabled=True, help='Available with Advanced access.', use_container_width=True)
