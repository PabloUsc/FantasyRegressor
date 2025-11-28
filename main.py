import streamlit as st
import pandas as pd
import numpy as np
import nflreadpy as nfl
import altair as alt

# --- IMPORT YOUR MODEL ---
try:
    from model import FantasyPredictor
except ImportError:
    st.error("Could not find 'model.py'. Please ensure it is in the same directory.")
    st.stop()

# 1. Setup Page Config
st.set_page_config(layout="wide", page_title="Fantasy Optimizer")

# --- 2. CSS ---
def add_background():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #2e3440;
        background-image: radial-gradient(#434c5e 2px, transparent 2px), radial-gradient(#434c5e 2px, #2e3440 2px);
        background-size: 30px 30px;
        background-position: 0 0, 15px 15px;
    }
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

add_background()

# --- 3. DATA LOADING & PREDICTION PIPELINE ---
@st.cache_data
def load_and_predict():
    # A. Initialize Model
    predictor = FantasyPredictor()

    # --- STEP 1: TRAIN THE MODEL ---
    try:
        predictor.train_model('complete.csv') 
    except FileNotFoundError:
        st.error("🚨 Error: 'complete.csv' not found. Model cannot train.")
        # We continue just to show the UI, but predictions won't work well
    except Exception as e:
        st.warning(f"Model training issue: {e}")

    # --- STEP 2: LOAD CURRENT STATS (The Source of Truth) ---
    try:
        stats_df = pd.read_csv("2025.csv")
        # Clean player names (strip whitespace)
        stats_df['Player'] = stats_df['Player'].astype(str).str.strip()
    except FileNotFoundError:
        st.error("🚨 Error: Could not find '2025.csv'.")
        return pd.DataFrame()

    # --- STEP 3: LOAD NFL ROSTER (For Images) ---
    roster_loaded = False
    try:
        # Try loading 2024 and 2025 to be safe
        roster = nfl.load_rosters(years=[2024, 2025]).to_pandas()
        
        # Standardize Columns
        if 'full_name' in roster.columns: name_col = 'full_name'
        elif 'player_name' in roster.columns: name_col = 'player_name'
        else: name_col = None

        if name_col:
            # Prepare Roster for Merge
            teams = nfl.load_teams().to_pandas()[['team_abbr', 'team_logo_espn']]
            roster = pd.merge(roster, teams, left_on='team', right_on='team_abbr', how='left')
            cols_keep = [name_col, 'team', 'position', 'headshot_url', 'team_logo_espn']
            roster = roster[cols_keep].rename(columns={name_col: 'Player', 'team_logo_espn': 'Team_Logo'})
            roster['Player'] = roster['Player'].astype(str).str.strip()
            roster_loaded = True
    except Exception as e:
        print(f"Roster load failed: {e}")
        roster_loaded = False

    # --- STEP 4: MERGE OR FALLBACK ---
    if roster_loaded:
        # Try merging
        main_df = pd.merge(roster, stats_df, on='Player', how='inner')
        
        # DEBUG: If merge lost all players, fallback to stats_df
        if main_df.empty:
            st.toast("⚠️ Roster merge returned 0 players. Switching to CSV data only.", icon="⚠️")
            main_df = stats_df.copy()
            # Add dummy visual columns since merge failed
            main_df['headshot_url'] = None
            main_df['Team_Logo'] = None
            main_df['team'] = main_df['Tm'] if 'Tm' in main_df.columns else "N/A"
            main_df['position'] = main_df['FantPos'] if 'FantPos' in main_df.columns else "UNK"
    else:
        # Fallback if internet/roster failed completely
        main_df = stats_df.copy()
        main_df['headshot_url'] = None
        main_df['Team_Logo'] = None
        main_df['team'] = main_df['Tm'] if 'Tm' in main_df.columns else "N/A"
        main_df['position'] = main_df['FantPos'] if 'FantPos' in main_df.columns else "UNK"

    # Fill Fallback Images
    main_df['headshot_url'] = main_df['headshot_url'].fillna("https://static.www.nfl.com/image/private/f_auto,q_auto/league/nfl-logo")
    main_df['Team_Logo'] = main_df['Team_Logo'].fillna("https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nfl.png")

    # --- STEP 5: ROBUST PREDICTION ---
    if not main_df.empty:
        def safe_predict(row):
            try:
                # Get Age
                if 'Age' in row and pd.notna(row['Age']):
                    age = int(float(row['Age']))
                else:
                    age = 25 
                
                # Get Position
                if 'position' in row and pd.notna(row['position']):
                    pos = str(row['position'])
                elif 'FantPos' in row and pd.notna(row['FantPos']):
                    pos = str(row['FantPos'])
                else:
                    pos = 'UNK'
                
                return predictor.predict(row['Player'], age, pos)
            except:
                return np.nan

        main_df['Predicted_FP'] = main_df.apply(safe_predict, axis=1)
        main_df = main_df.dropna(subset=['Predicted_FP'])

    return main_df

# Load Data
main_df = load_and_predict()

# --- DEBUG INFO (Remove after fixing) ---
if main_df.empty:
    st.error("❌ No players loaded! Please check '2025.csv' contains data.")
    st.stop()

all_players_list = sorted(main_df['Player'].unique().tolist())

# --- 5. VISUALIZATION FUNCTIONS ---

def render_player_card(player_name):
    """Renders HTML card"""
    p_data = main_df[main_df['Player'] == player_name].iloc[0]
    card_html = f"""<div style="width: 100%; background-color: #3b4252; padding: 15px; border-radius: 10px; display: flex; align-items: center; margin-top: 10px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); color: white; font-family: sans-serif;"><img src="{p_data['headshot_url']}" style="width: 55px; height: 55px; border-radius: 50%; object-fit: cover; border: 2px solid #4c566a; margin-right: 15px;"><div style="flex-grow: 1;"><div style="font-weight: 600; font-size: 16px; margin-bottom: 4px;">{p_data['Player']}</div><div style="font-size: 13px; color: #d8dee9; display: flex; align-items: center;"><img src="{p_data['Team_Logo']}" style="width: 16px; height: 16px; margin-right: 6px;">{p_data['team']} &nbsp; <span style="color:#81a1c1;">|</span> &nbsp; {p_data['position']}</div></div></div>"""
    st.markdown(card_html, unsafe_allow_html=True)

def draw_stat_group(stat_label, player_data_list, max_val):
    """Draws grouped bars"""
    max_val = max_val if max_val > 0 else 1
    html_content = f'<div style="margin-bottom: 15px; background-color: #3b4252; padding: 10px; border-radius: 8px; border: 1px solid #4c566a;"><div style="text-align: center; font-size: 13px; font-weight: bold; text-transform: uppercase; color: #d8dee9; margin-bottom: 8px; letter-spacing: 1px;">{stat_label}</div>'
    for p in player_data_list:
        val = p['value']
        score_pct = (float(val) / (max_val * 1.1)) * 100
        html_content += f'<div style="display: flex; align-items: center; margin-bottom: 6px; font-family: sans-serif;"><div style="width: 80px; font-size: 11px; color: #e5e9f0; text-align: right; margin-right: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{p["name"]}</div><div style="flex-grow: 1; background-color: #2e3440; height: 8px; border-radius: 4px; margin-right: 10px;"><div style="background-color: {p["color"]}; width: {score_pct}%; height: 100%; border-radius: 4px;"></div></div><div style="width: 35px; font-size: 12px; font-weight: bold; color: {p["color"]}; text-align: left;">{val}</div></div>'
    html_content += '</div>'
    st.markdown(html_content, unsafe_allow_html=True)

# --- 6. UI LAYOUT ---

st.markdown("<h1 style='font-size: 2.2em; margin-bottom: 0px;'>Fantasy Start/Sit Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #d8dee9;'>Select players to see their projection cards and compare stats.</p>", unsafe_allow_html=True)
st.write("---")

# Player Selection
st.markdown("<h3>Player Selection</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
selected_players = []

with col1:
    p1 = st.selectbox("Player 1", options=all_players_list, index=None, placeholder="Select...", label_visibility="collapsed")
    if p1:
        render_player_card(p1)
        selected_players.append(p1)

with col2:
    p2 = st.selectbox("Player 2", options=all_players_list, index=None, placeholder="Select...", label_visibility="collapsed")
    if p2:
        render_player_card(p2)
        selected_players.append(p2)

with col3:
    p3 = st.selectbox("Player 3", options=all_players_list, index=None, placeholder="Select...", label_visibility="collapsed")
    if p3:
        render_player_card(p3)
        selected_players.append(p3)

st.write("")

# Colors
colors = ["#88c0d0", "#bf616a", "#a3be8c"] 

if st.button(f"Compare {len(selected_players)} Players", use_container_width=True, type="primary"):
    if len(selected_players) > 0:
        
        # --- A. VERSUS BARS ---
        st.divider()
        st.markdown("<h3 style='text-align: center;'>Head-to-Head Stats</h3>", unsafe_allow_html=True)
        st.write("")
        
        stats_map = [
            ('Projected FP', 'Predicted_FP'), # From Model
            ('Total TDs', 'ScorTD'),          # From CSV
            ('Passing Yds', 'PassYds'),       # From CSV
            ('Rushing Yds', 'RushYds'),       # From CSV
            ('Completions', 'PassCmp'),       # From CSV
            ('Receptions', 'Rec'),            # From CSV
            ('Fumbles', 'Fmb')                # From CSV
        ]
        
        comparison_data = []
        for i, p_name in enumerate(selected_players):
            p_row = main_df[main_df['Player'] == p_name].iloc[0]
            comparison_data.append({
                'name': p_name,
                'row_data': p_row,
                'color': colors[i]
            })

        stat_cols = st.columns(4) 
        
        for idx, (label, col_key) in enumerate(stats_map):
            max_val = max((d['row_data'].get(col_key, 0) for d in comparison_data), default=1)
            if max_val == 0: max_val = 1
            
            draw_list = []
            for d in comparison_data:
                val = d['row_data'].get(col_key, 0)
                if pd.isna(val): val = 0
                
                formatted_val = int(val) if col_key != 'Predicted_FP' else round(val, 1)
                
                draw_list.append({
                    'name': d['name'],
                    'value': formatted_val,
                    'color': d['color']
                })
            
            with stat_cols[idx % 4]:
                draw_stat_group(label, draw_list, max_val)

        
        # --- B. STATISTICAL DEEP DIVE (Chart) ---
        st.divider()
        st.subheader("Statistical Deep Dive")
        
        comp_df = main_df[main_df['Player'].isin(selected_players)].copy()
        
        color_map = {player: colors[i] for i, player in enumerate(selected_players)}
        
        chart = alt.Chart(comp_df).mark_bar().encode(
            x=alt.X('Player', axis=None),
            y=alt.Y('Predicted_FP', title='Projected Points'),
            color=alt.Color('Player', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), legend=alt.Legend(title="Player")),
            tooltip=['Player', 'Predicted_FP', 'PassYds', 'ScorTD']
        ).properties(
            height=400
        ).configure_axis(
            grid=False
        ).configure_view(
            strokeWidth=0
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        st.caption("Detailed Projections")
        # Ensure cols exist before showing
        possible_cols = ['Predicted_FP', 'PassYds', 'RushYds', 'ScorTD', 'PassCmp', 'Rec', 'Fmb']
        final_cols = [c for c in possible_cols if c in comp_df.columns]
        
        st.dataframe(
            comp_df.set_index('Player')[final_cols], 
            use_container_width=True
        )
        
    else:
        st.warning("Please select at least one player above.")