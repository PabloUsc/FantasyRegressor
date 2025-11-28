import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- IMPORT MODEL ---
try:
    from model import FantasyPredictor
except ImportError:
    st.error("🚨 Critical Error: 'model.py' not found. Please ensure it is in the same directory.")
    st.stop()

# 1. Setup Page Config
st.set_page_config(layout="wide", page_title="Fantasy Optimizer")

# --- 2. CSS STYLING ---
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
    
    /* Dropdown text fix */
    div[data-baseweb="select"] > div {
        background-color: #3b4252;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

add_background()

# --- 3. ROBUST DATA LOADING ---
@st.cache_data
def load_and_predict():
    # A. Initialize and Train Model
    predictor = FantasyPredictor()
    try:
        predictor.train_model('complete.csv')
    except FileNotFoundError:
        st.error("🚨 Error: 'complete.csv' missing. Model cannot train.")
        # Proceeding strictly for UI debugging, though predictions will fail
    except Exception as e:
        st.warning(f"Model Training Warning: {e}")

    # B. Load 2025 Stats (The Source of Truth)
    try:
        stats_df = pd.read_csv("2025.csv")
        # Ensure Player column exists and is string
        stats_df['Player'] = stats_df['Player'].astype(str).str.strip()
    except FileNotFoundError:
        st.error("🚨 Error: '2025.csv' not found. This file is required.")
        return pd.DataFrame()

    # C. Load Roster (For Images only)
    try:
        # Try local CSV first
        roster_df = pd.read_csv("roster.csv")
    except FileNotFoundError:
        # If local fails, try downloading (fallback)
        try:
            import nflreadpy as nfl
            roster_df = nfl.load_rosters().to_pandas()
            teams = nfl.load_teams().to_pandas()[['team_abbr', 'team_logo_espn']]
            roster_df = pd.merge(roster_df, teams, left_on='team', right_on='team_abbr', how='left')
            # Standardize column names from nflreadpy
            name_col = 'full_name' if 'full_name' in roster_df.columns else 'player_name'
            roster_df = roster_df.rename(columns={name_col: 'Player', 'team_logo_espn': 'Team_Logo'})
        except:
            roster_df = pd.DataFrame() # Empty if everything fails

    # D. ADVANCED MERGING (Fixing the mismatches)
    
    # Helper to clean names for matching (removes 'Jr', '.', 'III')
    def normalize_name(name):
        n = str(name).lower()
        n = n.replace('.', '').replace(',', '').replace("'", "")
        n = n.replace(' jr', '').replace(' sr', '').replace(' iii', '').replace(' ii', '')
        return n.strip()

    stats_df['clean_name'] = stats_df['Player'].apply(normalize_name)
    
    if not roster_df.empty:
        # Rename roster columns to avoid collisions
        # We expect roster to have: Player, team, position, headshot_url, Team_Logo
        # Check specific columns exist
        if 'Player' in roster_df.columns:
            roster_df['clean_name'] = roster_df['Player'].apply(normalize_name)
            
            # Select only visual columns from roster
            visual_cols = ['clean_name', 'headshot_url', 'Team_Logo', 'team']
            # Only keep cols that actually exist
            visual_cols = [c for c in visual_cols if c in roster_df.columns]
            
            roster_subset = roster_df[visual_cols]
            
            # --- CRITICAL FIX: LEFT JOIN ---
            # We keep ALL players from stats_df (2025.csv), and attach images where possible.
            main_df = pd.merge(stats_df, roster_subset, on='clean_name', how='left')
        else:
            main_df = stats_df.copy()
    else:
        main_df = stats_df.copy()

    # E. FALLBACKS FOR VISUALS
    # If no match found, fill with placeholders
    if 'headshot_url' not in main_df.columns: main_df['headshot_url'] = None
    if 'Team_Logo' not in main_df.columns: main_df['Team_Logo'] = None
    if 'team' not in main_df.columns: 
        # Fallback to 'Tm' from 2025.csv
        main_df['team'] = main_df['Tm'] if 'Tm' in main_df.columns else "N/A"

    main_df['headshot_url'] = main_df['headshot_url'].fillna("https://static.www.nfl.com/image/private/f_auto,q_auto/league/nfl-logo")
    main_df['Team_Logo'] = main_df['Team_Logo'].fillna("https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nfl.png")

    # F. PREDICTIONS
    if not main_df.empty:
        def safe_predict(row):
            try:
                # 1. Get Age
                if 'Age' in row and pd.notna(row['Age']):
                    age = int(float(row['Age']))
                else:
                    age = 25
                
                # 2. Get Position (Prefer 'FantPos' from 2025.csv as it's the stats source)
                if 'FantPos' in row and pd.notna(row['FantPos']):
                    pos = str(row['FantPos'])
                elif 'position' in row and pd.notna(row['position']):
                    pos = str(row['position'])
                else:
                    pos = 'UNK'
                
                # 3. Call Model - Corrected Method Name: predict_player
                # returns None if player/pos combo not found in training data
                result = predictor.predict_player(row['Player'], age, pos)
                
                if result is None:
                    return np.nan
                return result

            except Exception as e:
                # print(e) # Debugging
                return np.nan

        main_df['Predicted_FP'] = main_df.apply(safe_predict, axis=1)
        # Drop failures (Rows where model returned None/NaN)
        main_df = main_df.dropna(subset=['Predicted_FP'])
        
        # Cleanup
        if 'clean_name' in main_df.columns: main_df = main_df.drop(columns=['clean_name'])

    return main_df

# --- LOAD DATA ---
main_df = load_and_predict()

# Check emptiness
if main_df.empty:
    st.error("❌ No players loaded. Please check that '2025.csv' has valid data.")
    st.stop()

# Sorting for dropdown
all_players_list = sorted(main_df['Player'].unique().tolist())

# --- 4. VISUALIZATION COMPONENTS ---

def render_player_card(player_name):
    """Renders HTML card"""
    # Safely get player row
    df_p = main_df[main_df['Player'] == player_name]
    if df_p.empty: return # Safety
    p_data = df_p.iloc[0]
    
    # Get Position (Handle columns safely)
    pos = p_data.get('FantPos', p_data.get('position', 'UNK'))
    
    card_html = f"""<div style="width: 100%; background-color: #3b4252; padding: 15px; border-radius: 10px; display: flex; align-items: center; margin-top: 10px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); color: white; font-family: sans-serif;"><img src="{p_data['headshot_url']}" style="width: 55px; height: 55px; border-radius: 50%; object-fit: cover; border: 2px solid #4c566a; margin-right: 15px;"><div style="flex-grow: 1;"><div style="font-weight: 600; font-size: 16px; margin-bottom: 4px;">{p_data['Player']}</div><div style="font-size: 13px; color: #d8dee9; display: flex; align-items: center;"><img src="{p_data['Team_Logo']}" style="width: 16px; height: 16px; margin-right: 6px;">{p_data['team']} &nbsp; <span style="color:#81a1c1;">|</span> &nbsp; {pos}</div></div></div>"""
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

# --- 5. UI LAYOUT ---

st.markdown("<h1 style='font-size: 2.2em; margin-bottom: 0px;'>Fantasy Start/Sit Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #d8dee9;'>Select players to see their projection cards and compare stats.</p>", unsafe_allow_html=True)
st.write("---")

# Player Selection Columns
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
        
        # --- A. HEAD-TO-HEAD BARS ---
        st.divider()
        st.markdown("<h3 style='text-align: center;'>Head-to-Head Stats</h3>", unsafe_allow_html=True)
        st.write("")
        
        # MAP: (Label, 2025.csv Column Name)
        stats_map = [
            ('Projected FP', 'Predicted_FP'),
            ('Total TDs', 'ScorTD'),
            ('Passing Yds', 'PassYds'),
            ('Rushing Yds', 'RushYds'),
            ('Completions', 'PassCmp'),
            ('Receptions', 'Rec'),
            ('Fumbles', 'Fmb')
        ]
        
        comparison_data = []
        for i, p_name in enumerate(selected_players):
            p_row = main_df[main_df['Player'] == p_name].iloc[0]
            comparison_data.append({
                'name': p_name,
                'row_data': p_row,
                'color': colors[i]
            })

        # --- UPDATED: 3 COLUMNS WIDE (Makes bars larger) ---
        stat_cols = st.columns(3) 
        
        for idx, (label, col_key) in enumerate(stats_map):
            # Calculate max (safely)
            vals = [d['row_data'].get(col_key, 0) for d in comparison_data]
            vals = [0 if pd.isna(v) else v for v in vals]
            max_val = max(vals, default=1)
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
            
            # Use modulus 3 to cycle through the 3 columns
            with stat_cols[idx % 3]:
                draw_stat_group(label, draw_list, max_val)

        # --- B. DEEP DIVE CHART ---
        st.divider()
        st.subheader("Statistical Deep Dive")
        
        comp_df = main_df[main_df['Player'].isin(selected_players)].copy()
        
        color_map = {player: colors[i] for i, player in enumerate(selected_players)}
        
        chart = alt.Chart(comp_df).mark_bar().encode(
            x=alt.X('Player', axis=None),
            y=alt.Y('Predicted_FP', title='Projected Points'),
            color=alt.Color('Player', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), legend=alt.Legend(title="Player")),
            tooltip=['Player', 'Predicted_FP', 'PassYds', 'ScorTD']
        ).properties(height=400).configure_axis(grid=False).configure_view(strokeWidth=0)
        
        st.altair_chart(chart, use_container_width=True)
        
        # Table of Details
        st.caption("Detailed Projections")
        possible_cols = ['Predicted_FP', 'PassYds', 'RushYds', 'ScorTD', 'PassCmp', 'Rec', 'Fmb']
        final_cols = [c for c in possible_cols if c in comp_df.columns]
        
        st.dataframe(comp_df.set_index('Player')[final_cols], use_container_width=True)
        
    else:
        st.warning("Please select at least one player above.")