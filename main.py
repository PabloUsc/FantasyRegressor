import streamlit as st
import pandas as pd
import nflreadpy as nfl
from datetime import datetime
import altair as alt
from model import FantasyPredictor

# 1. Setup Page Config
st.set_page_config(layout="wide", page_title="Fantasy Optimizer")
model = FantasyPredictor()


def add_background():
    st.markdown("""
    <style>
    /* Main Background - Lighter Slate Blue */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        background-image: radial-gradient(#3b4252 2px, transparent 2px), radial-gradient(#3b4252 2px, #1E232C 2px);
        background-size: 30px 30px;
        background-position: 0 0, 15px 15px;
    }
    
    /* Make Header Transparent */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    
    /* Optional: Tweaks to make dropdowns stand out more on lighter background */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #3b4252 !important;
        border-color: #4c566a !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

add_background()

# 2. Load Real NFL Data (Roster + Team Logos)
@st.cache_data
def load_data():
    # A. Load Roster
    try:
        roster = nfl.load_rosters()
        roster = roster.to_pandas()
    except:
        st.error("Failed to load NFL roster data. Check internet connection.")
        return pd.DataFrame()

    # B. Load Team Data (for Logos)
    try:
        teams = nfl.load_teams()
        teams = teams.to_pandas()[['team_abbr', 'team_logo_espn']]
    except:
        st.error("Failed to load Team data.")
        return pd.DataFrame()
    
    # C. Handle Column Name variations in nflreadpy
    if 'full_name' in roster.columns:
        name_col = 'full_name'
    elif 'player_name' in roster.columns:
        name_col = 'player_name'
    else:
        st.error("Column 'full_name' or 'player_name' not found in roster.")
        return pd.DataFrame()
    
    # D. Merge Roster with Team Logos
    #    Roster has 'team', Teams has 'team_abbr'
    df = pd.merge(roster, teams, left_on='team', right_on='team_abbr', how='left')
    
    # E. Select and Rename Columns
    cols_to_keep = [name_col, 'team', 'position', 'headshot_url', 'team_logo_espn', 'birth_date']
    df = df[cols_to_keep].rename(columns={
        name_col: 'Player', 
        'team_logo_espn': 'Team_Logo'
    })
    
    # F. Fill missing images with fallbacks
    df['headshot_url'] = df['headshot_url'].fillna("https://static.www.nfl.com/image/private/f_auto,q_auto/league/nfl-logo")
    df['Team_Logo'] = df['Team_Logo'].fillna("https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/nfl.png")
    
    return df

def get_player_details(player_name, df_roster):
    # 1. Filter the dataframe for the specific player
    player_row = df_roster[df_roster['Player'] == player_name]
    
    if player_row.empty:
        return None, None

    # 2. Get the first match (in case of duplicates, usually takes active one)
    data = player_row.iloc[0]
    
    # 3. Get Position
    position = data['position']
    
    # 4. Calculate Age from birth_date
    # nflreadpy provides 'birth_date' as a string (YYYY-MM-DD)
    if pd.isna(data.get('birth_date')):
        return None, position # Return position even if age is unknown

    try:
        b_date = pd.to_datetime(data['birth_date'])
        today = datetime.now()
        
        # Calculate age: (Difference in days) / 365.25
        age = int((today - b_date).days / 365.25)
        
        return age, position
    except:
        return None, position

# Load the base data
roster_df = load_data()
model.train_model('data/complete.csv')

# 3. Your Regression Model Output (Mock Data with Underscores!)
model_data = {
    'Player': ['Patrick Mahomes', 'Josh Allen', 'Jalen Hurts', 'Lamar Jackson', 
               'Joe Burrow', 'Justin Fields', 'Saquon Barkley', 'Christian McCaffrey',
               'Tyreek Hill', 'Justin Jefferson', 'Travis Kelce', 'T.J. Hockenson'],
    'Predicted_FP': [24.5, 28.2, 26.1, 24.4, 21.8, 20.2, 18.1, 22.5, 20.2, 23.1, 15.5, 12.3],
    'Pass_Yds': [283, 295, 240, 210, 275, 180, 0, 0, 0, 0, 0, 0],
    'Rush_Yds': [20, 45, 40, 60, 10, 80, 95, 85, 5, 2, 0, 0],
    'TDs': [2.5, 2.8, 2.2, 1.9, 1.8, 1.5, 1.1, 1.4, 0.9, 1.1, 0.7, 0.5]
}
model_df = pd.DataFrame(model_data)

# 4. Merge Model Predictions into Roster Data
#    We use inner join here so we only show players we have predictions for
main_df = pd.merge(model_df, roster_df, on='Player', how='inner')

# Create list for dropdowns
all_players_list = main_df['Player'].unique().tolist()


def render_player_card(player_name):
    """
    Renders a styled HTML card.
    Uses a single-line string to force Streamlit to render HTML instead of a code block.
    """
    # Get specific player data
    p_data = main_df[main_df['Player'] == player_name].iloc[0]
    
    # -------------------------------------------------------------------------
    # The HTML is compacted into one line below to prevent Markdown errors.
    # -------------------------------------------------------------------------
    # card_html = f"""<div style="width: 100%; background-color: #262626; padding: 15px; border-radius: 10px; display: flex; align-items: center; margin-top: 10px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); color: white; font-family: sans-serif;"><img src="{p_data['headshot_url']}" style="width: 55px; height: 55px; border-radius: 50%; object-fit: cover; border: 2px solid #444; margin-right: 15px;"><div style="flex-grow: 1;"><div style="font-weight: 600; font-size: 16px; margin-bottom: 4px;">{p_data['Player']}</div><div style="font-size: 13px; color: #b0b0b0; display: flex; align-items: center;"><img src="{p_data['Team_Logo']}" style="width: 16px; height: 16px; margin-right: 6px;">{p_data['team']} &nbsp; <span style="color:#666;">|</span> &nbsp; {p_data['position']}</div></div><div style="text-align: right;"><div style="font-size: 20px; font-weight: bold; color: #fff;">{p_data['Predicted_FP']:.1f}</div><div style="font-size: 11px; color: #888;">Proj</div></div></div>"""
    card_html = f"""<div style="width: 100%; background-color: #262626; padding: 15px; border-radius: 10px; display: flex; align-items: center; margin-top: 10px; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); color: white; font-family: sans-serif;"><img src="{p_data['headshot_url']}" style="width: 55px; height: 55px; border-radius: 50%; object-fit: cover; border: 2px solid #444; margin-right: 15px;"><div style="flex-grow: 1;"><div style="font-weight: 600; font-size: 16px; margin-bottom: 4px;">{p_data['Player']}</div><div style="font-size: 13px; color: #b0b0b0; display: flex; align-items: center;"><img src="{p_data['Team_Logo']}" style="width: 16px; height: 16px; margin-right: 6px;">{p_data['team']} &nbsp; <span style="color:#666;">|</span> &nbsp; {p_data['position']}</div></div></div>"""
    
    # Render the HTML
    st.markdown(card_html, unsafe_allow_html=True)

def draw_versus_bar(stat_name, val1, val2, max_val, color1, color2):
    """Draws a Left vs Right comparison bar with specific colors per player"""
    c1, c2, c3 = st.columns([4, 2, 4])
    
    # Avoid division by zero
    max_val = max_val if max_val > 0 else 1
    p1_score = float(val1) / (max_val * 1.1) 
    p2_score = float(val2) / (max_val * 1.1)

    with c1:
        # Player 1 (Left Side) - Uses color1
        st.markdown(f"<div style='text-align: right; font-weight:bold; color:white;'>{val1}</div>", unsafe_allow_html=True)
        # Custom HTML Progress Bar for Player 1 (Fills Right to Left visually via flex or just standard alignment)
        # To make it look like a versus bar, we use a custom div
        st.markdown(f"""
        <div style="background-color: #333; width: 100%; height: 8px; border-radius: 4px; display: flex; justify-content: flex-end;">
            <div style="background-color: {color1}; width: {p1_score*100}%; height: 100%; border-radius: 4px;"></div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div style='text-align: center; font-size:12px; font-weight:bold; text-transform:uppercase; margin-top:5px; color: #b0b0b0;'>{stat_name}</div>", unsafe_allow_html=True)

    with c3:
        # Player 2 (Right Side) - Uses color2
        st.markdown(f"<div style='text-align: left; font-weight:bold; color:white;'>{val2}</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color: #333; width: 100%; height: 8px; border-radius: 4px;">
            <div style="background-color: {color2}; width: {p2_score*100}%; height: 100%; border-radius: 4px;"></div>
        </div>
        """, unsafe_allow_html=True)

def draw_stat_group(stat_label, player_data_list, max_val):
    """
    Draws a group of bars for a single statistic.
    HTML is minified (one long string) to prevent Streamlit from treating it as code.
    """
    max_val = max_val if max_val > 0 else 1
    
    # Start Container
    html_content = f'<div style="margin-bottom: 15px; background-color: #1f1f1f; padding: 10px; border-radius: 8px; border: 1px solid #333;"><div style="text-align: center; font-size: 13px; font-weight: bold; text-transform: uppercase; color: #b0b0b0; margin-bottom: 8px; letter-spacing: 1px;">{stat_label}</div>'
    
    # Loop and add Bars
    for p in player_data_list:
        val = p['value']
        score_pct = (float(val) / (max_val * 1.1)) * 100
        
        # Add Player Row (Single line string)
        html_content += f'<div style="display: flex; align-items: center; margin-bottom: 6px; font-family: sans-serif;"><div style="width: 80px; font-size: 11px; color: #ccc; text-align: right; margin-right: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{p["name"]}</div><div style="flex-grow: 1; background-color: #333; height: 8px; border-radius: 4px; margin-right: 10px;"><div style="background-color: {p["color"]}; width: {score_pct}%; height: 100%; border-radius: 4px;"></div></div><div style="width: 35px; font-size: 12px; font-weight: bold; color: {p["color"]}; text-align: left;">{val}</div></div>'
    
    # Close Container
    html_content += '</div>'
    
    st.markdown(html_content, unsafe_allow_html=True)

# --- START OF UI ---

st.markdown("<h1 style='font-size: 2.2em; margin-bottom: 0px;'>Fantasy Start/Sit Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #b0b0b0;'>Select players to see their projection cards and compare stats.</p>", unsafe_allow_html=True)

st.write("---")

st.write("") # Spacer

# 2. Multi-Player Compare Headers
st.markdown("<h3>Player Selection</h3>", unsafe_allow_html=True)
st.markdown("<p style='color: #888; font-size: 0.9em; margin-top: -15px;'>Select from dropdown to view player card</p>", unsafe_allow_html=True)

# 3. The 3 Columns Layout
col1, col2, col3 = st.columns(3)

selected_players = []

# --- COLUMN 1 ---
with col1:
    p1 = st.selectbox(
        "Player 1", 
        options=all_players_list, 
        index=None, 
        placeholder="Select player 1...",
        label_visibility="collapsed"
    )
    if p1:
        render_player_card(p1)
        selected_players.append(p1)

# --- COLUMN 2 ---
with col2:
    p2 = st.selectbox(
        "Player 2", 
        options=all_players_list, 
        index=None, 
        placeholder="Select player 2...",
        label_visibility="collapsed"
    )
    if p2:
        render_player_card(p2)
        selected_players.append(p2)

# --- COLUMN 3 ---
with col3:
    p3 = st.selectbox(
        "Player 3", 
        options=all_players_list, 
        index=None, 
        placeholder="Select player 3...",
        label_visibility="collapsed"
    )
    if p3:
        render_player_card(p3)
        selected_players.append(p3)

st.write("")
st.write("")

colors = ["#3b82f6", "#ef4444", "#10b981"]

# Compare Button
if st.button(f"Compare {len(selected_players)} Players", use_container_width=True, type="primary"):
    if len(selected_players) > 0:
        
        # --- A. VERSUS BARS ---
        # if len(selected_players) == 2:
        #     st.divider()
        #     st.markdown("<h3 style='text-align: center;'>Head-to-Head</h3>", unsafe_allow_html=True)
        #     st.write("")
            
        #     p1_name = selected_players[0]
        #     p2_name = selected_players[1]
        #     p1_data = main_df[main_df['Player'] == p1_name].iloc[0]
        #     p2_data = main_df[main_df['Player'] == p2_name].iloc[0]
            
        #     stats = [
        #         ('Projected FP', 'Predicted_FP'),
        #         ('Passing Yds', 'Pass_Yds'),
        #         ('Rushing Yds', 'Rush_Yds'),
        #         ('Touchdowns', 'TDs')
        #     ]
            
        #     for label, col in stats:
        #         max_val = main_df[col].max()
        #         # Pass colors[0] for Player 1 and colors[1] for Player 2
        #         draw_versus_bar(label, p1_data[col], p2_data[col], max_val, colors[0], colors[1])

        st.divider()
        st.markdown("<h3 style='text-align: center;'>Head-to-Head Stats</h3>", unsafe_allow_html=True)
        st.write("")
        
        # 1. Define Stats
        stats_map = [
            ('Projected FP', 'Predicted_FP'),
            ('Passing Yds', 'Pass_Yds'),
            ('Rushing Yds', 'Rush_Yds'),
            ('Touchdowns', 'TDs')
        ]
        
        # 2. Prepare Data
        comparison_data = []
        for i, p_name in enumerate(selected_players):
            p_row = main_df[main_df['Player'] == p_name].iloc[0]
            comparison_data.append({
                'name': p_name,
                'row_data': p_row,
                'color': colors[i] # Assign color based on selection index
            })

        # 3. Render in 2 columns
        stat_col1, stat_col2 = st.columns(2)
        
        for idx, (label, col_key) in enumerate(stats_map):
            # Calculate Max for scaling
            max_val = max(d['row_data'][col_key] for d in comparison_data)
            
            # Prepare list for drawing
            draw_list = []
            for d in comparison_data:
                draw_list.append({
                    'name': d['name'],
                    'value': d['row_data'][col_key],
                    'color': d['color']
                })
            
            # Draw
            with stat_col1 if idx % 2 == 0 else stat_col2:
                draw_stat_group(label, draw_list, max_val)
        
        # --- B. STATISTICAL DEEP DIVE (Chart) ---
        st.divider()
        st.subheader("Statistical Deep Dive")
        
        # Prepare Data for Chart
        comp_df = main_df[main_df['Player'].isin(selected_players)].copy()
        
        # Create a dictionary to map Player Name -> Color
        # e.g. {'Josh Allen': '#3b82f6', 'Patrick Mahomes': '#ef4444'}
        color_map = {player: colors[i] for i, player in enumerate(selected_players)}
        
        # Create Altair Chart to enforce specific colors per bar
        chart = alt.Chart(comp_df).mark_bar().encode(
            x=alt.X('Player', axis=None), # Hide X axis labels (redundant with legend/tooltip)
            y=alt.Y('Predicted_FP', title='Projected Points'),
            color=alt.Color('Player', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), legend=alt.Legend(title="Player")),
            tooltip=['Player', 'Predicted_FP', 'Pass_Yds', 'Rush_Yds']
        ).properties(
            height=300
        ).configure_axis(
            grid=False
        ).configure_view(
            strokeWidth=0
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Table
        st.caption("Detailed Projections")
        st.dataframe(
            comp_df.set_index('Player')[['Predicted_FP', 'Pass_Yds', 'Rush_Yds', 'TDs', 'team', 'position']], 
            use_container_width=True
        )
        
    else:
        st.warning("Please select at least one player above.")