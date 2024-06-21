import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch,  VerticalPitch
from scipy.ndimage import gaussian_filter
import seaborn as sns
from tqdm import tqdm
import streamlit as st

pd.options.mode.chained_assignment = None  # default='warn'
from pitch_control import *


xt_matrix = [
  [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.0174506, 0.02122129, 0.02756312, 0.03485072, 0.0379259],
  [0.00750072, 0.00878589, 0.00942382, 0.0105949, 0.01214719, 0.0138454, 0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
  [0.0088799, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.0241224, 0.02855202, 0.05491138, 0.06442595],
  [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.0199707, 0.02385149, 0.03511326, 0.10805102, 0.25745362],
  [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.0199707, 0.02385149, 0.03511326, 0.10805102, 0.25745362],
  [0.0088799, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.0241224, 0.02855202, 0.05491138, 0.06442595],
  [0.00750072, 0.00878589, 0.00942382, 0.0105949, 0.01214719, 0.0138454, 0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
  [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.0174506, 0.02122129, 0.02756312, 0.03485072, 0.0379259]
]
xt_x = np.arange(10, 130, 10)
xt_y = np.arange(10, 90, 10)

def find_opt_pass(team = "home", PPCFhome='', PPCFaway=''):
    if team == "home":
        PPCF = PPCFhome
        xt_df = pd.DataFrame(xt_matrix, columns = xt_x, index=xt_y)
    else:
        PPCF = PPCFaway
        xtm = [a[::-1] for a in xt_matrix]
        xt_df = pd.DataFrame(xtm, columns = xt_x, index=xt_y)

    optimal_pass = []
    for j in range(len(PPCF)):
        optimal_pass_x = []
        for i in range(len(PPCF[j])):
            conv_i = (((i*2)//10)*10)+10
            conv_j = (((j*2)//10)*10)+10
            mult_val = xt_df[conv_i][conv_j]
            pp_cf = PPCF[j][i]
    
            optimal_pass_x.append(pp_cf*mult_val)
            #print(conv_j, conv_i, mult_val, pp_cf, pp_cf*mult_val)
        optimal_pass.append(optimal_pass_x)

    return optimal_pass

def heatmap_opt_pass(df, hm, min=20, sec=38, team='home'):
    cols = [a for a in df.columns.values if ('home' in a) and ('_x' in a or '_y' in a)]

    int_frame = df[(df.minute >= min) & (df.seconds >= sec)].head(1)
    #min = int(int_frame.periodGameClockTime.values[0]/60)
    df_b = int_frame[['ball_x', 'ball_y']].copy()
    
    positions = int_frame[cols].T
    positions.columns = ['avg_pos']
    positions["jersey_num"] = positions.index.str.split("_").str[1]
    positions["coordinate"] = positions.index.str.split("_").str[2]
    
    mean_coords = pd.pivot_table(positions, values="avg_pos", columns="coordinate", index="jersey_num")
    
    pitch = Pitch(line_color = "black")
    fig, ax = pitch.draw(figsize=(10, 7))
    pitch.scatter(mean_coords.x, mean_coords.y, alpha = 1, s = 500, color = "red", ax=ax)
    for i, row in mean_coords.reset_index().iterrows():
        pitch.annotate(row.jersey_num, xy=(row.x, row.y), c='black', va='center', ha='center', weight = "bold", size=16, ax=ax, zorder = 4)
    
    cols = [a for a in df.columns.values if ('away' in a) and ('_x' in a or '_y' in a)]

    int_frame = df[(df.minute >= min) & (df.seconds >= sec)].head(1)
    df_b = int_frame[['ball_x', 'ball_y']].copy()
    
    positions = int_frame[cols].T
    positions.columns = ['avg_pos']
    positions["jersey_num"] = positions.index.str.split("_").str[1]
    positions["coordinate"] = positions.index.str.split("_").str[2]
    
    mean_coords_away = pd.pivot_table(positions, values="avg_pos", columns="coordinate", index="jersey_num")
    pitch.scatter(mean_coords_away.x, mean_coords_away.y, alpha = 1, s = 500, color = "blue", ax=ax)
    pitch.scatter(df_b.ball_x, df_b.ball_y, s=200, ax=ax, zorder=8)
    
    for i, row in mean_coords_away.reset_index().iterrows():
        pitch.annotate(row.jersey_num, xy=(row.x, row.y), c='black', va='center', ha='center', weight = "bold", size=16, ax=ax, zorder = 4)

    vel = [a for a in df.columns.values if ('home' in a) and ('_vx' in a or '_vy' in a)]
    vels = int_frame[vel].T
    vels.columns = ['avg_pos']
    vels["jersey_num"] = vels.index.str.split("_").str[1]
    vels["coordinate"] = vels.index.str.split("_").str[2]
    
    mean_vels = pd.pivot_table(vels, values="avg_pos", columns="coordinate", index="jersey_num")

    full_mean_vels = pd.concat([mean_coords, mean_vels], axis=1)
    for i, row in full_mean_vels.iterrows():
        plt.quiver(row.x, row.y, row.vx, row.vy, color='r', units='xy', scale=1, width=0.5)


    vel = [a for a in df.columns.values if ('away' in a) and ('_vx' in a or '_vy' in a)]
    vels = int_frame[vel].T
    vels.columns = ['avg_pos']
    vels["jersey_num"] = vels.index.str.split("_").str[1]
    vels["coordinate"] = vels.index.str.split("_").str[2]
    
    mean_vels_away = pd.pivot_table(vels, values="avg_pos", columns="coordinate", index="jersey_num")

    full_mean_vels_away = pd.concat([mean_coords_away, mean_vels_away], axis=1)
    for i, row in full_mean_vels_away.iterrows():
        plt.quiver(row.x, row.y, row.vx, row.vy, color='b', units='xy', scale=0.8, width=0.3)

    
    mv = np.nanmax(hm)
    
    field_dimen=(120, 80)
    ax.imshow(hm, extent=(0, field_dimen[0], 0, field_dimen[1]),interpolation='spline36',vmin=0.0,vmax=mv,cmap='Reds',alpha=0.5)
    
    
    #plt.title(f"{team} Number {js_target} Heatmap")
    plt.show()
    return fig

def heatmap(df, jersey=9, team='home'):
    js_target=jersey
    pitch = Pitch(pitch_type='statsbomb', line_zorder=2,
                  pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=(10, 7))
    
    bin_statistic = pitch.bin_statistic(df[f"{team}_{js_target}_x"], df[f"{team}_{js_target}_y"], statistic='count', bins=(25, 25), normalize=True)
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    pitch.heatmap(bin_statistic, ax=ax, cmap='Reds', edgecolor='#f9f9f9')
    
    plt.title(f"{team.capitalize()} Number {js_target} Heatmap")
    plt.show()
    return fig

def tactical_pos(df, team='home'):
    cols = [a for a in df.columns.values if (team in a) and ('_x' in a or '_y' in a)]
    
    avg = df[cols].dropna().mean().to_frame("avg_pos")
    avg["jersey_num"] = avg.index.str.split("_").str[1]
    avg["coordinate"] = avg.index.str.split("_").str[2]

    mean_coords = pd.pivot_table(avg, values="avg_pos", columns="coordinate", index="jersey_num")

    pitch = Pitch(line_color = "black")
    fig, ax = pitch.draw(figsize=(10, 7))
    pitch.scatter(mean_coords.x, mean_coords.y, alpha = 1, s = 500, color = "red", ax=ax)
    
    for i, row in mean_coords.reset_index().iterrows():
        pitch.annotate(row.jersey_num, xy=(row.x, row.y), c='black', va='center', ha='center', weight = "bold", size=16, ax=ax, zorder = 4)
        
    plt.title(f"{team.capitalize()} team average position")
    return fig

def tactical_pos_frame_vel(df, min=20, sec=38, team='home'):
    cols = [a for a in df.columns.values if (team in a) and ('_x' in a or '_y' in a)]
    vel = [a for a in df.columns.values if (team in a) and ('_vx' in a or '_vy' in a)]

    int_frame = df[(df.minute >= min) & (df.seconds >= sec)].head(1)
    min = int(int_frame.periodGameClockTime.values[0]/60)

    df_b = int_frame[['ball_x', 'ball_y']].copy()
    
    positions = int_frame[cols].T
    vels = int_frame[vel].T
    
    positions.columns = ['avg_pos']
    positions["jersey_num"] = positions.index.str.split("_").str[1]
    positions["coordinate"] = positions.index.str.split("_").str[2]
    
    mean_coords = pd.pivot_table(positions, values="avg_pos", columns="coordinate", index="jersey_num")

    vels.columns = ['avg_pos']
    vels["jersey_num"] = vels.index.str.split("_").str[1]
    vels["coordinate"] = vels.index.str.split("_").str[2]
    
    mean_vels = pd.pivot_table(vels, values="avg_pos", columns="coordinate", index="jersey_num")
    
    pitch = Pitch(line_color = "black")
    fig, ax = pitch.draw(figsize=(10, 7))
    pitch.scatter(mean_coords.x, mean_coords.y, alpha = 1, s = 500, color = "red", ax=ax)
    pitch.scatter(df_b.ball_x, df_b.ball_y, s=200, ax=ax, zorder=8)
    
    for i, row in mean_coords.reset_index().iterrows():
        pitch.annotate(row.jersey_num, xy=(row.x, row.y), c='black', va='center', ha='center', weight = "bold", size=16, ax=ax, zorder = 4)

    full_mean_vels = pd.concat([mean_coords, mean_vels], axis=1)
    for i, row in full_mean_vels.iterrows():
        plt.quiver(row.x, row.y, row.vx, row.vy, color='r', units='xy', scale=1, width=0.5)
        
    plt.title(f"{team.capitalize()} position at minute {min}:{sec} - arrows indicating movement")
    return fig

def make_voronoi(df, team_view = "home", min =20, sec=38):
    int_frame = df[(df.minute >= min) & (df.seconds >= sec)].head(1)
    df_b = int_frame[['ball_x', 'ball_y']].copy()
    home_cols = [a for a in df.columns.values if ("home" in a) and ('_x' in a or '_y' in a)]
    away_cols = [a for a in df.columns.values if ("away" in a) and ('_x' in a or '_y' in a)]

    ht_locs = int_frame[home_cols].T#.reset_index()
    ht_locs.columns = ["loc"]
    
    ht_locs["jersey_num"] = ht_locs.index.str.split("_").str[1]
    ht_locs["coordinate"] = ht_locs.index.str.split("_").str[2]
    mean_coords = pd.pivot_table(ht_locs, values="loc", columns="coordinate", index="jersey_num").reset_index()
    home_locs = mean_coords.copy()
    home_locs["team"] = ["home"]*len(home_locs)

    at_locs = int_frame[away_cols].T#.reset_index()
    at_locs.columns = ["loc"]
    
    at_locs["jersey_num"] = at_locs.index.str.split("_").str[1]
    at_locs["coordinate"] = at_locs.index.str.split("_").str[2]
    mean_coords = pd.pivot_table(at_locs, values="loc", columns="coordinate", index="jersey_num").reset_index()
    away_locs = mean_coords.copy()
    away_locs["team"] = ["away"]*len(away_locs)

    int_frame = pd.concat([home_locs, away_locs])
    if team_view == "home":
        int_frame["teammate"] = int_frame.team.apply(lambda x: True if x=="home" else False)
    if team_view == "away":
        int_frame["teammate"] = int_frame.team.apply(lambda x: True if x=="away" else False)
        
    
    pitch  = Pitch(line_color='grey', line_zorder = 1, half = False, linewidth=5)
    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                         endnote_height=0.04, title_space=0, endnote_space=0)

    team1, team2 = pitch.voronoi(int_frame.x, int_frame.y, int_frame.teammate)
    t1 = pitch.polygon(team1, ax = ax["pitch"], color = 'black', ec = 'black', lw=3, alpha=0.4, zorder = 2)
    t2 = pitch.polygon(team2, ax = ax["pitch"], color = 'yellow', ec = 'black', lw=3, alpha=0.4, zorder = 2)
    
    
    pitch.scatter(home_locs.x, home_locs.y, color="black", s=400, ax=ax['pitch'])
    pitch.scatter(away_locs.x, away_locs.y, color="#8B8000", s=400, ax=ax['pitch'], zorder=4)
    pitch.scatter(df_b.ball_x, df_b.ball_y, s=200, ax=ax['pitch'], zorder=8)


    ax["pitch"].set_title("Home (black) team vs Away (yellow) team - space by each player view using Voronoi Diagrams - ball in blue")
        
    plt.show()
    return fig

def tactical_pos_frame_pcm(df, min=20, sec=38, team='home', PPCFhome='', PPCFaway=''):
    field_dimen=(120, 80)
    cols = [a for a in df.columns.values if ('home' in a) and ('_x' in a or '_y' in a)]

    int_frame = df[(df.minute >= min) & (df.seconds >= sec)].head(1)
    #min = int(int_frame.periodGameClockTime.values[0]/60)
    df_b = int_frame[['ball_x', 'ball_y']].copy()
    
    positions = int_frame[cols].T
    positions.columns = ['avg_pos']
    positions["jersey_num"] = positions.index.str.split("_").str[1]
    positions["coordinate"] = positions.index.str.split("_").str[2]
    
    mean_coords = pd.pivot_table(positions, values="avg_pos", columns="coordinate", index="jersey_num")
    
    pitch = Pitch(line_color = "black")
    fig, ax = pitch.draw(figsize=(10, 7))
    pitch.scatter(mean_coords.x, mean_coords.y, alpha = 1, s = 500, color = "red", ax=ax)
    for i, row in mean_coords.reset_index().iterrows():
        pitch.annotate(row.jersey_num, xy=(row.x, row.y), c='black', va='center', ha='center', weight = "bold", size=16, ax=ax, zorder = 4)
    
    cols = [a for a in df.columns.values if ('away' in a) and ('_x' in a or '_y' in a)]

    int_frame = df[(df.minute >= min) & (df.seconds >= sec)].head(1)
    df_b = int_frame[['ball_x', 'ball_y']].copy()
    
    positions = int_frame[cols].T
    positions.columns = ['avg_pos']
    positions["jersey_num"] = positions.index.str.split("_").str[1]
    positions["coordinate"] = positions.index.str.split("_").str[2]
    
    mean_coords_away = pd.pivot_table(positions, values="avg_pos", columns="coordinate", index="jersey_num")
    pitch.scatter(mean_coords_away.x, mean_coords_away.y, alpha = 1, s = 500, color = "blue", ax=ax)
    pitch.scatter(df_b.ball_x, df_b.ball_y, s=200, ax=ax, zorder=8)
    
    for i, row in mean_coords_away.reset_index().iterrows():
        pitch.annotate(row.jersey_num, xy=(row.x, row.y), c='black', va='center', ha='center', weight = "bold", size=16, ax=ax, zorder = 4)

    vel = [a for a in df.columns.values if ('home' in a) and ('_vx' in a or '_vy' in a)]
    vels = int_frame[vel].T
    vels.columns = ['avg_pos']
    vels["jersey_num"] = vels.index.str.split("_").str[1]
    vels["coordinate"] = vels.index.str.split("_").str[2]
    
    mean_vels = pd.pivot_table(vels, values="avg_pos", columns="coordinate", index="jersey_num")

    full_mean_vels = pd.concat([mean_coords, mean_vels], axis=1)
    for i, row in full_mean_vels.iterrows():
        plt.quiver(row.x, row.y, row.vx, row.vy, color='r', units='xy', scale=1, width=0.5)


    vel = [a for a in df.columns.values if ('away' in a) and ('_vx' in a or '_vy' in a)]
    vels = int_frame[vel].T
    vels.columns = ['avg_pos']
    vels["jersey_num"] = vels.index.str.split("_").str[1]
    vels["coordinate"] = vels.index.str.split("_").str[2]
    
    mean_vels_away = pd.pivot_table(vels, values="avg_pos", columns="coordinate", index="jersey_num")

    full_mean_vels_away = pd.concat([mean_coords_away, mean_vels_away], axis=1)
    for i, row in full_mean_vels_away.iterrows():
        plt.quiver(row.x, row.y, row.vx, row.vy, color='b', units='xy', scale=0.8, width=0.3)

    if team=='Home':
        PPCF = PPCFhome
        cmap = 'bwr'
    else:
        PPCF = PPCFaway
        cmap = 'bwr_r'
        
    ax.imshow(PPCF, extent=(0, field_dimen[0], 0, field_dimen[1]),interpolation='spline36',vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)
        
    plt.title(f"Pitch Control - Minute {min}:{sec}")
    return fig

def main():
    st.set_page_config(
        page_title="Dashboard Tracking",
        page_icon="⚽",
    )
    title_html = """
        <style>
            .title {
                font-size: 24px;
            }
        </style>
        <h1 class="title">Tracking Data</h1>
    """

    # Render the HTML using st.write() with unsafe_allow_html=True
    st.write(title_html, unsafe_allow_html=True)

    #dataframe
    df = pd.read_csv("data/tracking_vel_compact.csv")

    # Tabs
    tabs = st.tabs(['Player Heatmap', 'Team Average Position', 'Position at a Moment', 'Pitch Control'])

    with tabs[0]:
        #st.header('Player Heatmap')
        home_players = list(set([int(a.split("_")[1]) for a in df.columns if 'home_' in a]))
        away_players = list(set([int(a.split("_")[1]) for a in df.columns if 'away_' in a]))

        team = st.selectbox('Select Team', ['Home', 'Away'], key='team_heatmap')
        players = {'Home': home_players, 'Away': away_players}
        player_number = st.selectbox('Select Player Jersey Number', players[team])

        team = team.lower()
        fig = heatmap(df, (player_number), team=team)
        st.pyplot(fig)

    with tabs[1]:
        #st.header('Team Average Position')
        team = st.selectbox('Select Team', ['Home', 'Away'], key='team_avg_pos')
        team = team.lower()
        
        fig = tactical_pos(df, team=team)
        
        st.pyplot(fig)

    with tabs[2]:
        #st.header('Position at a Moment')
        
        team = st.selectbox('Select Team', ['Home', 'Away'], key='team_frame')
        
        

        min_minute = int(df['minute'].min())
        max_minute = int(df['minute'].max())
        min_second = int(df['seconds'].min())
        max_second = int(df['seconds'].max())

        col1, col2 = st.columns(2)
        with col1:
            minute = st.number_input('Minute', min_value=min_minute, max_value=max_minute,  value=20, key='minute_number_input')
        with col2:
            second = st.number_input('Second', min_value=min_second, max_value=max_second,  value=38, key='second_number_input')

        team = team.lower()
        fig = tactical_pos_frame_vel(df, min=minute, sec=second, team=team)
        st.pyplot(fig)

        fig = make_voronoi(df,min=minute, sec=second,  team_view=team)
        st.pyplot(fig)

    with tabs[3]:
        #st.header('Pitch Control')    
        team = st.selectbox('Select Team', ['Home', 'Away'], key='team_pcm')
        
        

        min_minute = int(df['minute'].min())
        max_minute = int(df['minute'].max())
        min_second = int(df['seconds'].min())
        max_second = int(df['seconds'].max())

        col1, col2 = st.columns(2)
        with col1:
            minute = st.number_input('Minute', min_value=min_minute, max_value=max_minute,  value=20, key='minute_pcm_input')
        with col2:
            second = st.number_input('Second', min_value=min_second, max_value=max_second,  value=38, key='second_pcm_input')

        team = team.lower()

        #PPCFhome,PPCFaway = generate_pitch_control(df,min = minute, sec=second, field_dimen = (120.,80.,),n_grid_cells_x=60)
        if st.button('Run Pitch Control Analysis', key='run_pitch_control'):
            with st.spinner('Loading Pitch Control Model...'):
                PPCFhome,PPCFaway = generate_pitch_control(df,min = minute, sec=second, field_dimen = (120.,80.,),n_grid_cells_x=60)
            
            fig = tactical_pos_frame_pcm(df, PPCFhome=PPCFhome, PPCFaway=PPCFaway)
            st.pyplot(fig)


            opt_pass = find_opt_pass(team=team, PPCFhome=PPCFhome, PPCFaway=PPCFaway)
            fig = heatmap_opt_pass(df, opt_pass)
            st.pyplot(fig)
            



    
if __name__ == "__main__":
    main()