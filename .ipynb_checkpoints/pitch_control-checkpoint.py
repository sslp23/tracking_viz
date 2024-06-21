import scipy as sci
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch,  VerticalPitch
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.ndimage import gaussian_filter
import seaborn as sns

from tqdm import tqdm
import json

from tqdm import tqdm

import bezier

##BALL ARRIVING TIME
def ball_arriving_time():
    m = 0.45 # In kg
    g = -9.81
    vt = -35 # Terminal velocity downwards
    k = m*g / vt # Drag coefficient

    def T(v):
        q = 1 - ((k*v) / (m*g)) # Just a constant
        
        return (m/k - v/g + (m/k * sci.special.lambertw(-q*np.exp(-q)))).real

    def s_y(t,v_y):
        return (m*g*t)/k + (m/k)*(v_y-(m*g/k))*(1-np.exp(-k*t/m))

    def s_x(t,v_x):
        return m*v_x/k * (1-np.exp(-k*t/m))
    
    max_speed = 35
    v_x_l = np.linspace(0,max_speed,num=500)
    v_y_l = np.linspace(0.001,max_speed,num=500)

    fun_map = np.empty((v_x_l.size, v_y_l.size))
    for i in range(v_x_l.size):
        for j in range(v_y_l.size):
            if((v_y_l[j]**2+(v_x_l[i]**2)>max_speed**2) or v_y_l[j]/v_x_l[i]>2): # Cap max speed and angle
                fun_map[i,j] = np.nan
            else:
                fun_map[i,j] = s_x(T(v_y_l[j]),v_x_l[i])

    ranges = pd.DataFrame(fun_map)
    ranges.columns = v_x_l
    ranges.index = v_y_l

    def time_interval(d):
        df = abs(ranges-d)
        df=df[df<0.2]
        df = df[df.columns[~df.isnull().all()]]
        h = df.columns.values
        
        return [T(h[0]),T(h[-1])] if len(h)>0 else [np.nan,np.nan]
    
    time_ranges = [[d,time_interval(d)] for d in np.linspace(0,65) if not np.isnan(time_interval(d)).any()]
    time_ranges_lower = [[d[0],d[1][0]]for d in time_ranges]
    time_ranges_upper = [[d[0],d[1][1]]for d in time_ranges]
    return time_ranges

def player_arrival_times(df, min, sec, ball_x=-1, ball_y=-1, vel_mult_factor = 3, avg_running_speed = 5, reaction_time = 0.3):
    df_slice = df[(df.minute >= min) & (df.seconds >= sec)].head(1)
    cols = [a for a in df.columns.values if ("home" in a) and (('_x' in a or '_y' in a) or ('_vx' in a or '_vy' in a))]    
    tracking_home_slice = df_slice[cols]

    cols = [a for a in df.columns.values if ("away" in a) and (('_x' in a or '_y' in a) or ('_vx' in a or '_vy' in a))]
    tracking_away_slice = df_slice[cols]    
    
    home_arrival_times = pd.Series([],dtype=pd.StringDtype()) 
    away_arrival_times = pd.Series([],dtype=pd.StringDtype())

    if ball_x == -1 and ball_y == -1:
        ball_x, ball_y = df_slice["ball_x"].values.tolist()[0], df_slice["ball_y"].values.tolist()[0]
        
    for team,arrival_times, team_name in zip([tracking_home_slice,tracking_away_slice],[home_arrival_times,away_arrival_times], ["home", "away"]) :
        #print(team.keys())
        player_ids = np.unique([a.split("_")[1] for a in team.keys()])
        for player in player_ids:
            pos_and_speed = (team[team_name+"_"+player+"_x"].values[0],
                             team[team_name+"_"+player+"_y"].values[0],
                             team[team_name+"_"+player+"_vx"].values[0],
                             team[team_name+"_"+player+"_vy"].values[0])
            
            if(not any(np.isnan(pos_and_speed))):
                x, y, v_x, v_y = pos_and_speed
                dist_to_ball =  np.linalg.norm(np.abs([x-ball_x,y-ball_y]))
                
                u = [x+v_x, y+v_y] # Vector in direction of current motion
                v = [ball_x-x, ball_y-y] # Pos vector of the ball from player pos
                #print((v))
                cos_alpha = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v)
                alpha = np.abs(np.arccos(np.clip(cos_alpha, -1, 1)))
                
                if dist_to_ball < 1:
                    arrival_times[player+"_arrival_time"] = 0
                elif dist_to_ball < 5 and alpha > 3.14/2: # If close & bezier would be too extreme
                    dist_to_go =  np.linalg.norm([x+reaction_time*v_x-ball_x,y+reaction_time*v_y-ball_y])
                    arrival_times[player+"_arrival_time"] = reaction_time + dist_to_go / avg_running_speed
                else: # Bezier curve approach
                    nodes = np.asfortranarray([
                        [x+reaction_time*v_x,x+vel_mult_factor*v_x,ball_x],
                        [y+reaction_time*v_y,y+vel_mult_factor*v_y,ball_y]
                    ])

                    path = bezier.Curve(nodes,degree=2)
                    arrival_times[player+"_arrival_time"] = path.length / avg_running_speed
            else:
                arrival_times[player+"_arrival_time"] = np.nan

    return home_arrival_times,away_arrival_times

def default_model_params(time_to_control_veto=3):
    # key parameters for the model, as described in Spearman 2018
    params = {}
    # model parameters
    params['tti_sigma'] = 0.45 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    params['lambda'] = 4.3 # ball control parameter
    params['lambda_gk'] = params['lambda']*3.0 # make goal keepers must quicker to control ball (because they can catch it)
    
    # numerical parameters for model evaluation
    params['int_dt'] = 0.04 # integration timestep (dt)
    params['max_int_time'] = 10 # upper limit on integral time
    params['model_converge_tol'] = 0.05 # assume convergence when PPCF>0.99 at a given location.
    
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
    # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
    params['time_to_control'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda'])
    return params


def probability_intercept_ball(time_to_intercept,T,params):
    # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
    f = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/params['tti_sigma'] * (T-time_to_intercept) ) )
    return f

def calculate_pitch_control_at_target(target,df,min, sec,params, time_ranges):
    #target -> target coordinate in the field

    df_slice = df[(df.minute >= min) & (df.seconds >= sec)].head(1)
    cols = [a for a in df.columns.values if ("home" in a) and (('_x' in a or '_y' in a) or ('_vx' in a or '_vy' in a))]    
    tracking_home = df_slice[cols]

    cols = [a for a in df.columns.values if ("away" in a) and (('_x' in a or '_y' in a) or ('_vx' in a or '_vy' in a))]
    tracking_away = df_slice[cols]    
    
    #tracking home -> dataset with home team infos (x, y, vx, vy)
    #tracking away -> dataset with away team infos (x, y, vx, vy)
    #time -> desired time
    # Set Up
    ball_start = [df["ball_x"].values[0],df["ball_y"].values[0]]
    
    
    ## 1 - Ball Travel Time Range
    dist = np.linalg.norm(np.asarray(target)-np.asarray(ball_start)) # Dist ball has to travel
    if dist < time_ranges[-1][0]:
        ball_flight_time_interval = time_ranges[np.argmin([np.abs(t[0]-dist) for t in time_ranges])][1] # range of times the ball could take to get there
    else:
        return np.nan , np.nan # No team can control this point
    
    ## 2 - Player Arrival Times
    home_arrival_times,away_arrival_times = player_arrival_times(df, min, sec,ball_x= target[0],ball_y = target[1])
    
    # Remove nans and sort quickest to slowest
    home_arrival_times = np.sort(home_arrival_times[~np.isnan(home_arrival_times)])
    away_arrival_times = np.sort(away_arrival_times[~np.isnan(away_arrival_times)])
    
    # Set up player arrays
    home_players = [{"arrival_time":x,"PPCF":0} for x in home_arrival_times]
    away_players = [{"arrival_time":x,"PPCF":0} for x in away_arrival_times]
    
    tau_min_home = home_arrival_times[0]
    tau_min_away = away_arrival_times[1]
    
    ## 3 - Ball travel time
    ## For the moment just take a normal flight time, could skew this to favour one team or the other
    skew = 0.5 # Can weight to make ball arrive earlier or later, or pick this based on th team with the ball
    ball_travel_time = skew*ball_flight_time_interval[1]+(1-skew)*ball_flight_time_interval[1]
    
    ## 4 - Solving eqtn 3 in the paper
    # Check whether we actually need to solve equation 3
    if tau_min_home-max(ball_travel_time,tau_min_away) >= params['time_to_control']:
        # if away team can arrive significantly before homw team, no need to solve pitch control model
        return 0., 1.
    elif tau_min_away-max(ball_travel_time,tau_min_home) >= params['time_to_control']:
        # if home team can arrive significantly before away team, no need to solve pitch control model
        return 1., 0.
    else: 
        # solve pitch control model by integrating equation 3 in Spearman et al. ]
        # set up integration arrays
        dT_array = np.arange(ball_travel_time-params['int_dt'],ball_travel_time+params['max_int_time'],params['int_dt']) 
        PPCFhome = np.zeros_like( dT_array )
        PPCFaway= np.zeros_like( dT_array )
        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1
       
        while 1-ptot>params['model_converge_tol'] and i<dT_array.size: 
            T = dT_array[i]
            for player in home_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1-PPCFhome[i-1]-PPCFaway[i-1])*probability_intercept_ball(player['arrival_time'],T,params) * params['lambda']
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid homeacking player probability (calculate_pitch_control_at_target)'
                player['PPCF'] += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFhome[i] += player['PPCF'] # add to sum over players in the homeacking team (remembering array element is zero at the start of each integration iteration)
            for player in away_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1-PPCFhome[i-1]-PPCFaway[i-1])*probability_intercept_ball(player['arrival_time'],T,params) * params['lambda']
               # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid awayending player probability (calculate_pitch_control_at_target)'
                player['PPCF'] += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFaway[i] += player['PPCF'] # add to sum over players in the awayending team
            ptot = PPCFaway[i]+PPCFhome[i] # total pitch control probability 
            i += 1
        if i>=dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot) )
        return PPCFhome[i-1], PPCFaway[i-1]
    

def generate_pitch_control(df,min, sec, field_dimen = (120.,80.,), n_grid_cells_x = 50):
    df_slice = df[(df.minute >= min) & (df.seconds >= sec)].head(1)
    cols = [a for a in df.columns.values if ("home" in a) and (('_x' in a or '_y' in a) or ('_vx' in a or '_vy' in a))]    
    tracking_home = df_slice[cols]

    cols = [a for a in df.columns.values if ("away" in a) and (('_x' in a or '_y' in a) or ('_vx' in a or '_vy' in a))]
    tracking_away = df_slice[cols]
    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
    dx = field_dimen[0]/n_grid_cells_x
    dy = field_dimen[1]/n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x)*dx + dx/2.
    ygrid = np.arange(n_grid_cells_y)*dy + dy/2.
    
    # initialise pitch control grids for home and away teams
    PPCFhome = np.zeros( shape = (len(ygrid), len(xgrid)) )
    PPCFaway = np.zeros( shape = (len(ygrid), len(xgrid)) )

    time_ranges = ball_arriving_time()
    params = default_model_params()
    # calculate pitch pitch control model at each location on the pitch
    checksum = 0
    num_grid_squares_in_range = 0
    for i in tqdm(range( len(ygrid) )):
        for j in range( len(xgrid) ):
            target = np.array( [xgrid[j], ygrid[i]] )
            
            PPCFhome[i,j],PPCFaway[i,j] = calculate_pitch_control_at_target(target,df,min,sec,params, time_ranges)
            
            if np.isfinite(PPCFhome[i,j]) and np.isfinite(PPCFaway[i,j]):
                checksum += PPCFhome[i,j]+PPCFaway[i,j]
                num_grid_squares_in_range +=1
    
    assert checksum/num_grid_squares_in_range > 0.95
    print(checksum/num_grid_squares_in_range)
    return PPCFhome,PPCFaway