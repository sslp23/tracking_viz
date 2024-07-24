import pandas as pd
import ast 

def calc_xg(df_sp, df):
    total_xgs = []
    for i, vals in df_sp.iterrows():
        max_time = vals['max_timestamp']
        min_time = vals['timestamp']
        int_team = vals['team']
        total_xg = df[(df.timestamp > min_time) & (df.timestamp < max_time) & (df.team == int_team)]#.shot_statsbomb_xg.sum()
        total_xgs.append(total_xg.shot_statsbomb_xg.sum())

    return total_xgs

def build_table(df):
    pos = ['Carry', 'Ball Receipt*', 'Pass', 'Ball Recovery', 'Block', 'Goal Keeper', 'Shot', 'Dribble', 'Clearance', 'Miscontrol', 'Foul Won', 'Interception']

    pos = df[df.type.isin(pos)]

    df['x'], df['y'] = zip(*df.location.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1]) if not pd.isnull(x) else (x,x)))
    df['end_x'], df['end_y'] = zip(*df.pass_end_location.apply(lambda x: (ast.literal_eval(x)[0], ast.literal_eval(x)[1]) if not pd.isnull(x) else (x,x)))
    
    possession = (pos.groupby('team').size() / len(pos)).to_frame('Ball Possession')

    df_pass = df[df.type == 'Pass']
    df_pass['into F3'] = df_pass.apply(lambda x: True if (x['end_x']>=80) and (x['x']<80) else False, axis=1)

    passes = df_pass.groupby('team').size().to_frame("Passes")
    passes_f3 = df_pass.groupby('team')['into F3'].sum().to_frame("Into F3")

    df_pass['in O3']  = df_pass.apply(lambda x: True if (x['x']<=40) else False, axis=1)
    df_pass['in F3']  = df_pass.apply(lambda x: True if (x['x']>=80) and (x['end_x'] >=80) else False, axis=1)
    df_pass['in att half']  = df_pass.apply(lambda x: True if (x['x']>=60) and (x['end_x'] >= 60) else False, axis=1)

    passes_att_third = df_pass.groupby('team')['in F3'].sum().to_frame("In Att 3rd")
    passes_att_half = df_pass.groupby('team')['in att half'].sum().to_frame("In Att Half")

    comp_pass = df_pass[df_pass.pass_outcome.isna()]
    comp_pass_pct = (comp_pass.groupby('team').size() / passes.Passes.values).to_frame('Complete Pass %')

    df_pass['long pass'] = df_pass.apply(lambda x: True if (x['pass_length']>=25) else False, axis=1)
    long_pass_pct = (df_pass.groupby('team')['long pass'].sum() / passes.Passes.values).to_frame('Long Pass %')

    df_shot = df[df.type == 'Shot']
    shots = df_shot.groupby('team').size().to_frame("Shots")

    shots_ot = (df_shot[df_shot.shot_outcome.isin(['Saved', 'Goal'])].groupby('team').size().to_frame('SoT'))# / shots.Shots.values).to_frame('SoT')
    shots_xg = df_shot.groupby('team').shot_statsbomb_xg.sum().to_frame('xG')

    df_fk = df[(df.pass_type == 'Free Kick')]
    fk = df_fk.groupby('team').size().to_frame("Free Kicks")

    df_c = df[(df.pass_type == 'Corner') | (df.shot_type == 'Corner')]
    cr = df_c.groupby('team').size().to_frame("Corner")

    df_c.timestamp = pd.to_datetime(df_c.timestamp)
    df_fk.timestamp = pd.to_datetime(df_fk.timestamp)
    df_fk['max_timestamp'] = df_fk.timestamp + pd.Timedelta(8, unit='s')
    df_c['max_timestamp'] = df_c.timestamp + pd.Timedelta(8, unit='s')

    df.timestamp = pd.to_datetime(df.timestamp)

    df_c['corner_xg'] = calc_xg(df_c, df)
    df_fk['fk_xg'] = calc_xg(df_fk, df)

    cr_xg = df_c.groupby('team').corner_xg.sum().to_frame("Corner xG")
    fk_xg = df_fk.groupby('team').fk_xg.sum().to_frame("FK xG")

    df_off = df[df.type.isin(['Shot', 'Carry', 'Pass', 'Dribble', 'Ball Receipt*'])]
    df_off['in F3']  = df_off.apply(lambda x: True if (x['x']>=80) else False, axis=1)
    field_tilt = (df_off[df_off['in F3'] == True].groupby('team').size() / len(df_off[df_off['in F3'] == True])).to_frame('Field Tilt')

    df_duel = df[df.type == 'Duel']

    duels  = df_duel.groupby('team').size().to_frame('Duels')
    df_def = df[df.type.isin(['Duel', 'Clearance', 'Foul Committed', 'Interception'])]
    df_def['in F3']  = df_def.apply(lambda x: True if (x['x']>=80) else False, axis=1)

    def_actions = df_def.groupby('team').size().to_frame('Def Actions')
    def_actions_f3 = df_def[df_def['in F3'] == True].groupby('team').size().to_frame('Def Actions F3')  

    ppda = (df_pass[df_pass['in O3'] == True].groupby('team').size() / def_actions_f3['Def Actions F3'].values).to_frame('PPDA')

    return pd.concat([possession, field_tilt, passes, passes_f3, passes_att_third, passes_att_half, comp_pass_pct, long_pass_pct, shots, shots_ot, shots_xg, fk, cr, fk_xg, cr_xg, duels, def_actions, def_actions_f3, ppda], axis=1).T