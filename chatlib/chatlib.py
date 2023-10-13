import pandas as pd
from pprint import pprint
import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd().parent.parent.joinpath('mlBridgeLib'))) # removed .parent
sys.path.append(str(pathlib.Path.cwd().parent.parent.joinpath('acbllib'))) # removed .parent
sys.path
import mlBridgeLib
import acbllib


# obsolete?
#from tenacity import retry, wait_random_exponential, stop_after_attempt
#from termcolor import colored
#import dotenv
#from dotenv import load_dotenv
#import os
#import inspect
# def pretty_print_conversation(messages):
#     role_to_color = {
#         "system": "red",
#         "user": "green",
#         "assistant": "blue",
#         "function": "magenta",
#     }
#     formatted_messages = []
#     for message in messages:
#         if message["role"] == "system":
#             formatted_messages.append(f"system: {message['content']}\n")
#         elif message["role"] == "user":
#             formatted_messages.append(f"user: {message['content']}\n")
#         elif message["role"] == "assistant":
#             if message.get("function_call"):
#                 formatted_messages.append(
#                     f"assistant: {message['function_call']}\n")
#             else:
#                 formatted_messages.append(f"assistant: {message['content']}\n")
#         elif message["role"] == "function":
#             formatted_messages.append(f"assistant: {message['content']}\n")
#             #formatted_messages.append(f"function ({message['name']}): {message['content']}\n")
#     for formatted_message in formatted_messages:
#         print(
#             colored(
#                 formatted_message,
#                 role_to_color[messages[formatted_messages.index(
#                     formatted_message)]["role"]],
#             )
#         )


# merge acbl json dicts into logically related dicts. dicts will be used to create dfs
def json_dict_to_df(d,kl,jdl):
    print(kl)
    dd = {}
    d[kl] = dd
    assert not isinstance(jdl,dict)
    for i,jd in enumerate(jdl):
        for k,v in jd.items():
            kkl = kl+(k,i)
            print(i,kl,k,kkl)
            #time.sleep(.00001)
            if isinstance(v,list):
                print('\n',type(v),kkl,v)
                json_dict_to_df(d,kkl,v)
                print('list:',kkl,len(d[kkl]))
            elif isinstance(v,dict):
                #kkl = kl+(k,)
                print('\n',type(v),kkl,v)
                json_dict_to_df(d,kkl,[v])
                print('dict:',kkl,len(d[kkl]))
            else:
                if k not in dd:
                    dd[k] = []
                dd[k].append(v)
            #assert k != 'points',[kl,k,type(v),v]
    return d


# todo: obsolete?
# todo: if dtype isnumeric() downcast to minimal size. Some columns may have dtype of int64 because of sql declaration ('Board').
def convert_to_best_dtype(k,v):
    vv = v.convert_dtypes(infer_objects=True)
    vvv = vv.copy()
    for col in vv.columns:
        print(col,vvv[col].dtype)
        # todo: special cases. maybe should be done before writing to acbl_club_results.sqlite?
        if col in ['ns_score','ew_score']:
            vvv[col] = vvv[col].replace('PASS','0')
        elif col == 'result':
            vvv[col] = vvv[col].replace('+','0').replace('=','0').replace('','0') # don't use .str. and all 3 are needed.
        elif col == 'tricks_taken':
            vvv[col] = vvv[col].replace('','0')
        if vvv[col].dtype == 'string' and vvv[col].notna().all() and vvv[col].ne('').all():
            print(f"String: {col}")
            try:
                if vvv[col].str.contains('.',regex=False).any():
                    print(f"Trying to convert {col} to float")
                    converted_values = pd.to_numeric(vvv[col], downcast='float', errors='raise')
                elif vvv[col].str.contains('-',regex=False).any():
                    print(f"Trying to convert {col} to integer")
                    converted_values = pd.to_numeric(vvv[col], downcast='integer', errors='raise')
                else:
                    print(f"Trying to convert {col} to unsigned")
                    converted_values = pd.to_numeric(vvv[col], downcast='unsigned', errors='raise')
                vvv[col] = converted_values
                print(f"Converted {col} to {vvv[col].dtype}")
            except ValueError:
                print(f"Can't convert {col} to float. Keeping as string")
    print(f"dfs_dtype_conversions['{k}'] = "+'{')
    for col in vvv.columns:
        print(f"    '{col}':'{v[col].dtype},{vv[col].dtype},{vvv[col].dtype}',")
    print("}\n")
    return vvv


# g_all_functions_in_module = {n:f for n,f in inspect.getmembers(sys.modules[__name__], inspect.isfunction)}

def json_dict_to_types(json_dict,root_name,path):
    dfs = {}
    root = []
    df = pd.json_normalize(json_dict,path,max_level=0)
    for k,v in df.items():
        if isinstance(v,dict):
            assert k not in dfs, k
            dfs[k] = pd.DataFrame(v)
            if all(isinstance(kk,int) or (isinstance(kk,str) and kk.isnumeric()) for kk,vv in v.items()): # dict but with list like indices
                dfs[k] = dfs[k].T
        elif isinstance(v,list):
            assert k not in dfs, k
            dfs[k] = pd.DataFrame(v)
        else:
            root.append({k:v})
    assert k not in dfs, k
    dfs[root_name] = pd.DataFrame(root)
    return dfs


def create_club_dfs(acbl_number,event_url):
    data = acbllib.get_club_results_details_data(event_url)
    if data is None:
        return None
    dfs = {}
    dfs['event'] = pd.json_normalize(data,max_level=0)
    for k,v in dfs['event'].items():
        if isinstance(v[0],dict) or isinstance(v[0],list):
            assert k not in dfs, k
            df = pd.json_normalize(data,max_level=0)[k]
            # must test whether df is all scalers. Very difficult to debug.
            if isinstance(v[0],dict) and not any([isinstance(vv,dict) or isinstance(vv,list) for kk,vv in df[0].items()]):
                dfs[k] = pd.DataFrame.from_records(df[0],index=[0]) # must use from_records to avoid 'all values are scaler must specify index' error
            else:
                dfs[k] = pd.DataFrame.from_records(df[0])
            dfs['event'].drop(columns=[k],inplace=True)
            #if all(isinstance(kk,int) or (isinstance(kk,str) and kk.isnumeric()) for kk,vv in v.items()):
    dfs['hand_records'] = pd.json_normalize(data,['sessions','hand_records'])
    dfs['strat_place'] = pd.json_normalize(data,['sessions','sections','pair_summaries','strat_place'])
    dfs['sections'] = pd.json_normalize(data,['sessions','sections'])
    dfs['boards'] = pd.json_normalize(data,['sessions','sections','boards'])
    dfs['pair_summaries'] = pd.json_normalize(data,['sessions','sections','pair_summaries'])
    dfs['players'] = pd.json_normalize(data,['sessions','sections','pair_summaries','players'])
    dfs['board_results'] = pd.json_normalize(data,['sessions','sections','boards','board_results'])
    return dfs


# ['mp_won', 'mp_color', 'percentage', 'score', 'sanction', 'event_id', 'session_id', 'trax_master_event_code', 'score_tournament_name', 'score_event_name', 'score_session_number', 'score_session_time_description', 'score_event_type', 'score_score_type', 'section', 'results_last_updated', 'session', 'event', 'tournament', 'date']
def merge_clean_augment_tournament_dfs(dfs, dfs_results, acbl_api_key, acbl_number):

    print('dfs keys:',dfs.keys())

    df = pd.DataFrame({k:[v] for k,v in dfs.items() if not (isinstance(v,dict) or isinstance(v,list))})
    print('df:\n', df)
    assert len(df) == 1, len(df)
    
    print('dfs session:',type(dfs['session']))
    df_session = pd.DataFrame({k:[v] for k,v in dfs['session'].items() if not (isinstance(v,dict) or isinstance(v,list))})
    assert len(df_session) == 1, len(df_session)
    print({k:pd.DataFrame(v) for k,v in dfs['session'].items() if (isinstance(v,dict) or isinstance(v,list))})

    print('dfs event:',type(dfs['event']))
    df_event = pd.DataFrame({k:[v] for k,v in dfs['event'].items() if not (isinstance(v,dict) or isinstance(v,list))})
    assert len(df_event) == 1, len(df_event)
    print({k:pd.DataFrame(v) for k,v in dfs['event'].items() if (isinstance(v,dict) or isinstance(v,list))})

    print('dfs tournament:',type(dfs['tournament']))
    df_tournament = pd.DataFrame({k:[v] for k,v in dfs['tournament'].items() if not (isinstance(v,dict) or isinstance(v,list))})
    assert len(df_tournament) == 1, len(df_tournament)
    print({k:pd.DataFrame(v) for k,v in dfs['tournament'].items() if (isinstance(v,dict) or isinstance(v,list))})

    for col in df.columns:
        print(col,df[col].dtype)

    # dfs scalers: ['_id', '_event_id', 'id', 'session_number', 'start_date', 'start_time', 'description', 'sess_type', 'box_number', 'is_online', 'results_available', 'was_not_played', 'results_last_updated']
    # dfs dicts: ['tournament', 'event', 'handrecord', 'sections']
    # dfs lists: ['overalls']
   
    print('dfs_results tournament:',type(dfs_results['tournament']))
    df_results_tournament = pd.DataFrame({k:[v] for k,v in dfs_results['tournament'].items() if not (isinstance(v,dict) or isinstance(v,list))})
    assert len(df_results_tournament) == 1, len(df_results_tournament)
    print({k:pd.DataFrame(v) for k,v in dfs_results['tournament'].items() if (isinstance(v,dict) or isinstance(v,list))})

    print('dfs_results event:',type(dfs_results['event']))
    df_results_event = pd.DataFrame({k:[v] for k,v in dfs_results['event'].items() if not (isinstance(v,dict) or isinstance(v,list))})
    assert len(df_event) == 1, len(df_event)
    print({k:pd.DataFrame(v) for k,v in dfs_results['event'].items() if (isinstance(v,dict) or isinstance(v,list))})

    print('dfs_results overalls:',type(dfs_results['overalls']))
    df_results_overalls = pd.DataFrame(dfs_results['overalls'])
    #assert len(df_results_overalls) == 1, len(df_results_overalls)
    print(pd.DataFrame(dfs_results['overalls']))

    print('dfs_results handrecord:',type(dfs_results['handrecord']))
    df_results_handrecord = pd.DataFrame(dfs_results['handrecord'])
    #assert len(df_results_handrecord) == 1, len(df_results_handrecord)
    print(pd.DataFrame(dfs_results['handrecord']))

    print('dfs_results sections:',type(dfs_results['sections']))
    df_results_sections = pd.DataFrame(dfs_results['sections'])

    df_board_results = pd.DataFrame()
    for i,section in df_results_sections.iterrows():
        br = pd.DataFrame(section['board_results'])
        if all(br['pair_acbl'].map(lambda x: int(acbl_number) not in x)): # if acbl_number is not in this section then skip
            continue # todo: what to do with sections not containing acbl_number? concat all sections? concat may be correct since they may be included in matchpoint calculations.
        df_board_results = pd.concat([df_board_results,br],axis='rows')
        ns_df = df_board_results[df_board_results['orientation'].eq('N-S')]
        ew_df = df_board_results[df_board_results['orientation'].eq('E-W')][['board_number','pair_number','pair_names','pair_acbl','score','match_points','percentage']]
        df_board_results = pd.merge(ns_df,ew_df,left_on=['board_number','opponent_pair_number'],right_on=['board_number','pair_number'],suffixes=('_NS','_EW'),how='left')
        df_board_results.drop(['opponent_pair_number','opponent_pair_names'],inplace=True,axis='columns')
        df_board_results.rename({
            'board_number':'Board',
            'contract':'Contract',
            'score_NS':'Score_NS',
            'score_EW':'Score_EW',
            'match_points_NS':'MatchPoints_NS',
            'match_points_EW':'MatchPoints_EW',
            'percentage_NS':'Pct_NS',
            'percentage_EW':'Pct_EW',
            'pair_number_NS':'Pair_Number_NS',
            'pair_number_EW':'Pair_Number_EW',
            'session_number':'Session',
        },axis='columns',inplace=True)
        df_board_results['pair_direction'] = df_board_results['orientation'].map({'N-S':'NS','E-W':'EW'})
        df_board_results['player_number_n'] = df_board_results.apply(lambda r: r['pair_acbl_NS'][0],axis='columns').astype('string')
        df_board_results['player_number_s'] = df_board_results.apply(lambda r: r['pair_acbl_NS'][1],axis='columns').astype('string')
        df_board_results['player_number_e'] = df_board_results.apply(lambda r: r['pair_acbl_EW'][0],axis='columns').astype('string')
        df_board_results['player_number_w'] = df_board_results.apply(lambda r: r['pair_acbl_EW'][1],axis='columns').astype('string')
        df_board_results['player_name_n'] = df_board_results.apply(lambda r: r['pair_names_NS'][0],axis='columns')
        df_board_results['player_name_s'] = df_board_results.apply(lambda r: r['pair_names_NS'][1],axis='columns')
        df_board_results['player_name_e'] = df_board_results.apply(lambda r: r['pair_names_EW'][0],axis='columns')
        df_board_results['player_name_w'] = df_board_results.apply(lambda r: r['pair_names_EW'][1],axis='columns')
        # todo: get from masterpoint dict
        df_board_results['Club'] = '12345678'
        df_board_results['Session'] = '87654321'
        df_board_results['mp_total_n'] = 300
        df_board_results['mp_total_e'] = 300
        df_board_results['mp_total_s'] = 300
        df_board_results['mp_total_w'] = 300
        df_board_results['MP_Sum_NS'] = 300
        df_board_results['MP_Sum_EW'] = 300
        df_board_results['MP_Geo_NS'] = 300
        df_board_results['MP_Geo_EW'] = 300
        df_board_results['declarer'] = df_board_results['declarer'].map(lambda x: x[0].upper() if len(x) else None) # None is needed for PASS
        df_board_results['Pct_NS'] = df_board_results['Pct_NS'].div(100)
        df_board_results['Pct_EW'] = df_board_results['Pct_EW'].div(100)
        df_board_results['table_number'] = None
        df_board_results['Round'] = None
        df_board_results['dealer'] = df_board_results['Board'].map(mlBridgeLib.BoardNumberToDealer)
        df_board_results['Vul'] = df_board_results['Board'].map(mlBridgeLib.BoardNumberToVul) # 0 to 3 # todo: use 'vul' instead for consistency?
        df_board_results['event_id'] = section['session_id'] # for club compatibility
        df_board_results['section_name'] = section['section_label'] # for club compatibility
        df_board_results['section_id'] = df_board_results['event_id']+'-'+df_board_results['section_name'] # for club compatibility
        df_board_results['Date'] = df_event['start_date'] # for club compatibility
        df_board_results['game_type'] = df_event['game_type'] # for club compatibility
        board_to_brs_d = dict(zip(df_results_handrecord['board_number'],mlBridgeLib.hrs_to_brss(df_results_handrecord)))
        df_board_results['board_record_string'] = df_board_results['Board'].map(board_to_brs_d)
        df_board_results.drop(['orientation','pair_acbl_NS', 'pair_acbl_EW', 'pair_names_NS', 'pair_names_EW'],inplace=True,axis='columns')


    df = clean_validate_df(df_board_results)
    df, sd_cache_d, matchpoint_ns_d = augment_df(df,{})

    return df, sd_cache_d, matchpoint_ns_d


# obsolete?
def clean_validate_tournament_df(df):

    # par, hand_record_id, DD, Vul, Hands, board_record_string?, ns_score, ew_score, Final_Stand_NS|EW, MatchPoints_NS, MatchPoints_EW, player_number_[nesw], contract, BidLvl, BidSuit, Dbl, Declarer_Direction

    # change clean_validate_club_df to handle these missing columns; par, Pair_Number_NS|EW, table_number, round_number, double_dummy_ns|ew, board_record_string, hand_record_id.

    return df


# obsolete?
def augment_tournament_df(df,sd_cache_d):
    return df, sd_cache_d, {}


def merge_clean_augment_club_dfs(dfs,sd_cache_d,acbl_number): # todo: acbl_number obsolete?

    print('dfs keys:',dfs.keys())

    df_brs = dfs['board_results']
    pprint(df_brs.head(1))
    assert len(df_brs.filter(regex=r'_[xy]$').columns) == 0,df_brs.filter(regex=r'_[xy]$').columns

    df_b = dfs['boards'].rename({'id':'board_id'},axis='columns')[['board_id','section_id','board_number']]
    pprint(df_b.head(1))
    assert len(df_b.filter(regex=r'_[xy]$').columns) == 0,df_b.filter(regex=r'_[xy]$').columns

    df_br_b = pd.merge(df_brs,df_b,on='board_id',how='left')
    pprint(df_br_b.head(1))
    assert len(df_br_b) == len(df_brs)
    assert len(df_br_b.filter(regex=r'_[xy]$').columns) == 0,df_br_b.filter(regex=r'_[xy]$').columns


    df_sections = dfs['sections'].rename({'id':'section_id','name':'section_name'},axis='columns').drop(['created_at','updated_at','pair_summaries','boards'],axis='columns') # ['pair_summaries','boards'] are unwanted dicts
    pprint(df_sections.head(1))


    df_br_b_sections = pd.merge(df_br_b,df_sections,on='section_id',how='left')
    pprint(df_br_b_sections.head(1))
    assert len(df_br_b_sections) == len(df_br_b)
    assert len(df_br_b_sections.filter(regex=r'_[xy]$').columns) == 0,df_br_b_sections.filter(regex=r'_[xy]$').columns


    df_sessions = dfs['sessions'].rename({'id':'session_id','number':'session_number'},axis='columns').drop(['created_at','updated_at','hand_records','sections'],axis='columns') # ['hand_records','sections'] are unwanted dicts
    # can't convert to int64 because SHUFFLE is a valid hand_record_id. Need to treat as string.
    # df_sessions['hand_record_id'] = df_sessions['hand_record_id'].astype('int64') # change now for merge
    pprint(df_sessions.head(1))


    df_br_b_sections_sessions = pd.merge(df_br_b_sections,df_sessions,on='session_id',how='left')
    pprint(df_br_b_sections_sessions.head(1))
    assert len(df_br_b_sections_sessions) == len(df_br_b_sections)
    assert len(df_br_b_sections_sessions.filter(regex=r'_[xy]$').columns) == 0,df_br_b_sections_sessions.filter(regex=r'_[xy]$').columns


    df_clubs = dfs['club'].rename({'id':'event_id','name':'club_name','type':'club_type'},axis='columns').drop(['created_at','updated_at'],axis='columns') # name and type are renamed to avoid conflict with df_events
    pprint(df_clubs.head(1))


    df_br_b_sections_sessions_clubs = pd.merge(df_br_b_sections_sessions,df_clubs,on='event_id',how='left')
    pprint(df_br_b_sections_sessions_clubs.head(1))
    assert len(df_br_b_sections_sessions_clubs) == len(df_br_b_sections)
    assert len(df_sections.filter(regex=r'_[xy]$').columns) == 0,df_sections.filter(regex=r'_[xy]$').columns

        
    df_events = dfs['event'].rename({'id':'event_id','club_name':'event_club_name'},axis='columns').drop(['created_at','updated_at','deleted_at'],axis='columns')
    pprint(df_events.head(1))


    df_br_b_sections_sessions_events = pd.merge(df_br_b_sections_sessions_clubs,df_events,on='event_id',how='left')
    pprint(df_br_b_sections_sessions_events.head(1))
    assert len(df_br_b_sections_sessions_events) == len(df_br_b_sections_sessions_clubs)
    assert len(df_br_b_sections_sessions_events.filter(regex=r'_[xy]$').columns) == 0,df_br_b_sections_sessions_events.filter(regex=r'_[xy]$').columns


    df_pair_summaries = dfs['pair_summaries'].rename({'id':'pair_summary_id'},axis='columns').drop(['created_at','updated_at'],axis='columns')
    pprint(df_pair_summaries.head(1))

    # todo: merge df_pair_summaries with strat_place. issue is that strat_place has multiple rows per pair_summary_id
    df_pair_summaries_strat = df_pair_summaries
    # df_strat_place = dfs['strat_place'].rename({'rank':'strat_rank','type':'strat_type'},axis='columns').drop(['id','created_at','updated_at'],axis='columns')
    # pprint(df_strat_place.head(1))

    # df_pair_summaries_strat = pd.merge(df_pair_summaries,df_strat_place,on='pair_summary_id',how='left')
    # pprint(df_pair_summaries_strat.head(1))
    # assert len(df_pair_summaries_strat.filter(regex=r'_[xy]$').columns) == 0,df_pair_summaries_strat.filter(regex=r'_[xy]$').columns

    df_br_b_pair_summary_ns = df_pair_summaries_strat[df_pair_summaries_strat['direction'].eq('NS')].add_suffix('_NS').rename({'pair_number_NS':'ns_pair','section_id_NS':'section_id'},axis='columns')
    assert len(df_br_b_pair_summary_ns.filter(regex=r'_[xy]$').columns) == 0,df_br_b_pair_summary_ns.filter(regex=r'_[xy]$').columns
    df_br_b_pair_summary_ew = df_pair_summaries_strat[df_pair_summaries_strat['direction'].eq('EW')].add_suffix('_EW').rename({'pair_number_EW':'ew_pair','section_id_EW':'section_id'},axis='columns')
    assert len(df_br_b_pair_summary_ew.filter(regex=r'_[xy]$').columns) == 0,df_br_b_pair_summary_ew.filter(regex=r'_[xy]$').columns

    df_players = dfs['players'].drop(['id','created_at','updated_at'],axis='columns').rename({'id_number':'player_number','name':'player_name'},axis='columns')
    player_n = df_players.groupby('pair_summary_id').first().reset_index().add_suffix('_n').rename({'pair_summary_id_n':'pair_summary_id_NS'},axis='columns')
    player_s = df_players.groupby('pair_summary_id').last().reset_index().add_suffix('_s').rename({'pair_summary_id_s':'pair_summary_id_NS'},axis='columns')
    player_e = df_players.groupby('pair_summary_id').first().reset_index().add_suffix('_e').rename({'pair_summary_id_e':'pair_summary_id_EW'},axis='columns')
    player_w = df_players.groupby('pair_summary_id').last().reset_index().add_suffix('_w').rename({'pair_summary_id_w':'pair_summary_id_EW'},axis='columns')

    player_ns = pd.merge(player_n,player_s,on='pair_summary_id_NS',how='left')
    pprint(player_ns.head(1))
    assert len(player_ns) == len(player_n)
    assert len(player_ns.filter(regex=r'_[xy]$').columns) == 0,player_ns.filter(regex=r'_[xy]$').columns
    player_ew = pd.merge(player_e,player_w,on='pair_summary_id_EW',how='left')
    pprint(player_ew.head(1))
    assert len(player_ew) == len(player_e)
    assert len(player_ew.filter(regex=r'_[xy]$').columns) == 0,player_ew.filter(regex=r'_[xy]$').columns

    # due to an oddity with merge(), must never merge on a column that has NaNs. This section avoids that but at the cost of added complexity.
    df_pair_summary_players_ns = pd.merge(df_br_b_pair_summary_ns,player_ns,on='pair_summary_id_NS',how='left')
    assert len(df_pair_summary_players_ns) == len(df_br_b_pair_summary_ns)
    df_pair_summary_players_ew = pd.merge(df_br_b_pair_summary_ew,player_ew,on='pair_summary_id_EW',how='left')
    assert len(df_pair_summary_players_ew) == len(df_br_b_pair_summary_ew)
    #df_pair_summary_players = pd.merge(df_pair_summary_players_ns,df_pair_summary_players_ew,how='left') # yes, on is not needed
    #assert len(df_pair_summary_players) == len(df_pair_summary_players_ns) # likely this is an issue on an EW sitout. Need to compare ns,ew lengths and how on the longer one?
    df_br_b_sections_sessions_events_pair_summary_players = pd.merge(df_br_b_sections_sessions_events,df_pair_summary_players_ns,on=('section_id','ns_pair'),how='left') # yes, requires inner. Otherwise right df non-on columns will be NaNs.
    pprint(df_br_b_sections_sessions_events_pair_summary_players.head(1))
    assert len(df_br_b_sections_sessions_events_pair_summary_players) == len(df_br_b_sections_sessions_events)
    assert len(df_br_b_sections_sessions_events_pair_summary_players.filter(regex=r'_[xy]$').columns) == 0,df_br_b_sections_sessions_events_pair_summary_players.filter(regex=r'_[xy]$').columns
    df_br_b_sections_sessions_events_pair_summary_players = pd.merge(df_br_b_sections_sessions_events_pair_summary_players,df_pair_summary_players_ew,on=('section_id','ew_pair'),how='left') # yes, requires inner. Otherwise right df non-on columns will be NaNs.
    pprint(df_br_b_sections_sessions_events_pair_summary_players.head(1))
    assert len(df_br_b_sections_sessions_events_pair_summary_players) == len(df_br_b_sections_sessions_events)
    assert len(df_br_b_sections_sessions_events_pair_summary_players.filter(regex=r'_[xy]$').columns) == 0,df_br_b_sections_sessions_events_pair_summary_players.filter(regex=r'_[xy]$').columns

    df_hrs = dfs['hand_records'].rename({'hand_record_set_id':'hand_record_id'},axis='columns').drop(['points.N','points.E','points.S','points.W'],axis='columns') # don't want points (HCP) from hand_records. will compute later.
    pprint(df_hrs.head(1))

    df_br_b_sections_sessions_events_pair_summary_players_hrs = pd.merge(df_br_b_sections_sessions_events_pair_summary_players,df_hrs.astype({'hand_record_id':'string'}).drop(['id','created_at','updated_at'],axis='columns'),left_on=('hand_record_id','board_number'),right_on=('hand_record_id','board'),how='left')
    pprint(df_br_b_sections_sessions_events_pair_summary_players_hrs.head(1))
    assert len(df_br_b_sections_sessions_events_pair_summary_players_hrs) == len(df_br_b_sections_sessions_events_pair_summary_players)
    assert len(df_br_b_sections_sessions_events_pair_summary_players_hrs.filter(regex=r'_[xy]$').columns) == 0,df_br_b_sections_sessions_events_pair_summary_players_hrs.filter(regex=r'_[xy]$').columns


    df = df_br_b_sections_sessions_events_pair_summary_players_hrs
    for col in df.columns:
        print(col,df[col].dtype)

    df.drop(['id','created_at','updated_at','board_id','double_dummy_ns','double_dummy_ew','pair_summary_id_NS','pair_summary_id_EW'],axis='columns',inplace=True)

    df.rename({
        'board':'Board',
        'club_id_number':'Club',
        'contract':'Contract',
        'game_date':'Date',
        'ns_pair':'Pair_Number_NS',
        'ew_pair':'Pair_Number_EW',
        'ns_match_points':'MatchPoints_NS',
        'ew_match_points':'MatchPoints_EW',
        'ns_score':'Score_NS',
        'ew_score':'Score_EW',
        'session_number':'Session',
        'tricks_taken':'Tricks',
        'percentage_NS':'Final_Standing_NS',
        'percentage_EW':'Final_Standing_EW',
        'result':'Result',
        'round_number':'Round',
       },axis='columns',inplace=True)

    # columns unique to club results
    df.astype({
        'Final_Standing_NS':'float32',
        'Final_Standing_EW':'float32',
        'hand_record_id':'string',
        'board_record_string':'string',
        })

    df = clean_validate_df(df)
    df, sd_cache_d, matchpoint_ns_d = augment_df(df,sd_cache_d)

    return df, sd_cache_d, matchpoint_ns_d


def clean_validate_df(df):

    df.rename({'declarer':'Declarer_Direction'},axis='columns',inplace=True)

    # Cleanup all sorts of columns. Create new columns where missing.
    df['Board'] = df['Board'].astype('uint8')
    assert df['Board'].ge(1).all()

    assert 'Board_Top' not in df.columns
    tops = {}
    for b in df['Board'].unique():
        tops[b] = df[df['Board'].eq(b)]['MatchPoints_NS'].count()-1
        assert tops[b] == df[df['Board'].eq(b)]['MatchPoints_EW'].count()-1
    # if any rows were dropped, the calculation of board's top/pct will be wrong (outside of (0,1)). Need to calculate Board_Top before dropping any rows.
    df['Board_Top'] = df['Board'].map(tops)
    if set(['Pct_NS', 'Pct_EW']).isdisjoint(df.columns): # disjoint means no elements of set are in df.columns
        df['Pct_NS'] = df['MatchPoints_NS'].astype('float32').div(df['Board_Top'])
        df['Pct_EW'] = df['MatchPoints_EW'].astype('float32').div(df['Board_Top'])
    assert set(['Pct_NS', 'Pct_EW', 'Board_Top']).issubset(df.columns) # subset means all elements of the set are in df.columns
    df.loc[df['Pct_NS']>1,'Pct_NS'] = 1 # assuming this can only happen if director adjusts score. todo: print >1 cases.
    assert df['Pct_NS'].between(0,1).all(), [df[~df['Pct_NS'].between(0,1)][['Board','MatchPoints_NS','Board_Top','Pct_NS']]]
    df.loc[df['Pct_EW']>1,'Pct_EW'] = 1 # assuming this can only happen if director adjusts score. todo: print >1 cases.
    assert df['Pct_EW'].between(0,1).all(), [df[~df['Pct_EW'].between(0,1)][['Board','MatchPoints_EW','Board_Top','Pct_EW']]]

    # transpose pair_name (last name, first_name).
    for d in 'NSEW':
        df.rename({'player_number_'+d.lower():'Player_Number_'+d for d in 'NSEW'},axis='columns',inplace=True)
        df['Player_Name_'+d] = df['player_name_'+d.lower()].str.split(',').str[::-1].str.join(' ') # github Copilot wrote this line!
        df.drop(['player_name_'+d.lower()],axis='columns',inplace=True)

    # clean up contracts. Create BidLvl, BidSuit, Dbl columns.
    contractdf = df['Contract'].str.replace(' ','').str.upper().str.replace('NT','N').str.extract(r'^(?P<BidLvl>\d)(?P<BidSuit>C|D|H|S|N)(?P<Dbl>X*)$')
    df['BidLvl'] = contractdf['BidLvl']
    df['BidSuit'] = contractdf['BidSuit']
    df['Dbl'] = contractdf['Dbl']
    del contractdf
    # There's all sorts of exceptional crap which needs to be done for 'PASS', 'NP', 'BYE', 'AVG', 'AV+', 'AV-', 'AVG+', 'AVG-', 'AVG+/-'. Only 'PASS' is handled, the rest are dropped.
    drop_rows = df['Contract'].ne('PASS')&(df['Score_NS'].eq('PASS')&df['Score_EW'].eq('PASS')&df['BidLvl'].isna()|df['BidSuit'].isna()|df['Dbl'].isna())
    print('Invalid contracts: drop_rows:',drop_rows.sum(),df[drop_rows][['Contract','BidLvl','BidSuit','Dbl']])
    df.drop(df[drop_rows].index,inplace=True)
    drop_rows = ~df['Declarer_Direction'].isin(list('NSEW')) # keep N,S,E,W. Drop EW, NS, w, ... < 500 cases.
    print('Invalid declarers: drop_rows:',drop_rows.sum(),df[drop_rows][['Declarer_Direction']])
    df.drop(df[drop_rows].index,inplace=True)
    df.loc[df['Contract'].ne('PASS'),'Contract'] = df['BidLvl']+df['BidSuit']+df['Dbl']+' '+df['Declarer_Direction']
    df['BidLvl'] = df['BidLvl'].astype('UInt8') # using UInt8 instead of uint8 because of NaNs
    assert (df['Contract'].eq('PASS')|df['BidLvl'].notna()).all()
    assert (df['Contract'].eq('PASS')|df['BidLvl'].between(1,7,inclusive='both')).all()
    assert (df['Contract'].eq('PASS')|df['BidSuit'].notna()).all()
    assert (df['Contract'].eq('PASS')|df['BidSuit'].isin(list('CDHSN'))).all()
    assert (df['Contract'].eq('PASS')|df['Dbl'].notna()).all()
    assert (df['Contract'].eq('PASS')|df['Dbl'].isin(['','X','XX'])).all()

    assert df['table_number'].isna().all() or df['table_number'].ge(1).all() # some events have NaN table_numbers.
 
    # create more useful Vul column
    df['Vul'] = df['Board'].map(mlBridgeLib.BoardNumberToVul) # 0 to 3

    if not pd.api.types.is_numeric_dtype(df['Score_NS']):
        df['Score_NS'] = df['Score_NS'].astype('string') # make sure all elements are a string
        df.loc[df['Score_NS'].eq('PASS'),'Score_NS'] = '0'
        assert df['Score_NS'].ne('PASS').all()
        drop_rows = ~df['Score_NS'].map(lambda c: c[c[0] == '-':].isnumeric()) | ~df['Score_NS'].map(lambda c: c[c[0] == '-':].isnumeric())
        df.drop(df[drop_rows].index,inplace=True)
        assert df['Score_NS'].isna().sum() == 0
        assert df['Score_NS'].isna().sum() == 0
    df['Score_NS'] = df['Score_NS'].astype('int16')
    df['Score_EW'] = -df['Score_NS']

    # tournaments do not have Tricks or Result columns. Create them.
    df['scores_l'] = mlBridgeLib.ContractToScores(df)
    if 'Result' in df:
        assert df['Result'].notna().all() and df['Result'].notnull().all()
        df['Result'] = df['Result'].map(lambda x: 0 if x in ['=','0',''] else int(x[1:]) if x[0]=='+' else int(x)).astype('int8') # 0 for PASS
    else:
        df['Result'] = df.apply(lambda r: pd.NA if  r['Score_NS'] not in r['scores_l'] else r['scores_l'].index(r['Score_NS'])-(r['BidLvl']+6),axis='columns').astype('Int8') # pd.NA is due to director's adjustment
    if df['Result'].isna().any():
        print('NaN Results:\n',df[df['Result'].isna()][['Board','Contract','BidLvl','BidSuit','Dbl','Declarer_Direction','Score_NS','Score_EW','Result','scores_l']])
    assert df['Result'].map(lambda x: x is pd.NA or -13 <= x <= 13).all()

    if 'Tricks' in df and df['Tricks'].notnull().all(): # tournaments have a Trick column with all None(?).
        assert df['Tricks'].notnull().all()
        df.loc[df['Contract'].eq('PASS'),'Tricks'] = pd.NA
    else:
        df['Tricks'] = df.apply(lambda r: pd.NA if r['BidLvl'] is pd.NA or r['Result'] is pd.NA else r['BidLvl']+6+r['Result'],axis='columns') # pd.NA is needed for PASS
    if df['Tricks'].isna().any():
        print('NaN Tricks:\n',df[df['Tricks'].isna()][['Board','Contract','BidLvl','BidSuit','Dbl','Declarer_Direction','Score_NS','Score_EW','Tricks','Result','scores_l']])
    df['Tricks'] = df['Tricks'].astype('UInt8')
    assert df['Tricks'].map(lambda x: x is pd.NA or 0 <= x <= 13).all()

    # drop invalid round numbers
    if df['Round'].notnull().any():
        drop_rows = df['Round'].isna()
        df.drop(df[drop_rows].index,inplace=True)

    df.drop(['scores_l'],axis='columns',inplace=True)

    for col in df.columns:
        assert not (col.startswith('ns_') or col.startswith('ew_') or col.startswith('NS_') or col.startswith('EW_') or col.endswith('_ns') or col.endswith('_ew')), col

    assert len(df) > 0
    return df.reset_index(drop=True)


# todo: use Augment_Metric_By_Suits or TuplesToSuits?
def Augment_Metric_By_Suits(metrics,metric,dtype='uint8'):
    for d,direction in enumerate(mlBridgeLib.NESW):
        for s,suit in  enumerate(mlBridgeLib.SHDC):
            metrics['_'.join([metric,direction])] = metrics[metric].map(lambda x: x[1][d][0]).astype(dtype)
            metrics['_'.join([metric,direction,suit])] = metrics[metric].map(lambda x: x[1][d][1][s]).astype(dtype)
    for direction in mlBridgeLib.NS_EW:
        metrics['_'.join([metric,direction])] = metrics['_'.join([metric,direction[0]])]+metrics['_'.join([metric,direction[1]])].astype(dtype)
        for s,suit in  enumerate(mlBridgeLib.SHDC):
            metrics['_'.join([metric,direction,suit])] = metrics['_'.join([metric,direction[0],suit])]+metrics['_'.join([metric,direction[1],suit])].astype(dtype)


def TuplesToSuits(df,tuples,column,excludes=[]):
    d = {}
    d['_'.join([column])] = tuples.map(lambda x: x[0])
    for i,direction in enumerate('NESW'):
        d['_'.join([column,direction])] = tuples.map(lambda x: x[1][i][0])
        for j,suit in enumerate('SHDC'):
            d['_'.join([column,direction,suit])] = tuples.map(lambda x: x[1][i][1][j])
    for i,direction in enumerate(['NS','EW']):
        d['_'.join([column,direction])] = tuples.map(lambda x: x[1][i][0]+x[1][i+2][0])
        for j,suit in enumerate('SHDC'):
            d['_'.join([column,direction,suit])] = tuples.map(lambda x: x[1][i][1][j]+x[1][i+2][1][j])
    for k,v in d.items():
        if k not in excludes:
            df[k] = v
    return d


def augment_df(df,sd_cache_d):

    # positions
    df['Pair_Declarer_Direction'] = df['Declarer_Direction'].map(mlBridgeLib.PlayerDirectionToPairDirection)
    df['Opponent_Pair_Direction'] = df['Pair_Declarer_Direction'].map(mlBridgeLib.PairDirectionToOpponentPairDirection)
    df['Direction_OnLead'] = df['Declarer_Direction'].map(mlBridgeLib.NextPosition)
    df['Direction_Dummy'] = df['Direction_OnLead'].map(mlBridgeLib.NextPosition)
    df['Direction_NotOnLead'] = df['Direction_Dummy'].map(mlBridgeLib.NextPosition)

    # hands
    df['hands'] = df['board_record_string'].map(mlBridgeLib.brs_to_hands)
    # ouch. Sometimes acbl hands use '-' in board_record_string, sometimes they don't. Are online hands without '-' and club f-f with '-'? Removing '-' in both so compare works.
    assert df['hands'].map(mlBridgeLib.hands_to_brs).eq(df['board_record_string'].str.replace('-','').str.replace('T','10')).all(), df[df['hands'].map(mlBridgeLib.hands_to_brs).ne(df['board_record_string'])][['Board','board_record_string','hands']]
    df['PBN'] = df['hands'].map(mlBridgeLib.HandToPBN)
    assert df['PBN'].map(mlBridgeLib.pbn_to_hands).eq(df['hands']).all(), df[df['PBN'].map(mlBridgeLib.pbn_to_hands).ne(df['hands'])]
    brs = df['PBN'].map(mlBridgeLib.pbn_to_brs)
    assert brs.map(mlBridgeLib.brs_to_pbn).eq(df['PBN']).all(), df[brs.map(mlBridgeLib.brs_to_pbn).ne(df['PBN'])]

    # hand evaluation metrics
    # todo: use Augment_Metric_By_Suits or TuplesToSuits?
    # 'hands' is ordered CDHS
    hcp = df['hands'].map(mlBridgeLib.HandsToHCP)
    TuplesToSuits(df,hcp,'HCP',['HCP'])
    qt = df['hands'].map(mlBridgeLib.HandsToQT)
    TuplesToSuits(df,qt,'QT',['QT'])
    dp = df['hands'].map(mlBridgeLib.HandsToDistributionPoints)
    TuplesToSuits(df,dp,'DP',['DP'])
    sl = df['hands'].map(mlBridgeLib.HandsToSuitLengths) # sl is needed later by LoTT
    TuplesToSuits(df,sl,'SL',['SL','SL_N','SL_E','SL_S','SL_W','SL_NS','SL_EW'])
    so = mlBridgeLib.CDHS
    for d in mlBridgeLib.NESW:
        df[f'SL_{d}_{so}'] = df.filter(regex=f'^SL_{d}_[{so}]$').values.tolist() # ordered from clubs to spades [CDHS]
        df[f'SL_{d}_{so}_J'] = df[f'SL_{d}_{so}'].map(lambda l:'-'.join([str(v) for v in l])).astype('string') # joined CDHS into string
        df[f'SL_{d}_ML_S'] = df[f'SL_{d}_{so}'].map(lambda l: [v for v,n in sorted([(ll,n) for n,ll in enumerate(l)],key=lambda k:(-k[0],k[1]))]) # ordered most-to-least
        df[f'SL_{d}_ML_SI'] = df[f'SL_{d}_{so}'].map(lambda l: [n for v,n in sorted([(ll,n) for n,ll in enumerate(l)],key=lambda k:(-k[0],k[1]))]) # ordered most-to-least containing indexes
        df[f'SL_{d}_ML_SJ'] = df[f'SL_{d}_ML_S'].map(lambda l:'-'.join([str(v) for v in l])).astype('string') # ordered most-to-least and joined into string

    # Create columns containing column names of the NS,EW longest suit.
    sl_cols = [('_'.join(['SL_Max',d]),['_'.join(['SL',d,s]) for s in mlBridgeLib.SHDC]) for d in mlBridgeLib.NS_EW]
    for d in sl_cols:
        df[d[0]] = df[d[1]].idxmax(axis=1).astype('category') # defaults to object so need string or category

    df = mlBridgeLib.append_double_dummy_results(df)

    # LoTT
    ddmakes = df.apply(lambda r: tuple([tuple([r['_'.join(['DD',d,s])] for s in 'CDHSN']) for d in 'NESW']),axis='columns')
    LoTT_l = [mlBridgeLib.LoTT_SHDC(t,l) for t,l in zip(ddmakes,sl)] # [mlBridgeLib.LoTT_SHDC(ddmakes[i],sl[i]) for i in range(len(df))]
    df['LoTT_Tricks'] = [t for t,l,v in LoTT_l]
    df['LoTT_Suit_Length'] = [l for t,l,v in LoTT_l] # todo: is this correct? use SL_Max_(NS|EW) instead? verify LoTT_Suit_Length against SL_Max_{declarer_pair_direction}.
    df['LoTT_Variance'] = [v for t,l,v in LoTT_l]
    del LoTT_l
    df = df.astype({'LoTT_Tricks':'uint8','LoTT_Suit_Length':'uint8','LoTT_Variance':'int8'})

    # ContractType
    df['ContractType'] = df.apply(lambda r: 'PASS' if r['BidLvl'] is pd.NA else mlBridgeLib.ContractType(r['BidLvl']+6,r['BidSuit']),axis='columns').astype('category')
    # Create column of contract types by partnership by suit. e.g. CT_NS_C.
    contract_types_d = mlBridgeLib.CategorifyContractType(ddmakes)
    contract_types_df = pd.DataFrame(contract_types_d,dtype='category')
    assert len(df) == len(contract_types_df)
    df = pd.concat([df,contract_types_df],axis='columns') # ,join='inner')
    del contract_types_df,contract_types_d

    # create dict of NS matchpoint data.
    matchpoint_ns_d = {} # key is board. values are matchpoint details (score, beats, ties, matchpoints, pct).
    for board,g in df.groupby('Board'):
        board_mps_ns = {}
        for score_ns in g['Score_NS']:
            board_mps_ns = mlBridgeLib.MatchPointScoreUpdate(score_ns,board_mps_ns) # convert to float32 here? It's still a string because it might originally have AVG+ or AVG- etc.
        matchpoint_ns_d[board] = board_mps_ns
    # validate boards are scored correctly
    for board,g in df.groupby('Board'):
        for score_ns,match_points_ns in zip(g['Score_NS'],g['MatchPoints_NS'].astype('float32')):
            if matchpoint_ns_d[board][score_ns][3] != match_points_ns: # match_points_ns is a string because it might originally have AVG+ or AVG- etc.
                print(f'Board {board} score {matchpoint_ns_d[board][score_ns][3]} tuple {matchpoint_ns_d[board][score_ns]} does not match matchpoint score {match_points_ns}') # ok if off by epsilon

    # Vul columns
    df['Vul_NS'] = (df['Vul']&1).astype('bool')
    df['Vul_EW'] = (df['Vul']&2).astype('bool')

    # board result columns
    df['OverTricks'] = df['Result'].gt(0)
    df['JustMade'] = df['Result'].eq(0)
    df['UnderTricks'] = df['Result'].lt(0)

    df[f"Vul_Declarer"] = df.apply(lambda r: r['Vul_'+r['Pair_Declarer_Direction']], axis='columns')
    df['Pct_Declarer'] = df.apply(lambda r: r['Pct_'+r['Pair_Declarer_Direction']], axis='columns')
    df['Pair_Number_Declarer'] = df.apply(lambda r: r['Pair_Number_'+r['Pair_Declarer_Direction']], axis='columns')
    df['Pair_Number_Defender'] = df.apply(lambda r: r['Pair_Number_'+r['Opponent_Pair_Direction']], axis='columns')
    df['Number_Declarer'] = df.apply(lambda r: r['Player_Number_'+r['Declarer_Direction']], axis='columns') # todo: keep as lower case?
    df['Name_Declarer'] = df.apply(lambda r: r['Player_Name_'+r['Declarer_Direction']], axis='columns')
    # todo: drop either Tricks or Tricks_Declarer as they are invariant and duplicates
    df['Tricks_Declarer'] = df['Tricks'] # synonym for Tricks
    df['Score_Declarer'] = df.apply(lambda r: r['Score_'+r['Pair_Declarer_Direction']], axis='columns')
    df['MPs_Declarer'] = df.apply(lambda r: r['MatchPoints_'+r['Pair_Declarer_Direction']], axis='columns')

    df['DDTricks'] = df.apply(lambda r: pd.NA if r['BidLvl'] is pd.NA else r['_'.join(['DD',r['Declarer_Direction'],r['BidSuit']])], axis='columns') # invariant
    df['DDTricks_Dummy'] = df.apply(lambda r: pd.NA if r['BidLvl'] is pd.NA else r['_'.join(['DD',r['Direction_Dummy'],r['BidSuit']])], axis='columns') # invariant
    # NA for NT. df['DDSLDiff'] = df.apply(lambda r: pd.NA if r['BidLvl'] is pd.NA else r['DDTricks']-r['SL_'+r['Pair_Declarer_Direction']+'_'+r['BidSuit']], axis='columns') # pd.NA or zero?
    df['DDScore_NS'] = df.apply(lambda r: 0 if r['BidLvl'] is pd.NA else mlBridgeLib.score(r['BidLvl']-1, 'CDHSN'.index(r['BidSuit']), len(r['Dbl']), ('NSEW').index(r['Declarer_Direction']), mlBridgeLib.DirectionSymToVulBool(r['Vul_Declarer'],r['Declarer_Direction']), r['DDTricks']-r['BidLvl']-6), axis='columns')
    df['DDScore_EW'] = -df['DDScore_NS']
    df['DDMPs_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['DDScore_NS'],matchpoint_ns_d[r['Board']])[r['DDScore_NS']][3],axis='columns')
    df['DDMPs_EW'] = df['Board_Top']-df['DDMPs_NS']
    df['DDPct_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['DDScore_NS'],matchpoint_ns_d[r['Board']])[r['DDScore_NS']][4],axis='columns')
    df['DDPct_EW'] = 1-df['DDPct_NS']

    # Declarer ParScore columns
    # ACBL online games have no par score data. Must create it.
    if 'par' not in df or df['par'].eq('').all():
        df.rename({'ParScore_EndPlay_NS':'ParScore_NS','ParScore_EndPlay_EW':'ParScore_EW','ParContracts_EndPlay':'ParContracts'},axis='columns',inplace=True)
        #df['ParScore_NS'] = df['ParScore_EndPlay_NS']
        #df['ParScore_EW'] = df['ParScore_EndPlay_EW']
        #df['ParContracts'] = df['ParContracts_EndPlay']
        #df.drop(['ParScore_EndPlay_NS','ParScore_EndPlay_EW','ParContracts_EndPlay'],axis='columns',inplace=True)
    else:
        # parse par column and create ParScore column.
        df['ParScore_NS'] = df['par'].map(lambda x: x.split(' ')[1]).astype('int16')
        df['ParScore_EW'] = -df['ParScore_NS']
        df['ParContracts'] = df['par'].map(lambda x: x.split(' ')[2:]).astype('string')
    if 'par' in df:
        df.drop(['par'],axis='columns',inplace=True)
    df['ParScore_MPs_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['ParScore_NS'],matchpoint_ns_d[r['Board']])[r['ParScore_NS']][3],axis='columns')
    df['ParScore_MPs_EW'] = df['Board_Top']-df['ParScore_MPs_NS']
    df['ParScore_Pct_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['ParScore_NS'],matchpoint_ns_d[r['Board']])[r['ParScore_NS']][4],axis='columns')
    df['ParScore_Pct_EW'] = 1-df['ParScore_Pct_NS']
    #df["ParScore_Declarer"] = df.apply(lambda r: r['ParScore_'+r['Pair_Declarer_Direction']], axis='columns')
    #df["ParScore_MPs_Declarer"] = df.apply(lambda r: r['ParScore_MPs_'+r['Pair_Declarer_Direction']], axis='columns')
    #df["ParScore_Pct_Declarer"] = df.apply(lambda r: r['ParScore_Pct_'+r['Pair_Declarer_Direction']], axis='columns')
    #df['ParScore_Diff_Declarer'] = df['Score_Declarer']-df['ParScore_Declarer'] # adding convenience column to df. Actual Par Score vs DD Score
    #df['ParScore_MPs_Diff_Declarer'] = df['MPs_Declarer'].astype('float32')-df['ParScore_MPs'] # forcing MPs_Declarer to float32. It is still string because it might originally have AVG+ or AVG- etc.
    #df['ParScore_Pct_Diff_Declarer'] = df['Pct_Declarer']-df['ParScore_Pct_Declarer']
    #df['Tricks_DD_Diff_Declarer'] = df['Tricks_Declarer']-df['DDTricks] # adding convenience column to df. Actual Tricks vs DD Tricks
    #df['Score_DD_Diff_Declarer'] = df['Score_Declarer']-df['DD_Score_Declarer'] # adding convenience column to df. Actual Score vs DD Score

    # masterpoints columns
    for d in mlBridgeLib.NESW:
        df['mp_total_'+d.lower()] = df['mp_total_'+d.lower()].astype('float32')
    # todo: use 'mp_total_*' (downloaded) instead of 'MP_*' (acbl_*_board_results)?
    df['MP_Sum_NS'] = df['mp_total_n']+df['mp_total_s']
    df['MP_Sum_EW'] = df['mp_total_e']+df['mp_total_w']
    df['MP_Geo_NS'] = df['mp_total_n']*df['mp_total_s']
    df['MP_Geo_EW'] = df['mp_total_e']*df['mp_total_w']

    df, sd_cache_d = Augment_Single_Dummy(df,sd_cache_d,10,matchpoint_ns_d) # {} is no cache

    # todo: check dtypes
    # df = df.astype({'Name_Declarer':'string','Score_Declarer':'int16','ParScore_Declarer':'int16','Pct_Declarer':'float32','DDTricks':'uint8','DD_Score_Declarer':'int16','DD_Pct_Declarer':'float32','Tricks_DD_Diff_Declarer':'int8','Score_DD_Diff_Declarer':'int16','ParScore_DD_Diff_Declarer':'int16','ParScore_Pct_Declarer':'float32','Pair_Declarer':'string','Pair_Defender':'string'})

    # todo: verify every dtype is correct.
    # todo: rename columns when there's a better name
    df.rename({'dealer':'Dealer'},axis='columns',inplace=True)
    df['Dealer'] = df['Dealer'].astype('string')
    df['Vul'] = df['Vul'].astype('string')
    
    return df, sd_cache_d, matchpoint_ns_d


def Augment_Single_Dummy(df,sd_cache_d,produce,matchpoint_ns_d):

    sd_cache_d = mlBridgeLib.append_single_dummy_results(df['PBN'],sd_cache_d,produce)
    df['SDProbs'] = df.apply(lambda r: sd_cache_d[r['PBN']].get(tuple([r['Pair_Declarer_Direction'],r['Declarer_Direction'],r['BidSuit']]),[0]*14),axis='columns') # had to use get(tuple([...]))
    df['SDScores'] = df.apply(Create_SD_Scores,axis='columns')
    df['SDScore_NS'] = df.apply(Create_SD_Score,axis='columns').astype('int16') # Declarer's direction
    df['SDScore_EW'] = -df['SDScore_NS']
    df['SDMPs_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['SDScore_NS'],matchpoint_ns_d[r['Board']])[r['SDScore_NS']][3],axis='columns')
    df['SDMPs_EW'] = (df['Board_Top']-df['SDMPs_NS']).astype('float32')
    df['SDPct_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['SDScore_NS'],matchpoint_ns_d[r['Board']])[r['SDScore_NS']][4],axis='columns')
    df['SDPct_EW'] = (1-df['SDPct_NS']).astype('float32')
    max_score_contract = df.apply(Create_SD_Score_Max,axis='columns')
    df['SDScore_Max_NS'] = pd.Series([score for score,contract in max_score_contract],dtype='float32')
    df['SDScore_Max_EW'] = pd.Series([-score for score,contract in max_score_contract],dtype='float32')
    df['SDContract_Max'] = pd.Series([contract for score,contract in max_score_contract],dtype='string') # invariant
    del max_score_contract
    df['SDMPs_Max_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['SDScore_Max_NS'],matchpoint_ns_d[r['Board']])[r['SDScore_Max_NS']][3],axis='columns')
    df['SDMPs_Max_EW'] = (df['Board_Top']-df['SDMPs_Max_NS']).astype('float32')
    df['SDPct_Max_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['SDScore_Max_NS'],matchpoint_ns_d[r['Board']])[r['SDScore_Max_NS']][4],axis='columns')
    df['SDPct_Max_EW'] = (1-df['SDPct_Max_NS']).astype('float32')
    df['SDScore_Diff_NS'] = (df['Score_NS']-df['SDScore_NS']).astype('int16')
    df['SDScore_Diff_EW'] = (df['Score_EW']-df['SDScore_EW']).astype('int16')
    df['SDScore_Max_Diff_NS'] = (df['Score_NS']-df['SDScore_Max_NS']).astype('int16')
    df['SDScore_Max_Diff_EW'] = (df['Score_EW']-df['SDScore_Max_EW']).astype('int16')
    df['SDPct_Diff_NS'] = (df['Pct_NS']-df['SDPct_NS']).astype('float32')
    df['SDPct_Diff_EW'] = (df['Pct_EW']-df['SDPct_EW']).astype('float32')
    df['SDPct_Max_Diff_NS'] = (df['Pct_NS']-df['SDPct_Max_NS']).astype('float32')
    df['SDPct_Max_Diff_EW'] = (df['Pct_EW']-df['SDPct_Max_EW']).astype('float32')
    df['SDParScore_Pct_Diff_NS'] = (df['ParScore_Pct_NS']-df['SDPct_Diff_NS']).astype('float32')
    df['SDParScore_Pct_Diff_EW'] = (df['ParScore_Pct_EW']-df['SDPct_Diff_EW']).astype('float32')
    df['SDParScore_Pct_Max_Diff_NS'] = (df['ParScore_Pct_NS']-df['SDPct_Max_Diff_NS']).astype('float32')
    df['SDParScore_Pct_Max_Diff_EW'] = (df['ParScore_Pct_EW']-df['SDPct_Max_Diff_EW']).astype('float32')
    # using same df to avoid the issue with creating new columns. New columns require meta data will need to be changed too.
    sd_df = pd.DataFrame(df['SDProbs'].values.tolist(),columns=[f'SDProbs_Taking_{i}' for i in range(14)])
    for c in sd_df.columns:
        df[c] = sd_df[c].astype('float32')
    return df, sd_cache_d


def Create_SD_Scores(r):
    if r['Score_Declarer']:
        level = r['BidLvl']-1
        suit = r['BidSuit']
        iCDHSN = 'CDHSN'.index(suit)
        nsew = r['Declarer_Direction']
        iNSEW = 'NSEW'.index(nsew)
        vul = mlBridgeLib.DirectionSymToVulBool(r['Vul_Declarer'],nsew)
        double = len(r['Dbl'])
        scores_l = mlBridgeLib.ScoreDoubledSets(level, iCDHSN, vul, double, iNSEW)
        return scores_l
    else:
        return [0]*14


#def Create_SD_Probs(r):
#    return [r['SDProb_Take_'+str(n)] for n in range(14)] # todo: this was previously computed. can we just use that?


def Create_SD_Score(r):
    probs = r['SDProbs']
    scores_l = r['SDScores']
    ps = sum(prob*score for prob,score in zip(probs,scores_l))
    return ps if r['Declarer_Direction'] in 'NS' else -ps


# Highest expected score, same suit, any level
def Create_SD_Score_Max(r):
    score_max = None
    if r['Score_Declarer']:
        suit = r['BidSuit']
        iCDHSN = 'CDHSN'.index(suit)
        nsew = r['Declarer_Direction']
        iNSEW = 'NSEW'.index(nsew)
        vul = mlBridgeLib.DirectionSymToVulBool(r['Vul_Declarer'],nsew)
        double = len(r['Dbl'])
        probs = r['SDProbs']
        for level in range(7):
            scores_l = mlBridgeLib.ScoreDoubledSets(level, iCDHSN, vul, double, iNSEW)
            score = sum(prob*score for prob,score in zip(probs,scores_l))
            # todo: do same for redoubled? or is that too rare to matter?
            #scoresx_l = mlBridgeLib.ScoreDoubledSets(level, iCDHSN, vul, 1, iNSEW)
            #scorex = sum(prob*score for prob,score in zip(probs,scoresx_l))
            isdoubled = double
            #if scorex > score:
            #    score = scorex
            #    if isdoubled == 0:
            #        isdoubled = 1
            # must be mindful that NS makes positive scores but EW makes negative scores.
            if nsew in 'NS' and (score_max is None or score > score_max):
                score_max = score
                contract_max = str(level+1)+suit+['','X','XX'][isdoubled]+' '+nsew
            elif nsew in 'EW' and (score_max is None or score < score_max):
                score_max = score
                contract_max = str(level+1)+suit+['','X','XX'][isdoubled]+' '+nsew
    else:
        score_max = 0
        contract_max = 'PASS'
    return (score_max, contract_max)
