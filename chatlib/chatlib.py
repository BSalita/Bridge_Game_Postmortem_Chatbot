import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # or DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def print_to_log_info(*args):
    print_to_log(logging.INFO, *args)
def print_to_log_debug(*args):
    print_to_log(logging.DEBUG, *args)
def print_to_log(level, *args):
    logging.log(level, ' '.join(str(arg) for arg in args))

import pandas as pd
import polars as pl
from pprint import pprint # obsolete?
import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd().parent.parent.joinpath('mlBridgeLib'))) # removed .parent
sys.path.append(str(pathlib.Path.cwd().parent.parent.joinpath('acbllib'))) # removed .parent
sys.path
from mlBridgeLib.mlBridgeLib import pd_options_display, Direction_to_NESW_d, brs_to_pbn, Vulnerability_to_Vul_d, vul_sym_to_index_d, BoardNumberToVul, ContractToScores
import mlBridgeLib.mlBridgeLib as mlBridgeLib
# import acbllib


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
#         print_to_log(
#             colored(
#                 formatted_message,
#                 role_to_color[messages[formatted_messages.index(
#                     formatted_message)]["role"]],
#             )
#         )


# merge acbl json dicts into logically related dicts. dicts will be used to create dfs
# def json_dict_to_df(d,kl,jdl):
#     print_to_log_debug(kl)
#     dd = {}
#     d[kl] = dd
#     assert not isinstance(jdl,dict)
#     for i,jd in enumerate(jdl):
#         for k,v in jd.items():
#             kkl = kl+(k,i)
#             print_to_log_debug(i,kl,k,kkl)
#             #time.sleep(.00001)
#             if isinstance(v,list):
#                 print_to_log_debug('\n',type(v),kkl,v)
#                 json_dict_to_df(d,kkl,v)
#                 print_to_log_debug('list:',kkl,len(d[kkl]))
#             elif isinstance(v,dict):
#                 #kkl = kl+(k,)
#                 print_to_log_debug('\n',type(v),kkl,v)
#                 json_dict_to_df(d,kkl,[v])
#                 print_to_log_debug('dict:',kkl,len(d[kkl]))
#             else:
#                 if k not in dd:
#                     dd[k] = []
#                 dd[k].append(v)
#             #assert k != 'points',[kl,k,type(v),v]
#     return d


# todo: obsolete?
# todo: if dtype isnumeric() downcast to minimal size. Some columns may have dtype of int64 because of sql declaration ('Board').
# def convert_to_best_dtypex(k,v):
#     vv = v.convert_dtypes(infer_objects=True)
#     vvv = vv.copy()
#     for col in vv.columns:
#         print_to_log_debug(col,vvv[col].dtype)
#         # todo: special cases. maybe should be done before writing to acbl_club_results.sqlite?
#         if col in ['ns_score','ew_score']:
#             vvv[col] = vvv[col].replace('PASS','0')
#         elif col == 'result':
#             vvv[col] = vvv[col].replace('+','0').replace('=','0').replace('','0') # don't use .str. and all 3 are needed.
#         elif col == 'tricks_taken':
#             vvv[col] = vvv[col].replace('','0')
#         if vvv[col].dtype == 'string' and vvv[col].notna().all() and vvv[col].ne('').all():
#             print_to_log_debug(f"String: {col}")
#             try:
#                 if vvv[col].str.contains('.',regex=False).any():
#                     print_to_log_debug(f"Trying to convert {col} to float")
#                     converted_values = pd.to_numeric(vvv[col], downcast='float', errors='raise')
#                 elif vvv[col].str.contains('-',regex=False).any():
#                     print_to_log_debug(f"Trying to convert {col} to integer")
#                     converted_values = pd.to_numeric(vvv[col], downcast='integer', errors='raise')
#                 else:
#                     print_to_log_debug(f"Trying to convert {col} to unsigned")
#                     converted_values = pd.to_numeric(vvv[col], downcast='unsigned', errors='raise')
#                 vvv[col] = converted_values
#                 print_to_log_debug(f"Converted {col} to {vvv[col].dtype}")
#             except ValueError:
#                 print_to_log(logging.WARNING, f"Can't convert {col} to float. Keeping as string")
#     print_to_log_debug(f"dfs_dtype_conversions['{k}'] = "+'{')
#     for col in vvv.columns:
#         print_to_log_debug(f"    '{col}':'{v[col].dtype},{vv[col].dtype},{vvv[col].dtype}',")
#     print_to_log_debug("}\n")
#     return vvv


# g_all_functions_in_module = {n:f for n,f in inspect.getmembers(sys.modules[__name__], inspect.isfunction)}

# def json_dict_to_types(json_dict,root_name,path):
#     dfs = {}
#     root = []
#     df = pd.json_normalize(json_dict,path,max_level=0)
#     for k,v in df.items():
#         if isinstance(v,dict):
#             assert k not in dfs, k
#             dfs[k] = pd.DataFrame(v)
#             if all(isinstance(kk,int) or (isinstance(kk,str) and kk.isnumeric()) for kk,vv in v.items()): # dict but with list like indices
#                 dfs[k] = dfs[k].T
#         elif isinstance(v,list):
#             assert k not in dfs, k
#             dfs[k] = pd.DataFrame(v)
#         else:
#             root.append({k:v})
#     assert k not in dfs, k
#     dfs[root_name] = pd.DataFrame(root)
#     return dfs


# todo: finish converting from pandas to polars. hitch is that pd.json_normalize() is pandas only.
def create_club_dfs(data):
    dfs = {}
    event_df = pd.json_normalize(data,max_level=0) # todo: convert to polars
    dfs['event'] = pl.from_pandas(event_df)
    for k,v in event_df.items():
        if isinstance(v[0],dict) or isinstance(v[0],list):
            assert k not in dfs, k
            df = pd.json_normalize(data,max_level=0)[k] # todo: convert to polars
            # must test whether df is all scalers. Very difficult to debug.
            if isinstance(v[0],dict) and not any([isinstance(vv,dict) or isinstance(vv,list) for kk,vv in df[0].items()]):
                dfs[k] = pl.from_pandas(pd.DataFrame.from_records(df[0],index=[0])) # must use from_records to avoid 'all values are scaler must specify index' error
            else:
                dfs[k] = pl.from_pandas(pd.DataFrame.from_records(df[0]).astype('string')) # warning: needed astype('string') to avoid int error
            event_df.drop(columns=[k],inplace=True)
            #if all(isinstance(kk,int) or (isinstance(kk,str) and kk.isnumeric()) for kk,vv in v.items()):
    dfs['hand_records'] = pl.from_pandas(pd.json_normalize(data,['sessions','hand_records']))
    dfs['strat_place'] = pl.from_pandas(pd.json_normalize(data,['sessions','sections','pair_summaries','strat_place']))
    dfs['sections'] = pl.from_pandas(pd.json_normalize(data,['sessions','sections']))
    dfs['boards'] = pl.from_pandas(pd.json_normalize(data,['sessions','sections','boards']))
    dfs['pair_summaries'] = pl.from_pandas(pd.json_normalize(data,['sessions','sections','pair_summaries']))
    dfs['players'] = pl.from_pandas(pd.json_normalize(data,['sessions','sections','pair_summaries','players']))
    dfs['board_results'] = pl.from_pandas(pd.json_normalize(data,['sessions','sections','boards','board_results']))
    return dfs


# def create_tournament_dfs(data):
#     return create_club_dfs(data)



def merge_clean_augment_club_dfs(dfs, sd_cache_d, acbl_number):  # todo: acbl_number obsolete?

    print_to_log_info('merge_clean_augment_club_dfs: dfs keys:', dfs.keys())

    df_brs = dfs['board_results']
    print_to_log_info(df_brs.head(1))
    #assert not df_brs.columns.contains(r'^.*_[xy]$').any()

    df_b = dfs['boards'].rename({'id': 'board_id'}).select(['board_id', 'section_id', 'board_number'])
    print_to_log_info(df_b.head(1))
    #assert not df_b.columns.str.contains(r'^.*_[xy]$').any()

    df_br_b = df_brs.join(df_b, on='board_id', how='left')
    print_to_log_info(df_br_b.head(1))
    assert df_br_b.height == df_brs.height
    #assert not df_br_b.columns.str.contains(r'^.*_[xy]$').any()

    df_sections = dfs['sections'].rename({'id': 'section_id', 'name': 'section_name'}).drop(['created_at', 'updated_at', 'transaction_date', 'pair_summaries', 'boards'])
    print_to_log_info(df_sections.head(1))

    df_br_b_sections = df_br_b.join(df_sections, on='section_id', how='left')
    print_to_log_info(df_br_b_sections.head(1))
    assert df_br_b_sections.height == df_br_b.height
    #assert not df_br_b_sections.columns.str.contains(r'^.*_[xy]$').any()

    df_sessions = dfs['sessions'].rename({'id': 'session_id', 'number': 'session_number'}).drop(['created_at', 'updated_at', 'transaction_date', 'hand_records', 'sections'])
    print_to_log_info(df_sessions.head(1))

    df_sessions = df_sessions.with_columns(pl.col("session_id").cast(pl.Int64)) # todo: hack to fix hack in pandas to polars conversion error
    df_br_b_sections_sessions = df_br_b_sections.join(df_sessions, on='session_id', how='left')
    print_to_log_info(df_br_b_sections_sessions.head(1))
    assert df_br_b_sections_sessions.height == df_br_b_sections.height
    #assert not df_br_b_sections_sessions.columns.str.contains(r'^.*_[xy]$').any()

    df_clubs = dfs['club'].rename({'id': 'event_id', 'name': 'club_name', 'type': 'club_type'}).drop(['created_at', 'updated_at', 'transaction_date'])
    print_to_log_info(df_clubs.head(1))

    df_br_b_sections_sessions = df_br_b_sections_sessions.with_columns(pl.col("event_id").cast(pl.Int64)) # todo: hack to fix hack in pandas to polars conversion error
    df_br_b_sections_sessions_clubs = df_br_b_sections_sessions.join(df_clubs, on='event_id', how='left')
    print_to_log_info(df_br_b_sections_sessions_clubs.head(1))
    assert df_br_b_sections_sessions_clubs.height == df_br_b_sections_sessions.height
    #assert not df_sections.columns.str.contains(r'^.*_[xy]$').any()

    df_events = dfs['event'].rename({'id': 'event_id', 'club_name': 'event_club_name', 'type': 'event_type'}).drop(['created_at', 'updated_at', 'transaction_date', 'deleted_at'])
    print_to_log_info(df_events.head(1))

    df_br_b_sections_sessions_events = df_br_b_sections_sessions_clubs.join(df_events, on='event_id', how='left')
    print_to_log_info(df_br_b_sections_sessions_events.head(1))
    assert df_br_b_sections_sessions_events.height == df_br_b_sections_sessions_clubs.height
    #assert not df_br_b_sections_sessions_events.columns.str.contains(r'^.*_[xy]$').any()

    df_pair_summaries = dfs['pair_summaries'].rename({'id': 'pair_summary_id'}).drop(['created_at', 'updated_at', 'transaction_date'])
    print_to_log_info(df_pair_summaries.head(1))

    df_br_b_pair_summary_ns = df_pair_summaries.filter(pl.col('direction') == 'NS').with_columns(pl.col('pair_number').alias('ns_pair'), pl.col('section_id').alias('section_id'))
    #assert not df_br_b_pair_summary_ns.columns.str.contains(r'^.*_[xy]$').any()
    df_br_b_pair_summary_ew = df_pair_summaries.filter(pl.col('direction') == 'EW').with_columns(pl.col('pair_number').alias('ew_pair'), pl.col('section_id').alias('section_id'))
    #assert not df_br_b_pair_summary_ew.columns.str.contains(r'^.*_[xy]$').any()

    df_players = dfs['players'].drop(['id', 'created_at', 'updated_at', 'transaction_date']).rename({'id_number': 'player_number', 'name': 'player_name'})
    player_n = df_players.group_by('pair_summary_id').agg(pl.first('player_number').alias('player_number_n'), pl.first('player_name').alias('player_name_n'))
    player_s = df_players.group_by('pair_summary_id').agg(pl.last('player_number').alias('player_number_s'), pl.last('player_name').alias('player_name_s'))
    player_e = df_players.group_by('pair_summary_id').agg(pl.first('player_number').alias('player_number_e'), pl.first('player_name').alias('player_name_e'))
    player_w = df_players.group_by('pair_summary_id').agg(pl.last('player_number').alias('player_number_w'), pl.last('player_name').alias('player_name_w'))

    player_ns = player_n.join(player_s, on='pair_summary_id', how='left')
    print_to_log_info(player_ns.head(1))
    assert player_ns.height == player_n.height
    #assert not player_ns.columns.str.contains(r'^.*_[xy]$').any()
    player_ew = player_e.join(player_w, on='pair_summary_id', how='left')
    print_to_log_info(player_ew.head(1))
    assert player_ew.height == player_e.height
    #assert not player_ew.columns.str.contains(r'^.*_[xy]$').any()

    df_pair_summary_players_ns = df_br_b_pair_summary_ns.join(player_ns, on='pair_summary_id', how='left')
    assert df_pair_summary_players_ns.height == df_br_b_pair_summary_ns.height
    df_pair_summary_players_ew = df_br_b_pair_summary_ew.join(player_ew, on='pair_summary_id', how='left')
    assert df_pair_summary_players_ew.height == df_br_b_pair_summary_ew.height

    df_br_b_sections_sessions_events_pair_summary_players = df_br_b_sections_sessions_events.join(df_pair_summary_players_ns, on=['section_id', 'ns_pair'], how='left')
    print_to_log_info(df_br_b_sections_sessions_events_pair_summary_players.head(1))
    assert df_br_b_sections_sessions_events_pair_summary_players.height == df_br_b_sections_sessions_events.height
    #assert not df_br_b_sections_sessions_events_pair_summary_players.columns.str.contains(r'^.*_[xy]$').any()
    df_br_b_sections_sessions_events_pair_summary_players = df_br_b_sections_sessions_events_pair_summary_players.join(df_pair_summary_players_ew, on=['section_id', 'ew_pair'], how='left')
    print_to_log_info(df_br_b_sections_sessions_events_pair_summary_players.head(1))
    assert df_br_b_sections_sessions_events_pair_summary_players.height == df_br_b_sections_sessions_events.height
    #assert not df_br_b_sections_sessions_events_pair_summary_players.columns.str.contains(r'^.*_[xy]$').any()

    df_hrs = dfs['hand_records'].rename({'hand_record_set_id': 'hand_record_id'}).drop(['points.N', 'points.E', 'points.S', 'points.W'])
    print_to_log_info(df_hrs.head(1))

    df_br_b_sections_sessions_events_pair_summary_players = df_br_b_sections_sessions_events_pair_summary_players.with_columns(pl.col("hand_record_id").cast(pl.Int64)) # todo: hack to fix hack in pandas to polars conversion error
    df_br_b_sections_sessions_events_pair_summary_players_hrs = df_br_b_sections_sessions_events_pair_summary_players.join(df_hrs.drop(['id', 'created_at', 'updated_at']), left_on=['hand_record_id', 'board_number'], right_on=['hand_record_id', 'board'], how='left')
    print_to_log_info(df_br_b_sections_sessions_events_pair_summary_players_hrs.head(1))
    assert df_br_b_sections_sessions_events_pair_summary_players_hrs.height == df_br_b_sections_sessions_events_pair_summary_players.height
    #assert not df_br_b_sections_sessions_events_pair_summary_players_hrs.columns.str.contains(r'^.*_[xy]$').any()

    df = df_br_b_sections_sessions_events_pair_summary_players_hrs
    for col in df.columns:
        print_to_log_info(f'cols: {col} {df[col].dtype}')

    df = df.drop(['id', 'created_at', 'updated_at', 'board_id', 'double_dummy_ns', 'double_dummy_ew'])

    df = df.rename({
        'board_number': 'Board',
        'club_id_number': 'Club',
        'contract': 'Contract',
        'game_date': 'Date',
        'ns_match_points': 'MP_NS',
        'ew_match_points': 'MP_EW',
        'ns_pair': 'Pair_Number_NS',
        'ew_pair': 'Pair_Number_EW',
        #'percentage_ns': 'Final_Standing_NS', # percentage_ns is not a column in the data. use percentage instead?
        #'percentage_ew': 'Final_Standing_EW', # percentage_ew is not a column in the data. use percentage instead?
        'result': 'Result',
        'round_number': 'Round',
        'ns_score': 'Score_NS',
        'ew_score': 'Score_EW',
        'session_number': 'Session',
        'table_number': 'Table',
        'tricks_taken': 'Tricks',
    })

    df = df.with_columns([
        pl.col('board_record_string').cast(pl.Utf8),
        pl.col('Date').str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        #pl.col('Final_Standing_NS').cast(pl.Float32), # Final_Standing_NS is not a column in the data. use percentage instead?
        #pl.col('Final_Standing_EW').cast(pl.Float32), # Final_Standing_EW is not a column in the data. use percentage instead?
        pl.col('hand_record_id').cast(pl.Int64),
        pl.col('Pair_Number_NS').cast(pl.UInt16),
        pl.col('Pair_Number_EW').cast(pl.UInt16),
        pl.col('pair_summary_id').alias('pair_summary_id_ns'), # todo: remove this alias after ai model is updated to use pair_summary_id
        pl.col('pair_summary_id').alias('pair_summary_id_ew'), # todo: remove this alias after ai model is updated to use pair_summary_id
    ])

    df = acbldf_to_mldf(df) # todo: temporarily convert to pandas to use augment_df until clean_validate_df is converted to polars

    return df

    #df, sd_cache_d, matchpoint_ns_d = augment_df(df, sd_cache_d)

    #return df, sd_cache_d, matchpoint_ns_d

def merge_clean_augment_tournament_dfs(dfs, json_results_d, sd_cache_d, player_id):
    # dfs_results contains: tournament[], event[], overalls[], handrecord[], sections[]
    print_to_log_info('merge_clean_augment_tournament_dfs: dfs keys:', dfs.keys())

    # tournament: ['_schedule_id', 'sanction', 'alt_sanction', 'name', 'start_date', 'end_date', 'district', 'unit', 'category', 'type', 'mp_restrictions', 'allowed_conventions', 'schedule_pdf', 'schedule_link', 'locations', 'last_updated', 'schedule_available', 'cancelled', 'contacts']
    # 'event': ['sanction', '_id', '_schedule_id', 'id', 'name', 'event_code', 'start_date', 'start_time', 'game_type', 'event_type', 'mp_limit', 'mp_color', 'mp_rating', 'is_charity', 'is_juniors', 'is_mixed', 'is_playthrough', 'is_seniors', 'is_side_game', 'is_womens', 'session_count', 'is_online', 'results_available', 'strat_count', 'strat_letters']
    # 'handrecord': ['box_number', 'board_number',
    #     'north_spades', 'north_hearts', 'north_diamonds', 'north_clubs', 'east_spades', 'east_hearts', 'east_diamonds', 'east_clubs', 'south_spades', 'south_hearts', 'south_diamonds', 'south_clubs', 'west_spades', 'west_hearts', 'west_diamonds', 'west_clubs',
    #     'double_dummy_north_south', 'double_dummy_east_west', 'double_dummy_par_score', 'dealer', 'vulnerability']
    # 'sections': ['id', 'number', 'transaction_date', 'hand_records', 'sections']
    # 'overall': ['session_id', 'mp_won', 'mp_color', 'score', 'team_number', 'pair_number', 'percentage', 'rank_strat_1', 'rank_strat_2', 'rank_strat_3', 'section', 'players']
    results_dfs_d = {}
    for k,v in json_results_d.items():
        print_to_log_info(k,type(v))
        if isinstance(v,list):
            results_dfs_d[k] = pl.DataFrame(v,strict=False) # concat the list of dicts into a single dict. 'sections' needs strict=False.
        elif isinstance(v,dict):
            results_dfs_d[k] = pl.DataFrame(v)
        else:
            results_dfs_d[k] = v

    section_label = dfs['section']
    df = results_dfs_d['sections'].filter(pl.col('section_label').eq(section_label)) # filter for the section of interest
    print_to_log_info(df.head(1))

    df = df.explode('board_results').unnest('board_results')
    print_to_log_info(df.head(1))

    # todo: no need to filter by orientation and concat. instead do everything in 1 df by filtering on orientation.
    ns_df = df.filter(pl.col('orientation') == 'N-S').drop(['orientation'])
    print_to_log_info(ns_df.head(1))
    ew_df = df.filter(pl.col('orientation') == 'E-W').drop(['orientation'])
    print_to_log_info(ew_df.head(1))
    assert ns_df.height == ew_df.height, f'ns_df.height: {ns_df.height}, ew_df.height: {ew_df.height}'
    assert ns_df.columns == ew_df.columns, f'ns_df.columns: {ns_df.columns}, ew_df.columns: {ew_df.columns}'

    identical_columns = ['session_id', 'section_label', 'movement_type', 'scoring_type', 'board_number', 'contract', 'declarer']
    print_to_log_info(identical_columns)
    ns_df = ns_df.rename({
        'match_points': 'match_points_ns',
        'percentage': 'percentage_ns',
    })
    print_to_log_info(ns_df.head(1))
    ew_df = ew_df.rename({
        'match_points': 'match_points_ew',
        'percentage': 'percentage_ew',
    })
    print_to_log_info(ew_df.head(1))

    ns_df = ns_df.with_columns(
        pl.col('pair_acbl').list.get(0).cast(pl.UInt32).alias('player_number_n'),
        pl.col('pair_acbl').list.get(1).cast(pl.UInt32).alias('player_number_s'),
        pl.col('pair_names').list.get(0).cast(pl.String).alias('player_name_n'),
        pl.col('pair_names').list.get(1).cast(pl.String).alias('player_name_s'),
        pl.col('opponent_pair_names').list.get(0).cast(pl.String).alias('opponent_pair_name_n'),
        pl.col('opponent_pair_names').list.get(1).cast(pl.String).alias('opponent_pair_name_s'),
    ).drop(['pair_acbl','pair_names','opponent_pair_names'])
    print_to_log_info(ns_df.head(1))

    ew_df = ew_df.with_columns(
        pl.col('pair_acbl').list.get(0).cast(pl.UInt32).alias('player_number_e'),
        pl.col('pair_acbl').list.get(1).cast(pl.UInt32).alias('player_number_w'),
        pl.col('pair_names').list.get(0).cast(pl.String).alias('player_name_e'),
        pl.col('pair_names').list.get(1).cast(pl.String).alias('player_name_w'),
        pl.col('opponent_pair_names').list.get(0).cast(pl.String).alias('opponent_pair_name_e'),
        pl.col('opponent_pair_names').list.get(1).cast(pl.String).alias('opponent_pair_name_w'),
    ).drop(['pair_acbl','pair_names','opponent_pair_names'])
    print_to_log_info(ew_df.head(1))

    ew_cols = [
        'player_number_e','player_number_w',
        'player_name_e','player_name_w',
        'opponent_pair_name_e','opponent_pair_name_w',
        'match_points_ew',
        'percentage_ew',
        'board_number',
        'pair_number',
        'opponent_pair_number',
        ]
    df = ns_df.join(ew_df[ew_cols],left_on=['pair_number','opponent_pair_number','board_number'],right_on=['opponent_pair_number','pair_number','board_number'],how='left')
    print_to_log_info(df.head(1))
    assert df.height == ns_df.height == ew_df.height # todo: pro tip. always assert heights after joins. it can save hours of debugging.

    # using df['section_results'].to_frame() because explode() creates a duplicate field 'session_id' unless selected.
    # section_results columns:
    # ['session_id', 'section_label', 'pair_number', 'team_number', 'orientation', 'strat', 'next_session_assignment',
    #     'next_session_qualification', 'score_cumulative', 'score_carryover', 'score_adjustment', 'percentage', 'mp_won', 'mp_color',
    #     'section_rank_strat_1', 'section_rank_strat_2', 'section_rank_strat_3',
    #     'overall_rank_strat_1', 'overall_rank_strat_2', 'overall_rank_strat_3', 'players', 'score_section']
    df_section_results = df['section_results'].to_frame().explode('section_results').unnest('section_results')

    # from mlBridgeLib. todo: convert to polars
    def hrs_to_brss(hrs,void='',ten='10'):
        cols = [d+'_'+s for d in ['north','west','east','south'] for s in ['spades','hearts','diamonds','clubs']] # remake of hands below, comments says the order needs to be NWES?????
        return hrs[cols].apply(lambda r: ''.join(['SHDC'[i%4]+c for i,c in enumerate(r.values)]).replace(' ','').replace('-',void).replace('10',ten), axis='columns')

    df_handrecord = results_dfs_d['handrecord']
    df_handrecord = df_handrecord.with_columns(
        pl.Series('board_record_string',mlBridgeLib.hrs_to_brss(df_handrecord.to_pandas()),pl.Utf8), # todo: eliminate polars to pandas conversion and back
    )

    df = df.join(df_handrecord['board_number','board_record_string','dealer','vulnerability'],on='board_number',how='left')

    df = df.rename({
        'board_number': 'Board',
        #'club_id_number': 'Club',
        'contract': 'Contract',
        #'game_date': 'Date',
        'match_points_ns': 'MP_NS',
        'match_points_ew': 'MP_EW',
        'pair_number': 'Pair_Number_NS',
        'opponent_pair_number': 'Pair_Number_EW',
        #'percentage_ns': 'Final_Standing_NS', # percentage_ns is not a column in the data. use percentage instead?
        #'percentage_ew': 'Final_Standing_EW', # percentage_ew is not a column in the data. use percentage instead?
        #'result': 'Result',
        #'round_number': 'Round',
        #'score_ns': 'Score_NS',
        #'score_ew': 'Score_EW',
        'score': 'Score_NS',
        'section_label': 'section_name', # change to section_name for compatibility with clubs
        #'table_number': 'Table',
        #'tricks_taken': 'Tricks',
    })

    df = df.with_columns([
        (pl.col('percentage_ns')/100).cast(pl.Float32).alias('Pct_NS'),
        (pl.col('percentage_ew')/100).cast(pl.Float32).alias('Pct_EW'),
        #pl.col('board_record_string').cast(pl.Utf8),
        pl.lit(results_dfs_d['event']['start_date'].to_list()[0]).alias('Date'), # no time in the data, can't use %Y-%m-%d %H:%M:%S
        pl.lit(results_dfs_d['event']['id'].to_list()[0]).alias('event_id'),
        #pl.col('hand_record_id').cast(pl.Int64), # only box_number is present in the data
        pl.col('Pair_Number_NS').cast(pl.UInt16),
        pl.col('Pair_Number_EW').cast(pl.UInt16),
        #pl.col('pair_summary_id').alias('pair_summary_id_ns'), # todo: remove this alias after ai model is updated to use pair_summary_id
        #pl.col('pair_summary_id').alias('pair_summary_id_ew'), # todo: remove this alias after ai model is updated to use pair_summary_id
    ])

    df = acbldf_to_mldf(df) # todo: temporarily convert to pandas to use augment_df until clean_validate_df is converted to polars

    return df


def acbldf_to_mldf(df: pl.DataFrame) -> pl.DataFrame:
    # Rename columns
    df = df.rename({'declarer': 'Declarer_Direction'})
    df = df.with_columns(pl.col('Declarer_Direction').replace_strict(mlBridgeLib.Direction_to_NESW_d,return_dtype=pl.String))

    # Drop rows where 'Board' is NaN
    df = df.filter(pl.col('Board').is_not_null() & pl.col('Board').gt(0))

    # Convert 'Board' to UInt8.
    # todo: use UInt16 instead of UInt8?
    df = df.with_columns(pl.col('Board').cast(pl.UInt8))

    if 'board_record_string' in df.columns:
        df = df.with_columns(pl.col('board_record_string').map_elements(mlBridgeLib.brs_to_pbn,return_dtype=pl.Utf8).alias('PBN'))

    df = df.rename({'dealer': 'Dealer'})
    df = df.with_columns(pl.col('Dealer').replace_strict(mlBridgeLib.Direction_to_NESW_d,return_dtype=pl.String))

    # Calculate percentages - problems strange values and with multiple section computations. Can't all be director's adjustments?
    if 'Pct_NS' not in df.columns and 'Pct_EW' not in df.columns:
        if 'MP_Top' not in df.columns:
            # Calculate 'MP_Top'
            df = df.with_columns([
                pl.col('MP_NS').count().over('Board').alias('MP_Top')
            ])
        df = df.with_columns([
            (pl.col('MP_NS').cast(pl.Float32) / pl.col('MP_Top')).alias('Pct_NS'),
            #(pl.col('MP_EW').cast(pl.Float32) / pl.col('MP_Top')).alias('Pct_EW')
        ])
        df = df.with_columns([
            pl.when(pl.col('Pct_NS') > 1).then(1).otherwise(pl.col('Pct_NS')).alias('Pct_NS'),
            #pl.when(pl.col('Pct_EW') > 1).then(1).otherwise(pl.col('Pct_EW')).alias('Pct_EW')
        ])
        df = df.with_columns([
            (1 - pl.col('Pct_NS')).alias('Pct_EW'),
        ])
    else:   
        # Cap percentages at 1
        df = df.with_columns([
            pl.when(pl.col('Pct_NS') > 1).then(1).otherwise(pl.col('Pct_NS')).alias('Pct_NS'),
            pl.when(pl.col('Pct_EW') > 1).then(1).otherwise(pl.col('Pct_EW')).alias('Pct_EW')
        ])
        # I've seen some seemingly correct Pct_NS but null for Pct_EW. Mystery. Can't all be directory adjustments?
        df = df.with_columns([
            (1 - pl.col('Pct_NS')).alias('Pct_EW'),
        ])
        df = df.with_columns([
            (pl.col('MP_NS') / pl.col('Pct_NS')).alias('MP_Top')
        ])

    # Function to transform names into "first last" format
    def last_first_to_first_last(name):
        # Replace commas with spaces and split
        parts = name.replace(',', ' ').split()
        # Return "first last" format
        return ' '.join(parts[1:] + parts[:1]) if len(parts) > 1 else name

    # Transpose player names
    for d in 'NESW':
        df = df.rename({f'player_number_{d.lower()}': f'Player_ID_{d}'})
        df = df.rename({f'player_name_{d.lower()}': f'Player_Name_{d}'})
        df = df.with_columns([
            pl.col(f'Player_ID_{d}').cast(pl.Utf8).alias(f'Player_ID_{d}'),
            #pl.col(f'Player_ID_{d}').cast(pl.UInt32).alias(f'iPlayer_Number_{d}'),
            pl.col(f'Player_Name_{d}').map_elements(last_first_to_first_last, return_dtype=pl.Utf8).alias(f'Player_Name_{d}')
        ])

    # chatgpt suggested this. nothing else looked as good. seems like something else would be better. it's one thing where pandas codes better than polars.
    df = df.with_columns(
        pl.when(pl.col("Declarer_Direction") == "N").then(pl.col("Player_ID_N"))
        .when(pl.col("Declarer_Direction") == "E").then(pl.col("Player_ID_E"))
        .when(pl.col("Declarer_Direction") == "S").then(pl.col("Player_ID_S"))
        .when(pl.col("Declarer_Direction") == "W").then(pl.col("Player_ID_W"))
        .otherwise(None)
        .alias("Number_Declarer")
    )

    # Clean up contracts
    df = df.with_columns(
           pl.col('Contract')
            .str.replace(' ', '',n=2)
            .str.to_uppercase()
            .str.replace('NT', 'N')
            .alias('Contract')
        )

    # drop rows where 'Score' is null. They are different from 'PASS' in 'Score'.
    df = df.filter(pl.col('Score_NS').is_not_null())
    # 'Score' is a string if no 'PASS' otherwise it might be an int (i64).
    if df.schema['Score_NS'] == pl.Utf8:
        # acbl tournament data has 'PASS' in 'score'.
        df = df.with_columns(
            pl.when(pl.col('Score_NS') == 'PASS')
            .then(pl.lit('PASS'))
            .otherwise(pl.col('Contract'))
            .alias('Contract')
        )
        # Drop invalid contracts
        drop_rows = (
            (pl.col('Contract') != 'PASS') & 
            (pl.col('Contract').is_null() | pl.col('Score_NS').is_null())
        )
        print_to_log_info('Dropping rows:',df.filter(drop_rows))
        df = df.filter(~drop_rows)
        df = df.with_columns(pl.col('Score_NS').str.replace('PASS', '0'))
    df = df.with_columns(pl.col('Score_NS').cast(pl.Int16,strict=False).alias('Score_NS'))
    df = df.with_columns(pl.col('Score_NS').neg().cast(pl.Int16,strict=False).alias('Score_EW'))

    df = df.with_columns([
        pl.when(pl.col('Contract') == 'PASS')
        .then(pl.lit(None))
        .otherwise(pl.col('Contract').str.slice(0, 1))
        .cast(pl.UInt8,strict=False)
        .alias('BidLvl'),

        pl.when(pl.col('Contract') == 'PASS')
        .then(pl.lit(None))
        .otherwise(pl.col('Contract').str.slice(1, 1))
        .cast(pl.String)
        .alias('BidSuit'),

        pl.when(pl.col('Contract') == 'PASS')
        .then(pl.lit(None))
        .otherwise(pl.col('Contract').str.slice(2))
        .cast(pl.String)
        .alias('Dbl'),
    ])

    # reformat contract to standard format. Using endplay's contract format.
    df = df.with_columns([
        pl.when(pl.col('Contract') == 'PASS')
        .then(pl.col('Contract'))
        .otherwise(pl.col('BidLvl').cast(pl.Utf8)+pl.col('BidSuit')+pl.col('Declarer_Direction')+pl.col('Dbl'))
        .cast(pl.String)
        .alias('Contract'),
    ])

    if 'vulnerability' in df.columns:
        df = df.rename({'vulnerability':'Vul'})
        df = df.with_columns([
            pl.col('Vul').replace_strict(mlBridgeLib.Vulnerability_to_Vul_d,return_dtype=pl.Utf8)
        ])
   
    if 'iVul' not in df.columns and 'Vul' not in df.columns:
        df = df.with_columns([
            pl.col('Board').map_elements(mlBridgeLib.BoardNumberToVul,return_dtype=pl.UInt8).alias('iVul'),
        ])

    if 'Vul' not in df.columns:
        df = df.with_columns([
            pl.col('iVul').replace_strict(mlBridgeLib.Vulnerability_to_Vul_d,return_dtype=pl.Utf8).alias('Vul')
        ])

    if 'iVul' not in df.columns:
        df = df.with_columns([
            pl.col('Vul').replace_strict(mlBridgeLib.vul_sym_to_index_d,return_dtype=pl.UInt8).alias('iVul'),
        ])

    # Create 'Result' and 'Tricks' columns
    if 'Result' in df.columns:
        df = df.with_columns([
            pl.when(pl.col('Result').is_not_null())
            .then(pl.col('Result').map_elements(lambda x: 0 if x in ['=', '0', ''] else int(x[1:]) if x[0] == '+' else int(x),return_dtype=pl.Int8))
            .otherwise(pl.col('Result'))
            .cast(pl.Int8)
            .alias('Result')
        ])
    else:
        assert 'Tricks' not in df.columns
        df = df.with_columns(pl.Series('scores_l',mlBridgeLib.ContractToScores(df),pl.List(pl.Int16)))

        # todo: use mlBridgeAugmentLib.ContractToScores?
        # adjusted score is the reason for any unexpected scores.
        df = df.with_columns(
            pl.Series('Result',[None if (None in r) or (r[1] not in r[2]) else r[2].index(r[1])-(r[3]+6) for r in df['Contract','Score_NS','scores_l','BidLvl'].rows()],dtype=pl.Int8),
        )
        # can be null if errata.
        df.filter(pl.col('Result').is_null())['Board','Contract','Score_NS','BidLvl','Vul','iVul','scores_l']
        df.drop_in_place('scores_l') # ugh, con.register() hangs unless this is done. Probably from when it was pl.Object.

    df = df.with_columns(
        pl.when((pl.col('Contract') == 'PASS') | (pl.col('Result').is_null())) # 'Result' can be null if errata.
        .then(pl.lit(None))
        .otherwise(pl.col('BidLvl') + 6 + pl.col('Result'))
        .alias('Tricks')

    )
    print_to_log_info('PASS:',df.filter(pl.col('Contract') == 'PASS').height)


    # Fill missing values
    if 'Round' in df.columns:
        df = df.with_columns([
            pl.col('Round').fill_null(0),
        ])

    if 'tb_count' in df.columns:
        df = df.with_columns([
            pl.col('tb_count').fill_null(0).cast(pl.UInt8),
        ])

    if 'Table' in df.columns:
        df = df.with_columns([
            pl.col('Table').fill_null(0).cast(pl.UInt8)
        ])

    # Assert no columns start with 'ns_' or 'ew_'
    for col in df.columns:
        assert not (col.startswith('ns_') or col.startswith('ew_') or col.startswith('NS_') or col.startswith('EW_')), col

    assert len(df) > 0
    return df


# todo: use Augment_Metric_By_Suits or TuplesToSuits?
# def Augment_Metric_By_Suits(metrics,metric,dtype='uint8'):
#     for d,direction in enumerate(mlBridgeLib.NESW):
#         for s,suit in  enumerate(mlBridgeLib.SHDC):
#             metrics['_'.join([metric,direction])] = metrics[metric].map(lambda x: x[1][d][0]).astype(dtype)
#             metrics['_'.join([metric,direction,suit])] = metrics[metric].map(lambda x: x[1][d][1][s]).astype(dtype)
#     for direction in mlBridgeLib.NS_EW:
#         metrics['_'.join([metric,direction])] = metrics['_'.join([metric,direction[0]])]+metrics['_'.join([metric,direction[1]])].astype(dtype)
#         for s,suit in  enumerate(mlBridgeLib.SHDC):
#             metrics['_'.join([metric,direction,suit])] = metrics['_'.join([metric,direction[0],suit])]+metrics['_'.join([metric,direction[1],suit])].astype(dtype)


# def TuplesToSuits(df,tuples,column,excludes=[]):
#     d = {}
#     d['_'.join([column])] = tuples.map(lambda x: x[0])
#     for i,direction in enumerate('NESW'):
#         d['_'.join([column,direction])] = tuples.map(lambda x: x[1][i][0])
#         for j,suit in enumerate('SHDC'):
#             d['_'.join([column,direction,suit])] = tuples.map(lambda x: x[1][i][1][j])
#     for i,direction in enumerate(['NS','EW']):
#         d['_'.join([column,direction])] = tuples.map(lambda x: x[1][i][0]+x[1][i+2][0])
#         for j,suit in enumerate('SHDC'):
#             d['_'.join([column,direction,suit])] = tuples.map(lambda x: x[1][i][1][j]+x[1][i+2][1][j])
#     for k,v in d.items():
#         if k not in excludes:
#             # PerformanceWarning: DataFrame is highly fragmented.
#             df[k] = v
#     return d


# # Pandas version of mlBridgeLib's Polars version
# # Create columns of contract types by partnership by suit by contract. e.g. CT_NS_C_Game
# def CategorifyContractTypeByDirection(df):
#     contract_types_d = {}
#     cols = df.filter(regex=r'CT_(NS|EW)_[CDHSN]').columns
#     for c in cols:
#         for t in mlBridgeLib.contract_types:
#             print_to_log_debug('CT:',c,t,len((t == df[c]).values))
#             new_c = c+'_'+t
#             contract_types_d[new_c] = (t == df[c]).values
#     return contract_types_d


# def augment_df(df,sd_cache_d):

#     # positions
#     df['Declarer_Pair_Direction'] = df['Declarer_Direction'].map(mlBridgeLib.PlayerDirectionToPairDirection)
#     df['Opponent_Pair_Direction'] = df['Declarer_Pair_Direction'].map(mlBridgeLib.PairDirectionToOpponentPairDirection)
#     df['Direction_OnLead'] = df['Declarer_Direction'].map(mlBridgeLib.NextPosition)
#     df['Direction_Dummy'] = df['Direction_OnLead'].map(mlBridgeLib.NextPosition)
#     df['Direction_NotOnLead'] = df['Direction_Dummy'].map(mlBridgeLib.NextPosition)
#     df['OnLead'] = df.apply(lambda r: r['Player_ID_'+r['Direction_OnLead']], axis='columns') # todo: keep as lower case?
#     df['Dummy'] = df.apply(lambda r: r['Player_ID_'+r['Direction_Dummy']], axis='columns') # todo: keep as lower case?
#     df['NotOnLead'] = df.apply(lambda r: r['Player_ID_'+r['Direction_NotOnLead']], axis='columns') # todo: keep as lower case?

#     # hands
#     df['hands'] = df['board_record_string'].map(mlBridgeLib.brs_to_hands)
#     assert df['hands'].map(mlBridgeLib.hands_to_brs).eq(df['board_record_string'].str.replace('-','').str.replace('T','10')).all(), df[df['hands'].map(mlBridgeLib.hands_to_brs).ne(df['board_record_string'])][['Board','board_record_string','hands']]
#     # ouch. Sometimes acbl hands use '-' in board_record_string, sometimes they don't. Are online hands without '-' and club f-f with '-'? Removing '-' in both so compare works.
#     df['PBN'] = df['hands'].map(mlBridgeLib.HandToPBN)
#     assert df['PBN'].map(mlBridgeLib.pbn_to_hands).eq(df['hands']).all(), df[df['PBN'].map(mlBridgeLib.pbn_to_hands).ne(df['hands'])]
#     brs = df['PBN'].map(mlBridgeLib.pbn_to_brs)
#     assert brs.map(mlBridgeLib.brs_to_pbn).eq(df['PBN']).all(), df[brs.map(mlBridgeLib.brs_to_pbn).ne(df['PBN'])]

#     # OHE cards
#     bin_handsl = mlBridgeLib.HandsLToBin(df['hands'])
#     ohe_handsl = mlBridgeLib.BinLToOHE(bin_handsl)
#     ohe_hands_df = mlBridgeLib.OHEToCards(df,ohe_handsl)
#     df = pd.concat([df,ohe_hands_df],axis='columns',join='inner')

#     # hand evaluation metrics
#     # todo: use Augment_Metric_By_Suits or TuplesToSuits?
#     # 'hands' is ordered CDHS
#     hcp = df['hands'].map(mlBridgeLib.HandsToHCP)
#     TuplesToSuits(df,hcp,'HCP',['HCP'])
#     qt = df['hands'].map(mlBridgeLib.HandsToQT)
#     TuplesToSuits(df,qt,'QT',['QT'])
#     dp = df['hands'].map(mlBridgeLib.HandsToDistributionPoints)
#     TuplesToSuits(df,dp,'DP',['DP'])
#     sl = df['hands'].map(mlBridgeLib.HandsToSuitLengths) # sl is needed later by LoTT
#     TuplesToSuits(df,sl,'SL',['SL','SL_N','SL_E','SL_S','SL_W','SL_NS','SL_EW'])
#     so = mlBridgeLib.CDHS
#     for d in mlBridgeLib.NESW:
#         # PerformanceWarning: DataFrame is highly fragmented.
#         df[f'SL_{d}_{so}'] = df.filter(regex=f'^SL_{d}_[{so}]$').values.tolist() # ordered from clubs to spades [CDHS]
#         # PerformanceWarning: DataFrame is highly fragmented.
#         df[f'SL_{d}_{so}_J'] = df[f'SL_{d}_{so}'].map(lambda l:'-'.join([str(v) for v in l])).astype('category') # joined CDHS into category
#         # PerformanceWarning: DataFrame is highly fragmented.
#         df[f'SL_{d}_ML_S'] = df[f'SL_{d}_{so}'].map(lambda l: [v for v,n in sorted([(ll,n) for n,ll in enumerate(l)],key=lambda k:(-k[0],k[1]))]) # ordered most-to-least
#         # PerformanceWarning: DataFrame is highly fragmented.
#         df[f'SL_{d}_ML_SI'] = df[f'SL_{d}_{so}'].map(lambda l: [n for v,n in sorted([(ll,n) for n,ll in enumerate(l)],key=lambda k:(-k[0],k[1]))]) # ordered most-to-least containing indexes
#         # PerformanceWarning: DataFrame is highly fragmented.
#         df[f'SL_{d}_ML_SJ'] = df[f'SL_{d}_ML_S'].map(lambda l:'-'.join([str(v) for v in l])).astype('category') # ordered most-to-least and joined into category

#     # Create columns containing column names of the NS,EW longest suit.
#     sl_cols = [('_'.join(['SL_Max',d]),['_'.join(['SL',d,s]) for s in mlBridgeLib.SHDC]) for d in mlBridgeLib.NS_EW]
#     for d in sl_cols:
#         # PerformanceWarning: DataFrame is highly fragmented.
#         df[d[0]] = df[d[1]].idxmax(axis=1).astype('category') # defaults to object so need string or category

#     df = mlBridgeLib.append_double_dummy_results(df)

#     # LoTT
#     ddmakes = df.apply(lambda r: tuple([tuple([r['_'.join(['DD',d,s])] for s in 'CDHSN']) for d in 'NESW']),axis='columns')
#     LoTT_l = [mlBridgeLib.LoTT_SHDC(t,l) for t,l in zip(ddmakes,sl)] # [mlBridgeLib.LoTT_SHDC(ddmakes[i],sl[i]) for i in range(len(df))]
#     df['LoTT_Tricks'] = [t for t,l,v in LoTT_l]
#     df['LoTT_Suit_Length'] = [l for t,l,v in LoTT_l] # todo: is this correct? use SL_Max_(NS|EW) instead? verify LoTT_Suit_Length against SL_Max_{declarer_pair_direction}.
#     df['LoTT_Variance'] = [v for t,l,v in LoTT_l]
#     del LoTT_l
#     df = df.astype({'LoTT_Tricks':'uint8','LoTT_Suit_Length':'uint8','LoTT_Variance':'int8'})

#     # ContractType
#     # PerformanceWarning: DataFrame is highly fragmented.
#     df['ContractType'] = df.apply(lambda r: 'PASS' if r['Contract'] == 'PASS' else mlBridgeLib.ContractType(r['BidLvl']+6,r['BidSuit']),axis='columns').astype('category')
#     # Create column of contract types by partnership by suit. e.g. CT_NS_C.
#     contract_types_d = mlBridgeLib.CategorifyContractTypeBySuit(ddmakes)
#     contract_types_df = pd.DataFrame(contract_types_d,dtype='category')
#     assert len(df) == len(contract_types_df)
#     df = pd.concat([df,contract_types_df],axis='columns') # ,join='inner')
#     del contract_types_df,contract_types_d
#     contract_types_d = CategorifyContractTypeByDirection(df) # using local pandas version instead of mlBridgeLib's Polars version
#     contract_types_df = pd.DataFrame(contract_types_d,dtype='category')
#     assert len(df) == len(contract_types_df)
#     df = pd.concat([df,contract_types_df],axis='columns') # ,join='inner')
#     del contract_types_df,contract_types_d

#     # create dict of NS matchpoint data.
#     matchpoint_ns_d = {} # key is board. values are matchpoint details (score, beats, ties, matchpoints, pct).
#     for board,g in df.groupby('Board'):
#         board_mps_ns = {}
#         for score_ns in g['Score_NS']:
#             board_mps_ns = mlBridgeLib.MatchPointScoreUpdate(score_ns,board_mps_ns) # convert to float32 here? It's still a string because it might originally have AVG+ or AVG- etc.
#         matchpoint_ns_d[board] = board_mps_ns
#     # validate boards are scored correctly
#     for board,g in df.groupby('Board'):
#         for score_ns,match_points_ns in zip(g['Score_NS'],g['MP_NS'].astype('float32')):
#             if matchpoint_ns_d[board][score_ns][3] != match_points_ns: # match_points_ns is a string because it might originally have AVG+ or AVG- etc.
#                 print_to_log(logging.WARNING,f'Board {board} score {matchpoint_ns_d[board][score_ns][3]} tuple {matchpoint_ns_d[board][score_ns]} does not match matchpoint score {match_points_ns}') # ok if off by epsilon

#     # Vul columns
#     df['Vul_NS'] = (df['iVul']&1).astype('bool')
#     df['Vul_EW'] = (df['iVul']&2).astype('bool')

#     # board result columns
#     df['OverTricks'] = df['Result'].gt(0)
#     df['JustMade'] = df['Result'].eq(0)
#     df['UnderTricks'] = df['Result'].lt(0)

#     df[f"Vul_Declarer"] = df.apply(lambda r: r['Vul_'+r['Declarer_Pair_Direction']], axis='columns')
#     df['Pct_Declarer'] = df.apply(lambda r: r['Pct_'+r['Declarer_Pair_Direction']], axis='columns')
#     df['Pair_Number_Declarer'] = df.apply(lambda r: r['Pair_Number_'+r['Declarer_Pair_Direction']], axis='columns')
#     df['Pair_Number_Defender'] = df.apply(lambda r: r['Pair_Number_'+r['Opponent_Pair_Direction']], axis='columns')
#     df['Number_Declarer'] = df.apply(lambda r: r['Player_ID_'+r['Declarer_Direction']], axis='columns') # todo: keep as lower case?
#     df['Name_Declarer'] = df.apply(lambda r: r['Player_Name_'+r['Declarer_Direction']], axis='columns')
#     # todo: drop either Tricks or Tricks_Declarer as they are invariant and duplicates
#     df['Tricks_Declarer'] = df['Tricks'] # synonym for Tricks
#     df['Score_Declarer'] = df.apply(lambda r: r['Score_'+r['Declarer_Pair_Direction']], axis='columns')
#     # recompute Score and compare against actual scores to catch scoring errors such as: Board 1 at https://my.acbl.org/club-results/details/878121
#     # just use Score_NS if score is uncomputable probably due to pd.NA from director's adjustment. (r['Result'] is pd.NA) works here. why?
#     df['Computed_Score_Declarer'] = df.apply(lambda r: 0 if r['Contract'] == 'PASS' else r['Score_NS'] if r['Result'] is pd.NA else mlBridgeLib.score(r['BidLvl']-1, 'CDHSN'.index(r['BidSuit']), len(r['Dbl']), ('NESW').index(r['Declarer_Direction']), r['Vul_Declarer'], r['Result'],True), axis='columns')
#     if (df['Score_Declarer'].ne(df['Computed_Score_Declarer'])|df['Score_NS'].ne(-df['Score_EW'])).any():
#         print_to_log(logging.WARNING, 'Invalid Scores:\n',df[df['Score_Declarer'].ne(df['Computed_Score_Declarer'])|df['Score_NS'].ne(-df['Score_EW'])][['Board','Contract','BidLvl','BidSuit','Dbl','Declarer_Direction','Vul_Declarer','Score_Declarer','Computed_Score_Declarer','Score_NS','Score_EW','Result']])
#     df['MPs_Declarer'] = df.apply(lambda r: r['MatchPoints_'+r['Declarer_Pair_Direction']], axis='columns')

#     df['DD_Tricks'] = df.apply(lambda r: pd.NA if r['Contract'] == 'PASS' else r['_'.join(['DD',r['Declarer_Direction'],r['BidSuit']])], axis='columns') # invariant
#     df['DD_Tricks_Dummy'] = df.apply(lambda r: pd.NA if r['Contract'] == 'PASS' else r['_'.join(['DD',r['Direction_Dummy'],r['BidSuit']])], axis='columns') # invariant
#     # NA for NT. df['DDSLDiff'] = df.apply(lambda r: pd.NA if r['Contract'] == 'PASS' else r['DD_Tricks']-r['SL_'+r['Declarer_Pair_Direction']+'_'+r['BidSuit']], axis='columns') # pd.NA or zero?
#     df['DD_Score_NS'] = df.apply(lambda r: 0 if r['Contract'] == 'PASS' else mlBridgeLib.score(r['BidLvl']-1, 'CDHSN'.index(r['BidSuit']), len(r['Dbl']), ('NSEW').index(r['Declarer_Direction']), r['Vul_Declarer'], r['DD_Tricks']-r['BidLvl']-6), axis='columns')
#     df['DD_Score_EW'] = -df['DD_Score_NS']
#     df['DD_MP_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['DD_Score_NS'],matchpoint_ns_d[r['Board']])[r['DD_Score_NS']][3],axis='columns')
#     df['DD_MP_EW'] = df['MP_Top']-df['DD_MP_NS']
#     df['DD_Pct_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['DD_Score_NS'],matchpoint_ns_d[r['Board']])[r['DD_Score_NS']][4],axis='columns')
#     df['DD_Pct_EW'] = 1-df['DD_Pct_NS']

#     # Declarer Par columns
#     # ACBL online games have no par score data. Must create it.
#     if 'par' not in df or df['par'].eq('').all():
#         df.rename({'Par_EndPlay_NS':'Par_NS','Par_EndPlay_EW':'Par_EW','Par_Contracts_EndPlay':'Par_Contracts'},axis='columns',inplace=True)
#         #df['Par_NS'] = df['Par_EndPlay_NS']
#         #df['Par_EW'] = df['Par_EndPlay_EW']
#         #df['Par_Contracts'] = df['Par_Contracts_EndPlay']
#         #df.drop(['Par_EndPlay_NS','Par_EndPlay_EW','Par_Contracts_EndPlay'],axis='columns',inplace=True)
#     else:
#         # parse par column and create Par column.
#         df['Par_NS'] = df['par'].map(lambda x: x.split(' ')[1]).astype('int16')
#         df['Par_EW'] = -df['Par_NS']
#         df['Par_Contracts'] = df['par'].map(lambda x: x.split(' ')[2:]).astype('string')
#     if 'par' in df:
#         df.drop(['par'],axis='columns',inplace=True)
#     df['Par_MPs_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['Par_NS'],matchpoint_ns_d[r['Board']])[r['Par_NS']][3],axis='columns')
#     df['Par_MPs_EW'] = df['MP_Top']-df['Par_MPs_NS']
#     df['Par_Pct_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['Par_NS'],matchpoint_ns_d[r['Board']])[r['Par_NS']][4],axis='columns')
#     df['Par_Pct_EW'] = 1-df['Par_Pct_NS']
#     df["Par_Declarer"] = df.apply(lambda r: r['Par_'+r['Declarer_Pair_Direction']], axis='columns')
#     #df["Par_MPs_Declarer"] = df.apply(lambda r: r['Par_MPs_'+r['Declarer_Pair_Direction']], axis='columns')
#     #df["Par_Pct_Declarer"] = df.apply(lambda r: r['Par_Pct_'+r['Declarer_Pair_Direction']], axis='columns')
#     #df['Par_Diff_Declarer'] = df['Score_Declarer']-df['Par_Declarer'] # adding convenience column to df. Actual Par Score vs DD Score
#     #df['Par_MPs_Diff_Declarer'] = df['MPs_Declarer'].astype('float32')-df['Par_MPs'] # forcing MPs_Declarer to float32. It is still string because it might originally have AVG+ or AVG- etc.
#     #df['Par_Pct_Diff_Declarer'] = df['Pct_Declarer']-df['Par_Pct_Declarer']
#     df['Tricks_DD_Diff_Declarer'] = df['Tricks_Declarer']-df['DD_Tricks'] # adding convenience column to df. Actual Tricks vs DD Tricks
#     #df['Score_DD_Diff_Declarer'] = df['Score_Declarer']-df['DD_Score_Declarer'] # adding convenience column to df. Actual Score vs DD Score

#     df['Declarer_Rating'] = df.groupby('Number_Declarer')['Tricks_DD_Diff_Declarer'].transform('mean').astype('float32')
#     # todo: resolve naming conflict: Defender_Par_GE, Defender_OnLead_Rating, Defender_NotOnLead_Rating vs Par_GE_Defender, OnLead_Rating_Defender, NotOnLead_Rating_Defender
#     df['Defender_Par_GE'] = df['Score_Declarer'].le(df['Par_Declarer'])
#     df['Defender_OnLead_Rating'] = df.groupby('OnLead')['Defender_Par_GE'].transform('mean').astype('float32')
#     df['Defender_NotOnLead_Rating'] = df.groupby('NotOnLead')['Defender_Par_GE'].transform('mean').astype('float32')

#     # masterpoints columns
#     # note: looks like masterpoints column is no longer available so need to obsolete it. fake it with 500 for everyone.
#     for d in mlBridgeLib.NESW:
#         #df['mp_total_'+d.lower()] = df['mp_total_'+d.lower()].astype('float32')
#         #df['mp_total_'+d.lower()] = df['mp_total_'+d.lower()].fillna(300) # unknown number of masterpoints. fill with 300.
#         df['mp_total_'+d.lower()] = 500 # todo: need to fake masterpoints because it's no longer available.
#     df['MP_Sum_NS'] = df['mp_total_n']+df['mp_total_s']
#     df['MP_Sum_EW'] = df['mp_total_e']+df['mp_total_w']
#     df['MP_Geo_NS'] = df['mp_total_n']*df['mp_total_s']
#     df['MP_Geo_EW'] = df['mp_total_e']*df['mp_total_w']

#     df, sd_cache_d = mlBridgeLib.Augment_Single_Dummy(df,sd_cache_d,10,matchpoint_ns_d) # {} is no cache

#     # todo: check dtypes
#     # df = df.astype({'Name_Declarer':'string','Score_Declarer':'int16','Par_Declarer':'int16','Pct_Declarer':'float32','DD_Tricks':'uint8','DD_Score_Declarer':'int16','DD_Pct_Declarer':'float32','Tricks_DD_Diff_Declarer':'int8','Score_DD_Diff_Declarer':'int16','Par_DD_Diff_Declarer':'int16','Par_Pct_Declarer':'float32','Pair_Declarer':'string','Pair_Defender':'string'})

#     # todo: verify every dtype is correct.
#     # todo: rename columns when there's a better name
#     df.rename({'dealer':'Dealer'},axis='columns',inplace=True)
#     assert df['Dealer'].isin(list('NESW')).all()
#     df['Dealer'] = df['Dealer'].astype('category') # todo: should this be done earlier?
#     assert df['iVul'].isin([0,1,2,3]).all() # 0 to 3
#     df['iVul'] = df['iVul'].astype('uint8') # todo: should this be done earlier?
#     df['iDate'] = df['Date'].astype('int64')
#     return df, sd_cache_d, matchpoint_ns_d


# # todo: merge into mlBridgeAugmentLib. requires converting to Polars.
# def mlBridgeLib.Augment_Single_Dummy(df,sd_cache_d,produce,matchpoint_ns_d):

#     sd_cache_d = mlBridgeLib.append_single_dummy_results(df['PBN'],sd_cache_d,produce)
#     df['SD_Prob'] = df.apply(lambda r: sd_cache_d[r['PBN']].get(tuple([r['Declarer_Pair_Direction'],r['Declarer_Direction'],r['BidSuit']]),[0]*14),axis='columns') # had to use get(tuple([...]))
#     df['SD_Scores'] = df.apply(Create_SD_Scores,axis='columns')
#     df['SD_Score_NS'] = df.apply(Create_SD_Score,axis='columns').astype('int16') # Declarer's direction
#     df['SD_Score_EW'] = -df['SD_Score_NS']
#     df['SD_MP_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['SD_Score_NS'],matchpoint_ns_d[r['Board']])[r['SD_Score_NS']][3],axis='columns')
#     df['SD_MP_EW'] = (df['MP_Top']-df['SD_MP_NS']).astype('float32')
#     df['SD_Pct_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['SD_Score_NS'],matchpoint_ns_d[r['Board']])[r['SD_Score_NS']][4],axis='columns')
#     df['SD_Pct_EW'] = (1-df['SD_Pct_NS']).astype('float32')
#     max_score_contract = df.apply(Create_SD_Score_Max,axis='columns')
#     df['SD_Score_Max_NS'] = pd.Series([score for score,contract in max_score_contract],dtype='float32')
#     df['SD_Score_Max_EW'] = pd.Series([-score for score,contract in max_score_contract],dtype='float32')
#     df['SD_Contract_Max'] = pd.Series([contract for score,contract in max_score_contract],dtype='string') # invariant
#     del max_score_contract
#     df['SD_MP_Max_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['SD_Score_Max_NS'],matchpoint_ns_d[r['Board']])[r['SD_Score_Max_NS']][3],axis='columns')
#     df['SD_MP_Max_EW'] = (df['MP_Top']-df['SD_MP_Max_NS']).astype('float32')
#     df['SD_Pct_Max_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['SD_Score_Max_NS'],matchpoint_ns_d[r['Board']])[r['SD_Score_Max_NS']][4],axis='columns')
#     df['SD_Pct_Max_EW'] = (1-df['SD_Pct_Max_NS']).astype('float32')
#     df['SD_Score_Diff_NS'] = (df['Score_NS']-df['SD_Score_NS']).astype('int16')
#     df['SD_Score_Diff_EW'] = (df['Score_EW']-df['SD_Score_EW']).astype('int16')
#     df['SD_Score_Max_Diff_NS'] = (df['Score_NS']-df['SD_Score_Max_NS']).astype('int16')
#     df['SD_Score_Max_Diff_EW'] = (df['Score_EW']-df['SD_Score_Max_EW']).astype('int16')
#     df['SD_Pct_Diff_NS'] = (df['Pct_NS']-df['SD_Pct_NS']).astype('float32')
#     df['SD_Pct_Diff_EW'] = (df['Pct_EW']-df['SD_Pct_EW']).astype('float32')
#     df['SD_Pct_Max_Diff_NS'] = (df['Pct_NS']-df['SD_Pct_Max_NS']).astype('float32')
#     df['SD_Pct_Max_Diff_EW'] = (df['Pct_EW']-df['SD_Pct_Max_EW']).astype('float32')
#     df['SD_Par_Pct_Diff_NS'] = (df['Par_Pct_NS']-df['SD_Pct_Diff_NS']).astype('float32')
#     df['SD_Par_Pct_Diff_EW'] = (df['Par_Pct_EW']-df['SD_Pct_Diff_EW']).astype('float32')
#     df['SD_Par_Pct_Max_Diff_NS'] = (df['Par_Pct_NS']-df['SD_Pct_Max_Diff_NS']).astype('float32')
#     df['SD_Par_Pct_Max_Diff_EW'] = (df['Par_Pct_EW']-df['SD_Pct_Max_Diff_EW']).astype('float32')
#     # using same df to avoid the issue with creating new columns. New columns require meta data will need to be changed too.
#     sd_df = pd.DataFrame(df['SD_Prob'].values.tolist(),columns=[f'SD_Prob_Taking_{i}' for i in range(14)])
#     for c in sd_df.columns:
#         df[c] = sd_df[c].astype('float32')
#     return df, sd_cache_d


# def Create_SD_Scores(r):
#     if r['Contract'] != 'PASS':
#         level = r['BidLvl']-1
#         suit = r['BidSuit']
#         iCDHSN = 'CDHSN'.index(suit)
#         nsew = r['Declarer_Direction']
#         iNSEW = 'NSEW'.index(nsew)
#         vul = r['Vul_Declarer']
#         double = len(r['Dbl'])
#         scores_l = mlBridgeLib.ScoreDoubledSets(level, iCDHSN, vul, double, iNSEW)
#         return scores_l
#     else:
#         return [0]*14


# #def Create_SD_Probs(r):
# #    return [r['Prob_Take_'+str(n)] for n in range(14)] # todo: this was previously computed. can we just use that?


# def Create_SD_Score(r):
#     probs = r['SD_Prob']
#     scores_l = r['SD_Scores']
#     ps = sum(prob*score for prob,score in zip(probs,scores_l))
#     return ps if r['Declarer_Direction'] in 'NS' else -ps


# # Highest expected score, same suit, any level
# # Note: score_max may exceed par score when probability of making/setting contract is high.
# def Create_SD_Score_Max(r):
#     score_max = None
#     if r['Contract'] != 'PASS':
#         suit = r['BidSuit']
#         iCDHSN = 'CDHSN'.index(suit)
#         nsew = r['Declarer_Direction']
#         iNSEW = 'NSEW'.index(nsew)
#         vul = r['Vul_Declarer']
#         double = len(r['Dbl'])
#         probs = r['SD_Prob']
#         for level in range(7):
#             scores_l = mlBridgeLib.ScoreDoubledSets(level, iCDHSN, vul, double, iNSEW)
#             score = sum(prob*score for prob,score in zip(probs,scores_l))
#             # todo: do same for redoubled? or is that too rare to matter?
#             #scoresx_l = mlBridgeLib.ScoreDoubledSets(level, iCDHSN, vul, 1, iNSEW)
#             #scorex = sum(prob*score for prob,score in zip(probs,scoresx_l))
#             isdoubled = double
#             #if scorex > score:
#             #    score = scorex
#             #    if isdoubled == 0:
#             #        isdoubled = 1
#             # must be mindful that NS makes positive scores but EW makes negative scores.
#             if nsew in 'NS' and (score_max is None or score > score_max):
#                 score_max = score
#                 contract_max = str(level+1)+suit+['','X','XX'][isdoubled]+' '+nsew
#             elif nsew in 'EW' and (score_max is None or score < score_max):
#                 score_max = score
#                 contract_max = str(level+1)+suit+['','X','XX'][isdoubled]+' '+nsew
#     else:
#         score_max = 0
#         contract_max = 'PASS'
#     return (score_max, contract_max)
