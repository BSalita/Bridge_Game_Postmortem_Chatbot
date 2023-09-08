#import openai
#from openai import openai_object # used to suppress vscode type checking errors
import pandas as pd
import re
#import time
import json
import requests
from bs4 import BeautifulSoup
from pprint import pprint
import pathlib
import sys
#from tenacity import retry, wait_random_exponential, stop_after_attempt
#from termcolor import colored
#import dotenv
#from dotenv import load_dotenv
#import os
#import inspect


sys.path.append(str(pathlib.Path.cwd().parent.parent.joinpath('mlBridgeLib'))) # removed .parent
sys.path.append(str(pathlib.Path.cwd().parent.parent.joinpath('chatlib'))) # removed .parent
sys.path
import mlBridgeLib
import endplay.parsers.lin as lin
import endplay.parsers.pbn as pbn
#load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")
#GPT_MODEL = "gpt-3.5-turbo-0613"


def get_club_results_details_data(url):
    print('details url:',url)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    assert response.status_code == 200, [url, response.status_code]

    soup = BeautifulSoup(response.content, "html.parser")

    if soup.find('result-details-combined-section'):
        data = soup.find('result-details-combined-section')['v-bind:data']
    elif soup.find('result-details'):
        data = soup.find('result-details')['v-bind:data']
    elif soup.find('team-result-details'):
        return None # todo: handle team events
        data = soup.find('team-result-details')['v-bind:data']
    else:
        assert False, "Can't find data tag."
    assert data is not None and isinstance(data,str) and len(data), [url, data]

    details_data = json.loads(data) # returns dict from json
    return details_data


def get_club_results_from_acbl_number(acbl_number):
    url = f"https://my.acbl.org/club-results/my-results/{acbl_number}"
    print('my-results url:',url)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    assert response.status_code == 200, [url, response.status_code]

    soup = BeautifulSoup(response.content, "html.parser")

    # Find all anchor tags with href attributes
    anchor_pattern = re.compile(r'/club\-results/details/\d{6}$')
    anchor_tags = soup.find_all('a', href=anchor_pattern)
    anchor_d = {a['href']:a for a in anchor_tags}
    hrefs = sorted(anchor_d.keys(),reverse=True)
    # 847339 2023-08-21, Ft Lauderdale Bridge Club, Mon Aft Stratified Pair, Monday Afternoon, 58.52%
    msgs = [', '.join([anchor_d[href].parent.parent.find_all('td')[i].text.replace('\n','').strip() for i in [0,1,2,3,5]]) for href in hrefs]
    assert len(hrefs) == len(msgs)

    # Print the href attributes
    my_results_details_data = {}
    for href,msg in zip(hrefs,msgs):
        detail_url = 'https://my.acbl.org'+href
        game_id = int(href.split('/')[-1]) # extract event_id from href which is the last part of url
        my_results_details_data[game_id] = (url, detail_url, msg)
    return my_results_details_data


# obsolete?
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


def create_dfs(acbl_number,event_url):
    data = get_club_results_details_data(event_url)
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


def merge_clean_augment_dfs(dfs,sd_cache_d,acbl_number):

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

    df = clean_validate_df(df)
    df, sd_cache_d, matchpoint_ns_d = augment_df(df,sd_cache_d)

    return df, sd_cache_d, matchpoint_ns_d


def clean_validate_df(df):

    # cleanup all sorts of columns including: 'round_number','table_number','board_number','dealer','Vul','contract','BidLvl','BidSuit','Dbl','declarer','tricks_taken','result','ns_score','ew_score','par'

    # if any rows were dropped, the calculation of board's top/pct will be wrong (outside of (0,1)). Need to calculate Board_Top now, before dropping rows.
    tops = {}
    df.rename({'board':'Board','ns_match_points':'MatchPoints_NS','ew_match_points':'MatchPoints_EW'},axis='columns',inplace=True)
    df['Board'] = df['Board'].astype('uint8')
    for b in df['Board'].unique():
        tops[b] = df[df['Board'].eq(b)]['MatchPoints_NS'].count()-1
        assert tops[b] == df[df['Board'].eq(b)]['MatchPoints_EW'].count()-1
    df['Board_Top'] = df['Board'].map(tops)
    df['Pct_NS'] = df['MatchPoints_NS'].astype('float32').div(df['Board_Top'])
    df[df['Pct_NS']>1] = 1 # assuming this can only happen if director adjusts score. todo: print >1 cases.
    assert df['Pct_NS'].between(0,1).all(), [df[~df['Pct_NS'].between(0,1)][['Board','MatchPoints_NS','Board_Top','Pct_NS']]]
    df['Pct_EW'] = df['MatchPoints_EW'].astype('float32').div(df['Board_Top'])
    df[df['Pct_EW']>1] = 1 # assuming this can only happen if director adjusts score. todo: print >1 cases.
    assert df['Pct_EW'].between(0,1).all(), [df[~df['Pct_EW'].between(0,1)][['Board','MatchPoints_EW','Board_Top','Pct_EW']]]

    # transpose pair_name (last name, first_name).
    for d in 'NSEW':
        df.rename({'player_number_'+d.lower():'Player_Number_'+d for d in 'NSEW'},axis='columns',inplace=True)
        df['Player_Name_'+d] = df['player_name_'+d.lower()].str.split(',').str[::-1].str.join(' ') # github Copilot wrote this line!
        df.drop(['player_name_'+d.lower()],axis='columns',inplace=True)

    # clean up contracts. Create BidLvl, BidSuit, Dbl columns.
    contractdf = df['contract'].str.replace(' ','').str.upper().str.replace('NT','N').str.extract(r'^(?P<BidLvl>\d)(?P<BidSuit>C|D|H|S|N)(?P<Dbl>X*)$')
    df.drop(['contract'],axis='columns',inplace=True)
    df['BidLvl'] = contractdf['BidLvl']
    df['BidSuit'] = contractdf['BidSuit']
    df['Dbl'] = contractdf['Dbl']
    del contractdf
    drop_rows = df['BidLvl'].isna()
    df.drop(df[drop_rows].index,inplace=True)
    drop_rows = ~df['declarer'].isin(list('NSEW')) # keep N,S,E,W. Drop EW, NS, w, ... < 500 cases.
    df.drop(df[drop_rows].index,inplace=True)
    df['Contract'] = df['BidLvl']+df['BidSuit']+df['Dbl']+' '+df['declarer']
    df['BidLvl'] = df['BidLvl'].astype('uint8')
    assert df['BidLvl'].isna().sum() == 0
    assert df['BidLvl'].between(1,7,inclusive='both').all()
    assert df['BidSuit'].isna().sum() == 0
    assert df['BidSuit'].isin(list('CDHSN')).all()
    assert df['Dbl'].isna().sum() == 0
    assert df['Dbl'].isin(['','X','XX']).all()

    # validate table_number, declarer, board_number
    assert df['table_number'].isna().all() or df['table_number'].ge(1).all() # some events have NaN table_numbers.
    assert df['board_number'].ge(1).all()

    # create more useful Vul column
    df['Vul'] = df['board_number'].map(mlBridgeLib.BoardNumberToVul) # 0 to 3

    # drop invalid tricks taken

    assert df['result'].notna().all() and df['result'].notnull().all()
    df['result'] = df['result'].map(lambda x: 0 if x=='=' else int(x[1:]) if x[0]=='+' else int(x)).astype('int8') # todo: use transform instead of map?

    if df['tricks_taken'].notna().any() or df['tricks_taken'].notnull().any(): # some events have empty tricks_taken. They're all None (not same as NaN).
        assert df['tricks_taken'].notnull().all()
        drop_rows = ~df['tricks_taken'].str.isnumeric()
        df.drop(df[drop_rows].index,inplace=True)
        df['tricks_taken'] = df['tricks_taken'].astype('uint8')
        assert df['tricks_taken'].eq(df['BidLvl']+6+df['result']).all()
    else:
        df['tricks_taken'] = (df['BidLvl']+6+df['result']).astype('uint8')
    drop_rows = ~df['tricks_taken'].between(0,13,inclusive='both')
    df.drop(df[drop_rows].index,inplace=True)
    df.rename({'tricks_taken':'Tricks'},axis='columns',inplace=True)

    # drop invalid round numbers
    if df['round_number'].notnull().any():
        drop_rows = df['round_number'].isna()
        df.drop(df[drop_rows].index,inplace=True)

    # cleanup invalid 'ns_score','ew_score','scores_l','Tricks','board_number','BidLvl','BidSuit','Vul','Dbl','declarer' 
    df.rename({'ns_score':'Score_NS', 'ew_score':'Score_EW'},axis='columns',inplace=True)
    df.loc[df['Score_NS'] == 'PASS','Score_NS'] = '0'
    assert df['Score_NS'].ne('PASS').all()
    drop_rows = ~df['Score_NS'].map(lambda c: c[c[0] == '-':].isnumeric()) | ~df['Score_NS'].map(lambda c: c[c[0] == '-':].isnumeric())
    df.drop(df[drop_rows].index,inplace=True)
    assert df['Score_NS'].isna().sum() == 0
    assert df['Score_NS'].isna().sum() == 0
    df['Score_NS'] = df['Score_NS'].astype('int16')
    df['Score_EW'] = -df['Score_NS']

    scoresd, setScoresd, makeScoresd = mlBridgeLib.ScoreDicts() # (level, suit, vulnerability, double, declarer)
    df['scores_l'] = df.apply(lambda r: scoresd[r['BidLvl']-1,mlBridgeLib.StrainSymToValue(r['BidSuit']),mlBridgeLib.DirectionSymToDealer(r['declarer']) in mlBridgeLib.vul_directions[r['Vul']],len(r['Dbl']),'NSEW'.index(r['declarer'])],axis='columns') # scoresd[level, suit, vulnerability, double, declarer]
    drop_rows = df[df.apply(lambda r: r['Score_NS'] not in r['scores_l'],axis='columns')].index
    df.drop(drop_rows,inplace=True)
    drop_rows = df[df.apply(lambda r: (r['scores_l'].index(r['Score_NS']) != r['Tricks']) | (r['Score_NS'] != -r['Score_EW']),axis='columns')].index
    df.drop(drop_rows,inplace=True)
    df.drop(['scores_l'],axis='columns',inplace=True)

    # todo: create table of renames instead of discrete renames.
    df.rename({'ns_pair':'Pair_Number_NS','ew_pair':'Pair_Number_EW'},axis='columns',inplace=True)
    df.rename({'pair_summary_id_ns':'Pair_Summary_ID_NS','pair_summary_id_ew':'Pair_Summary_ID_EW'},axis='columns',inplace=True)
    df.rename({'double_dummy_ns':'Double_Dummy_NS','double_dummy_ew':'Double_Dummy_EW'},axis='columns',inplace=True)
    df.rename({'percentage_NS':'Final_Standing_NS','percentage_EW':'Final_Standing_EW'},axis='columns',inplace=True) # attempting to give unique name so ChatGPT doesn't confuse with Pct_NS, Pct_EW.
    df.astype({'Final_Standing_NS':'float32','Final_Standing_EW':'float32'})

    # todo: move ns_matchpoints rename to here.
    # rename pair_summary_ns and ew. rename double_dummy_ns and ew.
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
    df.rename({'declarer':'Direction_Declarer'},axis='columns',inplace=True)
    df['Direction_Declarer'] = df['Direction_Declarer']
    df['Pair_Direction_Declarer'] = df['Direction_Declarer'].map(mlBridgeLib.PlayerDirectionToPairDirection)
    df['Opponent_Pair_Direction'] = df['Pair_Direction_Declarer'].map(mlBridgeLib.PairDirectionToOpponentPairDirection)
    df['Direction_OnLead'] = df['Direction_Declarer'].map(mlBridgeLib.NextPosition)
    df['Direction_Dummy'] = df['Direction_OnLead'].map(mlBridgeLib.NextPosition)
    df['Direction_NotOnLead'] = df['Direction_Dummy'].map(mlBridgeLib.NextPosition)

    # hands
    df['hands'] = df['board_record_string'].map(mlBridgeLib.brs_to_hands)
    # ouch. Sometimes acbl hands use '-' in board_record_string, sometimes they don't. Are online hands without and club f-f with? Removing '-' in both so compare works.
    assert df['hands'].map(mlBridgeLib.hands_to_brs).str.replace('-','').eq(df['board_record_string'].str.replace('-','')).all(), [df['hands'].map(mlBridgeLib.hands_to_brs).iloc[0], df['board_record_string'].iloc[0]]
    df['PBN'] = df['hands'].map(mlBridgeLib.HandToPBN)
    assert df['PBN'].map(mlBridgeLib.pbn_to_hands).eq(df['hands']).all(), [df['PBN'].map(mlBridgeLib.pbn_to_hands).iloc[0],df['hands'].iloc[0]]
    brs = df['PBN'].map(mlBridgeLib.pbn_to_brs)
    assert brs.map(mlBridgeLib.brs_to_pbn).eq(df['PBN']).all(), [brs.map(mlBridgeLib.brs_to_pbn).iloc[0],df['PBN'].iloc[0]]

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

    df = mlBridgeLib.append_double_dummy_results(df)

    # LoTT
    ddmakes = df.apply(lambda r: tuple([tuple([r['_'.join(['DD',d,s])] for s in 'SHDC']) for d in 'NESW']),axis='columns')
    LoTT_l = [mlBridgeLib.LoTT_SHDC(t,l) for t,l in zip(ddmakes,sl)] # [mlBridgeLib.LoTT_SHDC(ddmakes[i],sl[i]) for i in range(len(df))]
    df['LoTT_Tricks'] = [t for t,l,v in LoTT_l]
    df['LoTT_Suit_Length'] = [l for t,l,v in LoTT_l]
    df['LoTT_Variance'] = [v for t,l,v in LoTT_l]
    del LoTT_l
    df = df.astype({'LoTT_Tricks':'uint8','LoTT_Suit_Length':'uint8','LoTT_Variance':'int8'})

    # ContractType
    df['ContractType'] = df.apply(lambda r: mlBridgeLib.ContractType(r['BidLvl']+6,r['BidSuit']),axis='columns').astype('category')
    # Create column of contract types by partnership by suit. e.g. CT_NS_C.
    contract_types_d = mlBridgeLib.CategorifyContractType(ddmakes)
    contract_types_df = pd.DataFrame(contract_types_d,dtype='category')
    assert len(df) == len(contract_types_df)
    df = pd.concat([df,contract_types_df],axis='columns',join='inner')
    del contract_types_df,contract_types_d
    # Create columns of contract types by partnership by suit by contract. e.g. CT_NS_C_Game
    contract_types_d = {}
    cols = df.filter(regex=r'CT_(NS|EW)_[CDHSN]').columns
    for c in cols:
        for t in mlBridgeLib.contract_types:
            print(c,t,len((t == df[c]).values))
            new_c = c+'_'+t
            contract_types_d[new_c] = (t == df[c]).values
    #contract_types_d = CategorifyContractType(ddmakes)
    contract_types_df = pd.DataFrame(contract_types_d)
    assert len(df) == len(contract_types_df)
    df = pd.concat([df,contract_types_df],axis='columns',join='inner')
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
    df.rename({'result':'Result'},axis='columns',inplace=True)
    df['OverTricks'] = df['Result'].gt(0)
    df['JustMade'] = df['Result'].eq(0)
    df['UnderTricks'] = df['Result'].lt(0)

    df[f"Vul_Declarer"] = df.apply(lambda r: r['Vul_'+r['Pair_Direction_Declarer']], axis='columns')
    df['Pct_Declarer'] = df.apply(lambda r: r['Pct_'+r['Pair_Direction_Declarer']], axis='columns')
    df['Pair_Number_Declarer'] = df.apply(lambda r: r['Pair_Number_'+r['Pair_Direction_Declarer']], axis='columns')
    df['Pair_Number_Defender'] = df.apply(lambda r: r['Pair_Number_'+r['Opponent_Pair_Direction']], axis='columns')
    df['Number_Declarer'] = df.apply(lambda r: r['Player_Number_'+r['Direction_Declarer']], axis='columns') # todo: keep as lower case?
    df['Name_Declarer'] = df.apply(lambda r: r['Player_Name_'+r['Direction_Declarer']], axis='columns')
    # todo: drop either Tricks or Tricks_Declarer as they are invariant and duplicates
    df['Tricks_Declarer'] = df['Tricks'] # synonym for Tricks
    df['Score_Declarer'] = df.apply(lambda r: r['Score_'+r['Pair_Direction_Declarer']], axis='columns')
    df['MPs_Declarer'] = df.apply(lambda r: r['MatchPoints_'+r['Pair_Direction_Declarer']], axis='columns')

    df['DDTricks'] = df.apply(lambda r: r['_'.join(['DD',r['Direction_Declarer'],r['BidSuit']])], axis='columns') # invariant
    df['DDTricks_Dummy'] = df.apply(lambda r: r['_'.join(['DD',r['Direction_Dummy'],r['BidSuit']])], axis='columns') # invariant
    df['DDSLDiff'] = df.apply(lambda r: pd.NA if r['BidSuit']=='N' else r['DDTricks']-r['SL_'+r['Pair_Direction_Declarer']+'_'+r['BidSuit']], axis='columns') # pd.NA or zero?
    df['DDScore_NS'] = df.apply(lambda r: 0 if r['BidLvl']== 0 else mlBridgeLib.score(r['BidLvl']-1, 'CDHSN'.index(r['BidSuit']), len(r['Dbl']), ('NSEW').index(r['Direction_Declarer']), mlBridgeLib.DirectionSymToVulBool(r['Vul_Declarer'],r['Direction_Declarer']), r['DDTricks']-r['BidLvl']-6), axis='columns')
    df['DDScore_EW'] = -df['DDScore_NS']
    df['DDMPs_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['DDScore_NS'],matchpoint_ns_d[r['Board']])[r['DDScore_NS']][3],axis='columns')
    df['DDMPs_EW'] = df['Board_Top']-df['DDMPs_NS']
    df['DDPct_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['DDScore_NS'],matchpoint_ns_d[r['Board']])[r['DDScore_NS']][4],axis='columns')
    df['DDPct_EW'] = 1-df['DDPct_NS']

    # Declarer ParScore columns
    # ACBL online games have no par score data. Must create it.
    if df['par'].eq('').all():
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
    df.drop(['par'],axis='columns',inplace=True)
    df['ParScore_MPs_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['ParScore_NS'],matchpoint_ns_d[r['Board']])[r['ParScore_NS']][3],axis='columns')
    df['ParScore_MPs_EW'] = df['Board_Top']-df['ParScore_MPs_NS']
    df['ParScore_Pct_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['ParScore_NS'],matchpoint_ns_d[r['Board']])[r['ParScore_NS']][4],axis='columns')
    df['ParScore_Pct_EW'] = 1-df['ParScore_Pct_NS']
    #df["ParScore_Declarer"] = df.apply(lambda r: r['ParScore_'+r['Pair_Direction_Declarer']], axis='columns')
    #df["ParScore_MPs_Declarer"] = df.apply(lambda r: r['ParScore_MPs_'+r['Pair_Direction_Declarer']], axis='columns')
    #df["ParScore_Pct_Declarer"] = df.apply(lambda r: r['ParScore_Pct_'+r['Pair_Direction_Declarer']], axis='columns')
    #df['ParScore_Diff_Declarer'] = df['Score_Declarer']-df['ParScore_Declarer'] # adding convenience column to df. Actual Par Score vs DD Score
    #df['ParScore_MPs_Diff_Declarer'] = df['MPs_Declarer'].astype('float32')-df['ParScore_MPs'] # forcing MPs_Declarer to float32. It is still string because it might originally have AVG+ or AVG- etc.
    #df['ParScore_Pct_Diff_Declarer'] = df['Pct_Declarer']-df['ParScore_Pct_Declarer']
    #df['Tricks_DD_Diff_Declarer'] = df['Tricks_Declarer']-df['DDTricks] # adding convenience column to df. Actual Tricks vs DD Tricks
    #df['Score_DD_Diff_Declarer'] = df['Score_Declarer']-df['DD_Score_Declarer'] # adding convenience column to df. Actual Score vs DD Score

    df, sd_cache_d = Augment_Single_Dummy(df,sd_cache_d,10,matchpoint_ns_d) # {} is no cache

    # todo: check dtypes
    # df = df.astype({'Name_Declarer':'string','Score_Declarer':'int16','ParScore_Declarer':'int16','Pct_Declarer':'float32','DDTricks':'uint8','DD_Score_Declarer':'int16','DD_Pct_Declarer':'float32','Tricks_DD_Diff_Declarer':'int8','Score_DD_Diff_Declarer':'int16','ParScore_DD_Diff_Declarer':'int16','ParScore_Pct_Declarer':'float32','Pair_Declarer':'string','Pair_Defender':'string'})

    # todo: verify every dtype is correct.
    # todo: rename columns when there's a better name
    df['hand_record_id'] = df['hand_record_id'].astype('string')
    df.rename({'dealer':'Dealer'},axis='columns',inplace=True)
    df['Dealer'] = df['Dealer'].astype('string')
    df['Vul'] = df['Vul'].astype('string')
    df['board_record_string'] = df['board_record_string'].astype('string')
    
    return df, sd_cache_d, matchpoint_ns_d


def Augment_Single_Dummy(df,sd_cache_d,produce,matchpoint_ns_d):

    sd_cache_d = mlBridgeLib.append_single_dummy_results(df['PBN'],sd_cache_d,produce)
    df['SDProbs'] = df.apply(lambda r: sd_cache_d[r['PBN']][r['Pair_Direction_Declarer'],r['Direction_Declarer'],r['BidSuit']],axis='columns')
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
        nsew = r['Direction_Declarer']
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
    return ps if r['Direction_Declarer'] in 'NS' else -ps


# Highest expected score, same suit, any level
def Create_SD_Score_Max(r):
    score_max = None
    if r['Score_Declarer']:
        suit = r['BidSuit']
        iCDHSN = 'CDHSN'.index(suit)
        nsew = r['Direction_Declarer']
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
    return (score_max, contract_max)
