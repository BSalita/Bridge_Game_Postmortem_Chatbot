#!/usr/bin/env python
# coding: utf-8

# important: any change to df requires conn.register() to be called again

#!pip install openai python-dotenv pandas --quiet

# todo: load_model() is failing if numpy >= 2.0.0 is installed.

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

import sys
from collections import defaultdict
import pathlib
import re
import time
import streamlit as st
import openai
from openai import AsyncOpenAI
#from openai import openai_object  # used to suppress vscode type checking errors
import pandas as pd
import duckdb
import json
import os
from datetime import datetime, timezone 
from dotenv import load_dotenv
import streamlit_chat
import asyncio
#from streamlit_profiler import Profiler # Profiler -- temp?

# Only declared to display version information
import fastai
import numpy as np
import polars as pl
import safetensors
import sklearn
import torch

load_dotenv()
acbl_api_key = os.getenv("ACBL_API_KEY")
assert acbl_api_key is not None, "ACBL_API_KEY environment variable not set. See README.md for instructions."
openai_api_key = os.getenv("OPENAI_API_KEY")
assert openai_api_key is not None, "OPENAI_API_KEY environment variable not set. See README.md for instructions."
openai_async_client = AsyncOpenAI(api_key=openai_api_key)
DEFAULT_CHEAP_AI_MODEL = "gpt-3.5-turbo-1106" # -1106 until Dec 11th 2023. "gpt-3.5-turbo" is cheapest. "gpt-4" is most expensive.
DEFAULT_LARGE_AI_MODEL = "gpt-3.5-turbo-1106" # -1106 until Dec 11th 2023. now cheapest "gpt-3.5-turbo-16k" # might not be needed now that schema size is reduced.
DEFAULT_AI_MODEL = DEFAULT_LARGE_AI_MODEL
DEFAULT_GPT4_AI_MODEL = "gpt-4-turbo-preview" # preview is always newest
#DEFAULT_AI_MODEL = DEFAULT_GPT4_AI_MODEL
DEFAULT_AI_MODEL_TEMPERATURE = 0.0

# todo: doesn't some variation of import chatlib.chatlib work instead of using sys.path.append such as exporting via __init__.py?
#import acbllib.acbllib
#import streamlitlib.streamlitlib
#import chatlib.chatlib
#import mlBridgeLib.mlBridgeLib
sys.path.append(str(pathlib.Path.cwd().joinpath('acbllib')))  # global
sys.path.append(str(pathlib.Path.cwd().joinpath('chatlib')))  # global
sys.path.append(str(pathlib.Path.cwd().joinpath('mlBridgeLib')))  # global
sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))  # global
# streamlitlib, mlBridgeLib, chatlib must be placed after sys.path.append. vscode re-format likes to move them to the top
import acbllib
import streamlitlib # must be placed after sys.path.append. vscode re-format likes to move this to the top
import mlBridgeLib # must be placed after sys.path.append. vscode re-format likes to move this to the top
import chatlib  # must be placed after sys.path.append. vscode re-format likes to move this to the top

# override pandas display options
# mlBridgeLib.pd_options_display()

rootPath = pathlib.Path('.')  # e:/bridge/data')
acblPath = rootPath.joinpath('data')  # was 'acbl'
savedModelsPath = rootPath.joinpath('SavedModels')

# pd.options.display.float_format = lambda x: f"{x:.2f}" doesn't work with streamlit

# todo: obsolete in favor of complete_messages

async def create_chat_completion(messages, model=DEFAULT_AI_MODEL, functions=None, function_call='auto', temperature=DEFAULT_AI_MODEL_TEMPERATURE, response_format={"type":"json_object"}):
    return await openai_async_client.chat.completions.create(messages=messages, model=model, functions=functions, function_call=function_call, temperature=temperature, response_format=response_format if model.startswith('gpt-4-') else None)


def ask_database(query):
    print_to_log_info('ask_database query:', query)
    conn = st.session_state.conn
    #"""Function to query duckdb database with a provided SQL query."""
    try:
        results = conn.execute(query)
    except Exception as e:
        results = f"query failed with error: {e}"
    print_to_log_info('ask_database: results:', results)
    return results


# def execute_function_call(message):
#     # todo: use try except?
#     if message["function_call"]["name"] == "ask_database":
#         query = json.loads(message["function_call"]["arguments"])["query"]
#         results = ask_database(query)
#     else:
#         results = f"Error: function {message['function_call']['name']} does not exist"
#     return results


def prompt_keyword_replacements(s):
    replacement_strings = [
        # todo: generalize {} replacements by using df.columns lookup?
        (r'\{Pair_Direction\}', st.session_state.pair_direction),
        (r'\{Opponent_Pair_Direction\}', st.session_state.opponent_pair_direction),
    ]
    for original, new in replacement_strings:
        s = re.sub(original, new, s.replace(
            '  ', ' '), flags=re.IGNORECASE)
    return s


def chat_up_user(up, messages, function_calls, model=None):
    return asyncio.run(async_chat_up_user({'prompt':up}, messages, function_calls, model))

async def async_chat_up_user(prompt_sql, messages, function_calls, model=None):

    if model is None:
        model = st.session_state.ai_api
    up = prompt_sql['prompt']
    # internal commands
    if up == '/about':
        content = slash_about()
        messages.append({"role": "assistant", "content": up+' '+content})
        prompt_sql['sql'] = content # will always return same sql for same query from now on. Is this what we want?
        return True

    if 'sql' in prompt_sql and prompt_sql['sql']: # already has sql. no need for chat-to-sql call.
        sql_query = prompt_sql['sql']
        sql_query = prompt_keyword_replacements(sql_query)
        if up == '':
            up = sql_query
        else:
            up = prompt_keyword_replacements(up)
        messages.append({"role": "user", "content": up})
        # fake message
        assistant_message = {'function_call':{'name':'ask_database'}}

    else:

        i = len(messages)

        # help out AI by enhancing prompt before calling. Replace undesired characters or replace common phrases with actual column names.
        if up[0] == '"': # escape 'Prefer ...' appending
            enhanced_prompt = ''
        else:
            if 'opponent' in up.lower():  # assumes any prompt containing 'opponent' is a prompt requesting opponent_pair_direction
                enhanced_prompt = f"Prefer appending {st.session_state.opponent_pair_direction} instead of {st.session_state.pair_direction}. "
            else:
                enhanced_prompt = f"Prefer appending {st.session_state.pair_direction} instead of {st.session_state.opponent_pair_direction}. "
            #enhanced_prompt = f"Try appending _Declarer or "+enhanced_prompt+up.replace("'", "").replace('"', '')
        enhanced_prompt += up.replace("'", "").replace('"', '')
        # todo: put this into config file.
        replacement_strings = [
            (r'boards i declared', 'Boards_I_Declared = True'),
            (r'boards i played', 'Boards_I_Played = True'),
            (r'boards we played', 'Boards_We_Played = True'),
            (r'boards we declared', 'Boards_We_Declared = True'),
            (r'my boards', 'Boards_I_Played = True'),
            (r'boards partner declared', 'Boards_Partner_Declared = True'),
            (r'boards my partner declared', 'Boards_Partner_Declared = True'),
            (r'boards opponent declared', 'Boards_Opponent_Declared = True'),
            # todo: generalize {} replacements by columnn lookup?
            (r'\{Pair_Direction\}', st.session_state.pair_direction),
            (r'\{Opponent_Pair_Direction\}', st.session_state.opponent_pair_direction),
        ]
        for original, new in replacement_strings:
            enhanced_prompt = re.sub(original, new, enhanced_prompt.replace(
                '  ', ' '), flags=re.IGNORECASE)
        print_to_log_info('enhanced_prompt:', enhanced_prompt)
        # add enhanced prompt to messages
        messages.append({"role": "user", "content": enhanced_prompt})

        # request chat completion of user message
        chat_response = await create_chat_completion( # chat_completion_request(
            messages, model, function_calls)  # chat's response from user input
        print_to_log_info('chat_response status:', type(chat_response), chat_response)
        chat_response_json = json.loads(chat_response.model_dump_json()) # create_chat_completion returns json directly
        print_to_log_info('chat_response_json:', type(chat_response_json), chat_response_json)
        print_to_log_info('chat_response_json id:', type(chat_response_json['id']), chat_response_json['id'])
        print_to_log_info('chat_response_json choices:', type(chat_response_json['choices']), chat_response_json['choices'])

        # restore original user prompt
        messages[-1] = {"role": "user", "content": up}

        if "choices" not in chat_response_json or not isinstance(chat_response_json['choices'], list) or len(chat_response_json['choices']) == 0:
            # fake message
            if 'error' in chat_response_json and 'message' in chat_response_json['error']:
                messages.append(
                    {"role": "assistant", "content": chat_response_json['error']['message']})
            else:
                messages.append(
                    {"role": "assistant", "content": f"Unexpected response from {model} (missing choices or zero length choices). Try again later."})
            return False
        # chat's first and best response message.
        first_choice = chat_response_json["choices"][0]
        if 'message' not in first_choice:
            # fake message
            messages.append(
                {"role": "assistant", "content": f"Unexpected response from {model} (missing message). Try again later."})
            return False
        assistant_message = first_choice['message']
        print_to_log_info('assistant_message:', assistant_message)
        if 'role' not in assistant_message or assistant_message['role'] != 'assistant':
            # fake message
            messages.append(
                {"role": "assistant", "content": f"Unexpected response from {model} (missing choices[0].role or unexpected role). Try again later."})
            return False
        if 'content' not in assistant_message:  # content of None is ok
            # fake message
            messages.append(
                {"role": "assistant", "content": f"Unexpected response from {model} (missing choices[0].content). Try again later."})
            return False
        if "function_call" not in assistant_message:
            assert first_choice['finish_reason'] == 'stop'
            if assistant_message["message"]['content'][0] == '{': # added for 1106 response_format={"type":"json_object"}
                try:
                    function_call_json = json.loads(
                        assistant_message["message"]["content"].replace('\n',''))  # rarely, but sometimes, there are newlines in the json.
                except Exception as e:
                    print_to_log_info(f"Exception: Invalid JSON. Error: {e}")
                    # fake message
                    messages.append(
                        {"role": "assistant", "content": f"Invalid JSON. Error: {e}"})
                    return False
                assert 'query' in function_call_json
                sql_query = function_call_json['query']
            else:
                # ?="} is a lookahead assertion
                # must remove newlines for regex to work
                match = re.search(r'SELECT .*(?="})?$',
                                assistant_message['content'].replace('\n', ''))
                if match is None:
                    messages.append(
                        {"role": "assistant", "content": assistant_message['content']})
                    # fake message
                    messages.append(
                        {"role": "assistant", "content": f"Unexpected response from {model} (missing function_call). Try again later."})
                    return False
                sql_query = match[0]
        else:
            if first_choice['finish_reason'] == 'length':
                # fake message
                messages.append(
                    {"role": "assistant", "content": f"Unexpected finish_reason from {model} ({first_choice['finish_reason']}). Try again later."})
                return False
            assert first_choice['finish_reason'] == 'function_call'
            if 'name' not in assistant_message["function_call"] or assistant_message["function_call"]['name'] != 'ask_database':
                # fake message
                messages.append(
                    {"role": "assistant", "content": f"Unexpected response from {model} (missing choices[0].function_call or unexpected name). Try again later."})
                return False
            if 'arguments' not in assistant_message["function_call"]:
                # fake message
                messages.append(
                    {"role": "assistant", "content": f"Unexpected response from {model} (missing choices[0].function_call.arguments). Try again later."})
                return False
            if assistant_message["function_call"]['arguments'][0] == '{':
                try:
                    function_call_json = json.loads(
                        assistant_message["function_call"]["arguments"].replace('\n',''))  # rarely, but sometimes, there are newlines in the json.
                except Exception as e:
                    print_to_log_info(f"Exception: Invalid JSON. Error: {e}")
                    # fake message
                    messages.append(
                        {"role": "assistant", "content": f"Invalid JSON. Error: {e}"})
                    return False
                assert 'query' in function_call_json
                sql_query = function_call_json['query']
            else:
                # here's hoping it's a SELECT or other SQL statement
                sql_query = assistant_message["function_call"]['arguments']

    # todo: execute via function call, not explicitly
    ask_database_results = ask_database(sql_query)
    print_to_log_info('ask_database_results:', ask_database_results)
    if not isinstance(ask_database_results, duckdb.DuckDBPyConnection):
        # fake message
        messages.append(
            {"role": "assistant", "content": ask_database_results})
        return False
    df = ask_database_results.df()
    df.index.name = 'Row'
    st.session_state.dataframes[sql_query].append(df)

    if 'function_call' in assistant_message:
        messages.append(
            {"role": "function", "name": assistant_message["function_call"]["name"], "content": sql_query})  # todo: what is the content suppose to be? and elsewhere?
    else:
        messages.append({"role": "assistant", "content": sql_query})
    
    prompt_sql['sql'] = sql_query # will always return same sql for same query from now on. Is this what we want?
    print_to_log_info('prompt_sql:', prompt_sql)

    return True


def get_club_results_from_acbl_number(acbl_number):
    return acbllib.get_club_results_from_acbl_number(acbl_number)


def get_tournament_sessions_from_acbl_number(acbl_number, acbl_api_key):
    return acbllib.get_tournament_sessions_from_acbl_number(acbl_number, acbl_api_key)


def get_tournament_session_results(session_id, acbl_api_key):
    return acbllib.get_tournament_session_results(session_id, acbl_api_key)


def create_club_dfs(player_number, event_url):
    return chatlib.create_club_dfs(player_number, event_url)


def merge_clean_augment_club_dfs(dfs, sd_cache_d, player_number):
    return chatlib.merge_clean_augment_club_dfs(dfs, sd_cache_d, player_number)


def merge_clean_augment_tournament_dfs(dfs, dfs_results, sd_cache_d, player_number):
    return chatlib.merge_clean_augment_tournament_dfs(dfs, dfs_results, sd_cache_d, player_number)


# no caching because of hashing parameter concerns
def Augment_Single_Dummy(df, sd_cache_d, sd_observations, match_point_ns_d):
    return chatlib.Augment_Single_Dummy(df, sd_cache_d, sd_observations, match_point_ns_d)


def create_schema_string(df, conn):

    if True:
        df_dtypes_d = {}
        dtypes_d = defaultdict(list)
        complex_objects = []
        for col in df.columns:
            assert col not in df_dtypes_d, col
            dtype_name = df[col].dtype.name
            if dtype_name == 'object':
                if isinstance(df[col].iloc[0], list) or isinstance(df[col].iloc[0], dict):
                    complex_objects.append(col)
                    continue
                dtype_name = 'string'
                df[col] = df[col].astype(dtype_name)
            elif dtype_name == 'uint8':
                df[col] = pd.to_numeric(df[col]) # errors='ignore' removed because its being deprecated
            df_dtypes_d[col] = dtype_name
            dtypes_d[dtype_name].append(col)
        for obj in complex_objects:
            print_to_log_debug(str(obj), df[obj].iloc[0])
        df.drop(columns=complex_objects, inplace=True)

        # warning: fake sql CREATE TABLE because types are dtypes not sql types.
        #df_schema_string = f'CREATE TABLE "results" ({",".join([n+" "+t.name for n,t in zip(df.columns,df.dtypes)])})' # using f' not f"
        df_schema_string = 'CREATE TABLE "results" (\n'+',\n'.join(df.columns)+'\n)' # df.sort_values(key=lambda col: col.str.lower())?

    else:

        conn.execute("CREATE TABLE my_table AS SELECT * FROM results LIMIT 1")

        # show table info. must be after conn.register.
        # create df of all tables known to duckdb
        st.session_state.df_meta = conn.execute("SHOW ALL TABLES").df()

        # st.session_state.df_schema_string = '\n'.join([f"Table:{table_name}\nColumns: {', '.join(['('+table_name+'.'+n+','+t+')' for n,t in zip(column_names,column_types)])}" for table_name,
        #                                                     column_names, column_types in zip(st.session_state.df_meta["name"], st.session_state.df_meta["column_names"], st.session_state.df_meta["column_types"])])
        '''
            -- example CREATE TABLE
            CREATE TABLE "events" (
            "id" INT NOT NULL PRIMARY KEY,
            "created_at" VARCHAR NOT NULL,
            "updated_at" VARCHAR NOT NULL,
            "club_tournament_id" INT NULL,
            -- list of VARCHAR
            FOREIGN KEY ("rounds") REFERENCES "rounds"(id) ON DELETE NO ACTION
            );
        '''
        # todo: would chatgpt be more effective passing a CREATE TABLE instead of textual description of table?
        # st.session_state.df_schema_string = '\n'.join([f'Table:"{table_name}\nColumns: {", ".join(["("+n+","+t+")" for n,t in zip(column_names,column_types)])}' for table_name,
        #                                                     column_names, column_types in zip(st.session_state.df_meta["name"], st.session_state.df_meta["column_names"], st.session_state.df_meta["column_types"])]) # using f' not f"
        df_schema_string = '\n'.join([f'CREATE TABLE "{table_name}" (\n{",".join([n+" "+t for n,t in zip(column_names,column_types)])}' for table_name,
                                            column_names, column_types in zip(st.session_state.df_meta['name'], st.session_state.df_meta['column_names'], st.session_state.df_meta['column_types'])]) # using f' not f"

    return df_schema_string


def chat_initialize(player_number, session_id): # todo: rename to session_id?

    print_to_log_info(f"Retrieving latest results for {player_number}")

    conn = st.session_state.conn

    with st.spinner(f"Retrieving a list of games for {player_number} ..."):
        t = time.time()
        game_urls = get_club_results_from_acbl_number(player_number)
        if game_urls is None:
            st.error(f"Player number {player_number} not found.")
            return False
        if len(game_urls) == 0:
            st.error(f"Could not find any club games for {player_number}.")
        elif session_id is None:
            session_id = list(game_urls.keys())[0]  # default to most recent club game
        print_to_log_info('get_club_results_from_acbl_number time:', time.time()-t) # takes 4s

    with st.spinner(f"Retrieving a list of tournament sessions for {player_number} ..."):
        t = time.time()
        tournament_session_urls = get_tournament_sessions_from_acbl_number(player_number, acbl_api_key) # returns [url, url, description, dfs]
        if tournament_session_urls is None:
            st.error(f"Player number {player_number} not found.")
            return False
        if len(tournament_session_urls) == 0:
            st.error(f"Could not find any tournament sessions for {player_number}.")
        elif session_id is None:
            session_id = list(tournament_session_urls.keys())[0]  # default to most recent tournament session
        print_to_log_info('get_tournament_sessions_from_acbl_number time:', time.time()-t) # takes 2s

    if session_id is None:
        st.error(f"Could not find any club or tournament sessions for {player_number}.")
        return False
    
    reset_data()
    st.session_state.player_number = player_number
    st.session_state.game_urls = game_urls
    st.session_state.tournament_session_urls = tournament_session_urls

    if session_id in game_urls:
        with st.spinner(f"Collecting data for club game {session_id} and player {player_number}."):
            t = time.time()
            # game_urls[session_id][1] is detail_url
            dfs = create_club_dfs(player_number, game_urls[session_id][1])
            if dfs is None or 'event' not in dfs or len(dfs['event']) == 0:
                st.error(
                    f"Game {session_id} has missing or invalid game data. Select a different club game or tournament session from left sidebar.")
                return False
            print_to_log_info('dfs:',dfs.keys())

            # todo: probably need to check if keys exist to control error processing -- pair_summaries, event, sessions, ...

            if dfs['pair_summaries']['pair_number'].value_counts().eq(1).all(): # Assuming pair_numbers are all unique for Howell
                st.error(
                    f"Game {session_id}. I can only chat about Mitchell movements. Select a different club game or tournament session from left sidebar.")
                return False

            if dfs['event']['type'].iloc[0] != 'PAIRS':
                st.error(
                    f"Game {session_id} is {dfs['event']['type'].iloc[0]}. Expecting an ACBL pairs match point game. Select a different club game or tournament session from left sidebar.")
                return False

            if dfs['event']['board_scoring_method'].iloc[0] != 'MATCH_POINTS':
                st.error(
                    f"Game {session_id} is {dfs['event']['board_scoring_method'].iloc[0]}. Expecting an ACBL pairs match point game. Select a different club game or tournament session from left sidebar.")
                return False

            if not dfs['sessions']['hand_record_id'].iloc[0][0].isdigit():
                st.error(
                    f"Game {session_id} is {dfs['sessions']['hand_record_id'].iloc[0]}. Expecting a valid hand record number. Select a different club game or tournament session from left sidebar.")
                return False
            print_to_log_info('create_club_dfs time:', time.time()-t) # takes 3s

        with st.spinner(f"Processing data for club game: {session_id} and player {player_number}. Takes 30 seconds ..."):
            t = time.time()
            df, sd_cache_d, matchpoint_ns_d = merge_clean_augment_club_dfs(dfs, {}, player_number) # doesn't use any caching
            if df is None:
                st.error(
                    f"Game {session_id} has an invalid game file. Select a different club game or tournament session from left sidebar.")
                return False
            print_to_log_info('merge_clean_augment_club_dfs time:', time.time()-t) # takes 30s

    elif session_id in tournament_session_urls:
        dfs = tournament_session_urls[session_id][3]
        if dfs is None or 'event' not in dfs or len(dfs['event']) == 0:
            st.error(
                f"Session {session_id} has missing or invalid session data. Choose another session.")
            return False
        print_to_log_info(dfs.keys())

        if dfs['event']['game_type'] != 'Pairs':
            st.error(
                f"Session {session_id} is {dfs['event']['game_type']}. Expecting an ACBL pairs match point session. Choose another session.")
            return False

        if dfs['score_score_type'] != 'Matchpoints':
            st.error(
                f"Session {session_id} is {dfs['score_score_type']}. Expecting an ACBL pairs match point session. Choose another session.")
            return False

        with st.spinner(f"Collecting data for tournament session {session_id} and player {player_number} from ACBL."):
            t = time.time()

            response = get_tournament_session_results(session_id, acbl_api_key)
            assert response.status_code == 200, response.status_code
            dfs_results = response.json()
            if dfs_results is None:
                st.error(
                    f"Session {session_id} has an invalid tournament session file. Choose another session.")
                return False
            print_to_log_info('dfs_results:',dfs_results.keys())

            if len(dfs_results['sections']) == 0:
                st.error(
                    f"Session {session_id} has no sections. Choose another session.")
                return False

            if 'handrecord' not in dfs_results or len(dfs_results['handrecord']) == 0 or 'box_number' not in dfs_results or not dfs_results['box_number'].isdigit():
                st.error(
                    f"Session {session_id} has a missing hand record. Cannot chat about shuffled sessions. Choose another session.")
                return False

            for section in dfs_results['sections']: # is it better/possible to only examine the section which the player played in?

                if section['scoring_type'] != 'Matchpoints':
                    st.error(
                        f"Session {session_id} section {section['section_label']} is {section['scoring_type']}. Expecting an ACBL pairs match point session. Choose another session.")
                    return False

                if section['movement_type'] != 'Mitchell':
                    st.error(
                        f"Session {session_id} section {section['section_label']} is {section['movement_type']}. I can only chat about Mitchell movements. Choose another session.")
                    return False
            print_to_log_info('get_tournament_session_results time:', time.time()-t)

        with st.spinner(f"Processing data for tournament session {session_id} for player {player_number}. Takes 30 seconds ..."):
            t = time.time()
            #with Profiler():

            df, sd_cache_d, matchpoint_ns_d = merge_clean_augment_tournament_dfs(tournament_session_urls[session_id][3], dfs_results, acbl_api_key, player_number) # doesn't use any caching
            if df is None:
                st.error(
                    f"Session {session_id} has an invalid tournament session file. Choose another session.")
                return False
            print_to_log_info('merge_clean_augment_tournament_dfs time:', time.time()-t)

        st.session_state.dfs_results = dfs_results
    else:
        assert False, f"session_id not found: {session_id}"

    # game appears valid. Save it.
    st.session_state.session_id = session_id

    with st.spinner(f"Creating everything data table."):
        t = time.time()
        results = ask_database(st.session_state.commands_sql)
        assert isinstance(results, duckdb.DuckDBPyConnection), results
        df = results.df()  # update df with results of SQL query.
        assert df is not None
        move_to_front = ['Board', 'Contract', 'Result', 'Tricks', 'Score_NS',
                            'Pct_NS', 'ParScore_NS']  # arbitrary list of columns to show first
        df = df[move_to_front + (df.columns.drop(move_to_front).tolist())]
        df.index.name = 'Row'
        st.session_state.df = df
        st.session_state.matchpoint_ns_d = matchpoint_ns_d

        # extract scalers
        st.session_state.game_date = pd.to_datetime(st.session_state.df['Date'].iloc[0]).strftime(
            '%Y-%m-%d')
        assert st.session_state.df['event_id'].eq(session_id).all()
        # my_table_df = pd.DataFrame(pd.concat([dfs['event'],dfs['club']],axis='columns')) # single row of invariant data
        # my_table_df['Date'] = st.session_state.game_date # temp?
        for player_direction, pair_direction, partner_direction, opponent_pair_direction in [('North', 'NS', 'S', 'EW'), ('South', 'NS', 'N', 'EW'), ('East', 'EW', 'W', 'NS'), ('West', 'EW', 'E', 'NS')]:
            rows = df[df[f"Player_Number_{player_direction[0]}"].str.contains(
                player_number)]
            if len(rows) > 0:
                st.session_state.player_number = player_number
                st.session_state.player_direction = player_direction
                st.session_state.pair_direction = pair_direction
                st.session_state.partner_direction = partner_direction
                st.session_state.opponent_pair_direction = opponent_pair_direction
                section_ids = rows['section_id']
                assert section_ids.nunique() == 1, f"Oops. section_id non-unique."
                st.session_state.section_id = section_ids.iloc[0]
                section_names = rows['section_name']
                assert section_names.nunique() == 1, f"Oops. section_name non-unique."
                st.session_state.section_name = section_names.iloc[0]
                player_names = rows[f"Player_Name_{player_direction[0]}"]
                assert player_names.nunique() == 1, f"Oops. player_names non-unique."
                st.session_state.player_name = player_names.iloc[0]
                pair_numbers = rows[f"Pair_Number_{pair_direction}"]
                assert pair_numbers.nunique() == 1,  f"Oops. pair_numbers non-unique."
                st.session_state.pair_number = pair_numbers.iloc[0]
                partner_numbers = rows[f"Player_Number_{partner_direction}"]
                assert partner_numbers.nunique() == 1, f"Oops. partner_numbers non-unique."
                st.session_state.partner_number = partner_numbers.iloc[0]
                partner_names = rows[f"Player_Name_{partner_direction}"]
                assert partner_names.nunique() == 1, f"Oops. partner_names non-unique."
                st.session_state.partner_name = partner_names.iloc[0]

                # hack: this dopey hack works really well! chatgpt gets much less confused.
                # create columns for these scalers just to make it easier to use them in SQL queries.
                #df['My_Player_Number'] = st.session_state.player_number
                #df['My_Player_Name'] = st.session_state.player_name
                #df['My_Player_Direction'] = st.session_state.player_direction
                #df['My_Pair_Direction'] = st.session_state.pair_direction
                #df['My_Pair_Number'] = st.session_state.pair_number
                #df['My_Partner_Number'] = st.session_state.partner_number
                #df['My_Partner_Name'] = st.session_state.partner_name
                #df['My_Partner_Direction'] = st.session_state.partner_direction
                df['Opponent_Pair_Direction'] = st.session_state.opponent_pair_direction
                df['My_Section'] = df['section_name'].eq(
                    st.session_state.section_name)
                df['Our_Section'] = df['section_name'].eq(
                    st.session_state.section_name)
                #df['Players'] = df.apply(lambda r: [r[f"Player_Number_{d}"] for d in 'NESW'],axis='columns')
                df['My_Pair'] = df['My_Section'] & df[f"Pair_Number_{st.session_state.pair_direction}"].eq(
                    st.session_state.pair_number)  # boolean
                df['Our_Pair'] = df['My_Section'] & df[f"Pair_Number_{st.session_state.pair_direction}"].eq(
                    st.session_state.pair_number)  # boolean # obsolete?
                df['Boards_I_Played'] = df['My_Pair']  # boolean # obsolete?
                df['Boards_We_Played'] = df['My_Pair']  # boolean
                df['Our_Boards'] = df['My_Pair']  # boolean # obsolete?
                df['Boards_I_Declared'] = df['My_Pair'] & df['Number_Declarer'].eq(
                    st.session_state.player_number)  # boolean
                df['Boards_We_Declared'] = df['My_Pair'] & df['Number_Declarer'].isin(
                    [st.session_state.player_number, st.session_state.partner_number])  # boolean
                df['Boards_Partner_Declared'] = df['My_Pair'] & df['Number_Declarer'].eq(
                    st.session_state.partner_number)  # boolean
                df['Partners_Boards'] = df['My_Pair'] & df['Number_Declarer'].eq(
                    st.session_state.partner_number)  # boolean
                df['Boards_Opponent_Declared'] = df['My_Pair'] & ~df['Number_Declarer'].isin(
                    [st.session_state.player_number, st.session_state.partner_number])  # boolean
                break
        print_to_log_info('create everything data table time:', time.time()-t)

    # make predictions
    with st.spinner(f"Making AI Predictions. Takes 15 seconds."):
        t = time.time()
        df = Predict_Game_Results(df) # returning updated df for conn.register()
        print_to_log_info('Predict_Game_Results time:', time.time()-t) # takes 10s

    # Create a DuckDB table from the DataFrame
    # register df as a table named 'results' for duckdb discovery. SQL queries will reference this df/table.
    conn.register('results', df)
    #conn.register('my_table', df)

    st.session_state.df_schema_string = create_schema_string(df, conn)
    # temp
    with open('df_schema.sql','w') as f:
        f.write(st.session_state.df_schema_string)

    function_calls = [
        {
            "name": "ask_database",
            # from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb
            # Needs to explictly be told to return a SQL string otherwise returns json.
            "description": "Use this function to answer user questions about duplicate bridge statistics. Output should be a fully formed SQL query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"""
                                SQL query for extracting info to answer the user"s question.
                                SQL should be written using this database schema:
                                {st.session_state.df_schema_string}
                                The schema contains table name, column name and column type.
                                The returned value should be a plain text SQL query embedded in JSON.
                                """,
                    }
                },
                "required": ["query"],
            },
        }
    ]
    st.session_state.function_calls = function_calls
    reset_messages()

    #content = slash_about()
    #streamlit_chat.message(f"Morty: {content}", logo=st.session_state.assistant_logo)

    if st.session_state.show_sql_query:
        t = time.time()
        streamlit_chat.message(
            f"Morty: Here's a dataframe of game results. There's {len(df)} rows and {len(df.columns)} columns.", logo=st.session_state.assistant_logo)
        streamlitlib.ShowDataFrameTable(df, key='clear_conversation_game_data_df', tooltips=st.session_state.dataframe_tooltips)
        print_to_log_info('ShowDataFrameTable time:', time.time()-t)

    return True


def slash_about():
    content = f"Hey {st.session_state.player_name} ({st.session_state.player_number}), let's chat about your game on {st.session_state.game_date} (event id {st.session_state.session_id}). Your pair was {st.session_state.pair_number}{st.session_state.pair_direction} in section {st.session_state.section_name}. You played {st.session_state.player_direction}. Your partner was {st.session_state.partner_name} ({st.session_state.partner_number}) who played {st.session_state.partner_direction}."
    return content


def ask_questions_without_context(ups, model=None):
    # pandasai doesn't work. context length is limited to 4100 tokens. need 8k?
    # llm = OpenAI(api_token=openai.api_key)
    # df = st.session_state.df
    # sdf = SmartDataframe(df, config={"llm": llm})
    # #sdf = SmartDataframe(df)
    # qdf = sdf.chat("Show board, contract")
    # print_to_log('qdf:', qdf)

    if model is None:
        model = st.session_state.ai_api
    function_calls = st.session_state.function_calls
    # ups can be a string, list of strings, or list of lists of strings.
    assert isinstance(ups, list), ups
    with st.spinner(f"Morty is judging you ..."): # {len(ups)} responses from {model}."):
        tasks = []
        list_of_new_messages = []
        for i, up in enumerate(ups):
            # pass system prompt only
            new_messages = [st.session_state.messages[0]]
            list_of_new_messages.append(new_messages)
            tasks.append(async_chat_up_user(up, new_messages, function_calls, model))
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(asyncio.gather(*tasks))
            #for result in results:
            #    print_to_log(result)
        finally:
            loop.close()
        for messages in list_of_new_messages:
            # append all new messages to list except for system prompt
            st.session_state.messages.extend(messages[1:])


def ask_a_question_with_context(ups, model=None):
    t = time.time()
    if model is None:
        model = st.session_state.ai_api
    messages = st.session_state.messages
    # removed because no longer an issue?
    #if model != DEFAULT_LARGE_AI_MODEL:
    #    if len(messages) > 12:
    #        messages = messages[:1+3]+messages[1+3-10:]
    function_calls = st.session_state.function_calls
    if isinstance(ups, list):
        for i, up in enumerate(ups):
            with st.spinner(f"Waiting for response {i} of {len(ups)} from {model}."):
                chat_up_user(up, messages, function_calls, model)
    else:
        with st.spinner(f"Waiting for response from {model}."):
            chat_up_user(ups, messages, function_calls, model)
    st.session_state.messages = messages
    print_to_log_info('ask_a_question_with_context time:', time.time()-t)


def reset_messages():
    assert st.session_state.player_number is not None, "Oops. Player number is None."
    assert st.session_state.system_prompt is not None, "Oops. System prompt is None."
    augmented_system_prompt = st.session_state.system_prompt
    #augmented_system_prompt += f" Player Number is '{st.session_state.player_number}'."
    #augmented_system_prompt += f" Player Name is '{st.session_state.player_name}'."
    # todo: howell, individual, team not implemented!
    #augmented_system_prompt += f" Game Date is always {st.session_state.game_date}."
    #augmented_system_prompt += f" Event ID is always {st.session_state.session_id}."
    #augmented_system_prompt += f" My partner's Player Direction is always {st.session_state.partner_direction[0]}."
    #augmented_system_prompt += f" My Player Direction is always {st.session_state.player_direction[0]}."
    #augmented_system_prompt += f" My partner's Player Direction is always {st.session_state.partner_direction[0]}."
    #augmented_system_prompt += f" My partner's Player Number is always '{st.session_state.partner_number}'."
    #augmented_system_prompt += f" My partner's Player Name is always '{st.session_state.partner_name}'."
    #augmented_system_prompt += f" My Pair Direction is always {st.session_state.pair_direction}."
    #augmented_system_prompt += f" My 'Pair Number {st.session_state.opponent_pair_direction}' is always {st.session_state.pair_number}."
    #augmented_system_prompt += f" Opponent Pair Direction is always {st.session_state.opponent_pair_direction}."
    #augmented_system_prompt += f" My Section Id is always {st.session_state.section_id}."
    #augmented_system_prompt += f" My Section Name is always {st.session_state.section_name}."
    #augmented_system_prompt += f" \"My boards means\" boards having a Declarer Number of always {st.session_state.player_number}."
    #augmented_system_prompt += f" \"Our boards\" means boards having a Pair Number always {st.session_state.pair_direction} of {st.session_state.pair_number}."
    #augmented_system_prompt += f" \"Boards I declared\" means boards having my Declarer Number of always {st.session_state.player_number}."
    #augmented_system_prompt += f" \"Boards my partner declared\" means boards having my Declarer Number of always {st.session_state.partner_number}"
    #augmented_system_prompt += f" \"Boards we declared\" means boards having a Declarer Number of always {st.session_state.player_number} of {st.session_state.partner_number}."
    #augmented_system_prompt += f" \"Boards we played\" means boards having a Pair Number always {st.session_state.pair_direction} of {st.session_state.pair_number}."
    st.session_state.augmented_system_prompt = augmented_system_prompt
    system_message = {"role": "system",
                      "content": st.session_state.augmented_system_prompt}
    messages = [system_message]
    st.session_state.messages = messages


def player_number_change():
    # assign changed textbox value (player_number_input) to player_number
    player_number = st.session_state.player_number_input
    if not chat_initialize(player_number, None):
        pass


def debug_player_number_names_change():
    # assign changed selectbox value (debug_player_number_names_selectbox). e.g. ['2663279','Robert Salita']
    player_number_name = st.session_state.debug_player_number_names_selectbox
    #if not chat_initialize(player_number_name[0], None):  # grab player number
    #    chat_initialize(st.session_state.player_number, None)
    chat_initialize(player_number_name[0], None)


def club_session_id_change():
    #st.session_state.tournament_session_ids_selectbox = None # clear tournament index whenever club index changes. todo: doesn't seem to update selectbox with new index.
    session_id = int(st.session_state.club_session_ids_selectbox.split(',')[0]) # split selectbox item on commas. only want first split.
    if not chat_initialize(st.session_state.player_number, session_id):
        chat_initialize(st.session_state.player_number, None)


def tournament_session_id_change():
    #st.session_state.club_session_ids_selectbox = None # clear club index whenever tournament index changes. todo: doesn't seem to update selectbox with new index.
    tournament_session_id = st.session_state.tournament_session_ids_selectbox.split(',')[0] # split selectbox item on commas. only want first split.
    if not chat_initialize(st.session_state.player_number, tournament_session_id):
        chat_initialize(st.session_state.player_number, None)


def show_sql_query_change():
    # toggle whether to show sql query
    st.session_state.show_sql_query = st.session_state.sql_query_checkbox


def ai_api_selectbox_change():
    # assign changed selectbox value (ai_api_selectbox) to ai_api
    st.session_state.ai_api = st.session_state.ai_api_selectbox


def prompts_selectbox_change():
    if st.session_state.prompts_selectbox is not None:
        title = st.session_state.prompts_selectbox
        if st.session_state.vetted_prompt_titles is not None: # this fixes the situation when an unsupported game event is selected.
            box = st.session_state.vetted_prompt_titles[title]
            ups = box['prompts']
            if len(ups):
                ask_questions_without_context(ups, st.session_state.ai_api)
            read_favorites()


def sd_observations_changed():
    st.session_state.sd_observations = st.session_state.sd_observations_number_input
    with st.spinner(f"Calculating single dummy probabilities using {st.session_state.sd_observations} trials per board. {st.session_state.df['Board'].nunique()} boards. Please wait..."):
        #with Profiler():
        st.session_state.df, st.session_state.sd_cache_d = Augment_Single_Dummy(
            st.session_state.df, {}, st.session_state.sd_observations, st.session_state.matchpoint_ns_d) # using {} to disable cache
        # must reregister if df is changed!
        st.session_state.conn.register('results', st.session_state.df)
        # todo: experimenting with outputing a dataframe of some SD relevant columns
        streamlitlib.ShowDataFrameTable(st.session_state.df[['Board', 'PBN', 'Pair_Direction_Declarer', 'Direction_Declarer', 'BidSuit']+st.session_state.df.filter(regex=r'^SD').columns.to_list(
        )].sort_values(['Board'], tooltips=st.session_state.dataframe_tooltips).drop_duplicates(subset=['Board', 'PBN', 'Pair_Direction_Declarer', 'Direction_Declarer', 'BidSuit']), key='sd_observations_changed_sd_df')


import mlBridgeAi

def Predict_Game_Results(df):
    # Predict game results using a saved model.

    if df is None:
        return None

    club_or_tournament = 'club' if st.session_state.session_id in st.session_state.game_urls else 'tournament'

    df['Declarer_Rating'] = df['Declarer_Rating'].fillna(.5) # todo: NS sitout. Why is this needed? Are empty opponents required to have a declarer rating? Event id: 893775.

    # create columns from model's predictions.
    predicted_contracts_model_filename = f"acbl_{club_or_tournament}_predicted_contract_fastai_model.pkl"
    predicted_contracts_model_file = savedModelsPath.joinpath(predicted_contracts_model_filename)
    print_to_log_info('predicted_contract_model_file:',predicted_contracts_model_file)
    if not predicted_contracts_model_file.exists():
        st.error(f"Oops. {predicted_contracts_model_filename} not found.")
        return None
    # todo: not needed right now. However, need to change *_augment.ipynb to output ParScore_MPs_(NS|EW) df['ParScore_MPs'] = df['ParScore_MPs_NS']
    learn = mlBridgeAi.load_model(predicted_contracts_model_file)
    print_to_log_debug('isna:',df.isna().sum())
    contracts_all = ['PASS']+[str(level+1)+strain+dbl+direction for level in range(7) for strain in 'CDHSN' for dbl in ['','X','XX'] for direction in 'NESW']
    #df['Contract'] = df['Contract'].astype('category',categories=contracts_all)
    #df['Contract'] = df['Contract'].astype('string')
    #print(df['Contract'])
    #df = df.drop(df[~df['Contract'].isin(learn.dls.vocab)].index)
    assert df['Contract'].isin(mlBridgeLib.contract_classes).all(), df['Contract'][~df['Contract'].isin(mlBridgeLib.contract_classes)]
    #df[learn.dls.y_names[0]] = pd.Categorical(df[learn.dls.y_names[0]], categories=learn.dls.vocab)
    #import pickle
    #save_df_filename = "app_df.pkl"
    #save_df_file = savedModelsPath.joinpath(save_df_filename)
    #with open(save_df_file, 'wb') as f:
    #    pickle.dump(df,f)
    #print(f"Saved {save_df_filename}: size:{save_df_file.stat().st_size}")
    pred_df = mlBridgeAi.get_predictions(learn, df) # classifier returns list containing a probability for every class label (NESW)
    df = pd.concat([df,pred_df],axis='columns')
    print(df)

    # create columns from model's predictions.
    predicted_directions_model_filename = f"acbl_{club_or_tournament}_predicted_declarer_direction_fastai_model.pkl"
    predicted_directions_model_file = savedModelsPath.joinpath(predicted_directions_model_filename)
    print_to_log_info('predicted_declarer_direction_model_file:',predicted_directions_model_file)
    if not predicted_directions_model_file.exists():
        st.error(f"Oops. {predicted_directions_model_file} not found.")
        return None
    # todo: not needed right now. However, need to change *_augment.ipynb to output ParScore_MPs_(NS|EW) df['ParScore_MPs'] = df['ParScore_MPs_NS']
    learn = mlBridgeAi.load_model(predicted_directions_model_file)
    print_to_log_debug('isna:',df.isna().sum())
    #df['Tricks'].fillna(.5,inplace=True)
    #df['Result'].fillna(.5,inplace=True)
    # FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    df['Declarer_Rating'] = df['Declarer_Rating'].fillna(.5) # todo: NS sitout. Why is this needed? Are empty opponents required to have a declarer rating? Event id: 893775.
    # encode categories using original y categories
    #df[learn.dls.y_names[0]] = pd.Categorical(df[learn.dls.y_names[0]], categories=learn.dls.procs.categorify.classes[learn.dls.y_names[0]])
    print(df['Declarer_Direction'])
    #df['Declarer_Direction'] = df['Declarer_Direction'].astype('string')
    print('vocab:',learn.dls.vocab)
    pred_df = mlBridgeAi.get_predictions(learn, df) # classifier returns list containing a probability for every class label (NESW)
    df = pd.concat([df,pred_df],axis='columns')
    y_name = learn.dls.y_names[0]
    print(y_name)
    print(df)
    df['Declarer_Number_Pred'] = df.apply(lambda r: r['Player_Number_'+(r['Dealer'] if r[y_name+'_Pred']=='PASS' else r[y_name+'_Pred'][-1])],axis='columns')
    df['Declarer_Name_Pred'] = df.apply(lambda r: r['Player_Name_'+(r['Dealer'] if r[y_name+'_Pred']=='PASS' else r[y_name+'_Pred'][-1])],axis='columns')
    df['Declarer_Pair_Direction_Match'] = df.apply(lambda r: (r[y_name+'_Actual'] in 'NS') == (r[y_name+'_Pred'] in 'NS'),axis='columns')

    # create columns from model's predictions.
    predicted_rankings_model_filename = f"acbl_{club_or_tournament}_predicted_pct_ns_fastai_model.pkl"
    predicted_rankings_model_file = savedModelsPath.joinpath(predicted_rankings_model_filename)
    print_to_log_info('predicted_pct_ns_model_file:',predicted_rankings_model_file)
    if not predicted_rankings_model_file.exists():
        st.error(f"Oops. {predicted_rankings_model_file} not found.")
        return None
    #y_name = 'Pct_NS'
    #predicted_board_result_pcts_ns, _ = mlBridgeAi.make_predictions(predicted_rankings_model_file, df)
    learn = mlBridgeAi.load_model(predicted_rankings_model_file)
    #df[learn.dls.y_names[0]] = pd.Categorical(df[learn.dls.y_names[0]], categories=learn.dls.procs.categorify.classes[learn.dls.y_names[0]])
    pred_df = mlBridgeAi.get_predictions(learn, df) # classifier returns list containing a probability for every class label (NESW)
    df = pd.concat([df,pred_df],axis='columns')
    y_name = learn.dls.y_names[0]
    print(y_name)
    y_name_ns = y_name
    y_name_ew = y_name.replace('NS','EW')
    df[y_name_ew+'_Actual'] = df[y_name_ew]
    df[y_name_ew+'_Pred'] = 1-df[y_name_ns+'_Pred']
    df[y_name_ns+'_Diff'] = df[y_name_ns+'_Actual']-df[y_name_ns+'_Pred']
    df[y_name_ew+'_Diff'] = df[y_name_ew+'_Actual']-df[y_name_ew+'_Pred']

    return df # return newly created df. created by df = pd.concat().

def read_favorites():

    if st.session_state.default_favorites_file.exists():
        with open(st.session_state.default_favorites_file, 'r') as f:
            favorites = json.load(f)
            st.session_state.favorites = favorites

    if st.session_state.player_number_custom_favorites_file.exists():
        with open(st.session_state.player_number_custom_favorites_file, 'r') as f:
            player_number_favorites = json.load(f)
            st.session_state.player_number_favorites = player_number_favorites

    if st.session_state.debug_favorites_file.exists():
        with open(st.session_state.debug_favorites_file, 'r') as f:
            debug_favorites = json.load(f)
            st.session_state.debug_favorites = debug_favorites


def reset_data():
    # resets all data. used initially and when player number changes.
    # todo: put all session state into st.session_state.data so that clearing data clears all session states.

    print_to_log_info('reset_data()')

    # app
    #st.session_state.app_datetime = None
    st.session_state.help = None
    st.session_state.release_notes = None

    # game
    st.session_state.game_urls = {}
    st.session_state.tournament_session_urls = {}
    st.session_state.tournament_sessions = {}
    st.session_state.game_date = None
    st.session_state.session_id = None
    st.session_state.dfs_results = None

    # chat
    #st.session_state.ai_api = None
    st.session_state.system_prompt = None
    st.session_state.augmented_system_prompt = None
    st.session_state.messages = []
    st.session_state.pdf_link = None
    st.session_state.dataframes = defaultdict(list)
    st.session_state.df = None
    st.session_state.matchpoint_ns_d = None
    st.session_state.function_calls = None

    # sql
    #st.session_state.conn = None
    #st.session_state.show_sql_query = None
    st.session_state.commands_sql = None
    st.session_state.df_meta = None
    st.session_state.df_schema_string = None

    # favorite files
    st.session_state.favorites = None
    st.session_state.default_favorites_file = None
    st.session_state.player_number_favorites = None
    st.session_state.player_number_custom_favorites_file = None
    st.session_state.debug_favorites = None
    st.session_state.debug_favorites_file = None
    st.session_state.prompts_selectbox = 'Choose a Prompt'
    st.session_state.vetted_prompts = None
    st.session_state.vetted_prompt_titles = None
    st.session_state.dataframe_tooltips = None

    # augmented columns
    st.session_state.player_number = None
    st.session_state.player_direction = None
    st.session_state.player_name = None
    st.session_state.partner_number = None
    st.session_state.partner_direction = None
    st.session_state.partner_name = None
    st.session_state.pair_number = None
    st.session_state.pair_direction = None
    st.session_state.pair_name = None
    st.session_state.opponent_pair_direction = None
    st.session_state.section_id = None
    st.session_state.section_name = None

    main_message_df_count = 0
    for k, v in st.session_state.items():
        print_to_log_info('session_state:',k)
        if k.startswith('main_messages_df_'):
            # assert st.session_state[k] is None # This happened once on 29-Sep-2023. Not sure why. Maybe there's a timing issue with st.session_state and st.container being destroyed?
            #del st.session_state[k] # delete the key. This is a hack. It's not clear why the key is not being deleted when the container is destroyed.
            main_message_df_count += 1
    print_to_log_info('main_message_df_: count:',main_message_df_count,st.session_state.df_unique_id)

    # These files are repeatedly reloaded for development purposes. Only takes a second.

    system_prompt_file = pathlib.Path('system_prompt.txt')
    if not system_prompt_file.exists():
        st.write(f"Oops. {system_prompt_file} file does not exist.")
        st.stop()
    with open(system_prompt_file, 'r') as f:
        system_prompt = f.read()  # text string
        st.session_state.system_prompt = system_prompt

    commands_sql_file = pathlib.Path('commands.sql')
    if commands_sql_file.exists():
        with open(commands_sql_file, 'r') as f:
            commands_sql = f.read()  # text string
            st.session_state.commands_sql = commands_sql

    ai_apis_file = pathlib.Path('ai_apis.json')
    if ai_apis_file.exists():
        with open(ai_apis_file, 'r') as f:
            ai_apis = json.load(f)  # dict
            st.session_state.ai_apis = ai_apis['AI_APIs']['Models']
            assert len(st.session_state.ai_apis) > 0 and DEFAULT_AI_MODEL in st.session_state.ai_apis, f"Oops. {DEFAULT_AI_MODEL} not in {st.session_state.ai_apis}."
    else:
        st.session_state.ai_apis = [DEFAULT_AI_MODEL]
    st.session_state.ai_api = DEFAULT_AI_MODEL


def app_info():
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in Streamlit. AI API is OpenAI. Data engine is Pandas. Query engine is Duckdb. Chat UI uses streamlit-chat. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot Club data scraped from public ACBL webpages. Tournament data from ACBL API.")
    # fastai:{fastai.__version__} pytorch:{torch.__version__} safetensors:{safetensors.__version__} sklearn:{sklearn.__version__}
    st.caption(
        f"App:{st.session_state.app_datetime} Python:{'.'.join(map(str, sys.version_info[:3]))} Streamlit:{st.__version__} Pandas:{pd.__version__} duckdb:{duckdb.__version__} Default AI model:{DEFAULT_AI_MODEL} OpenAI client:{openai.__version__} fastai:{fastai.__version__} numpy:{np.__version__} polars:{pl.__version__} safetensors:{safetensors.__version__} sklearn:{sklearn.__version__} torch:{torch.__version__} Query Params:{st.query_params.to_dict()}")


def create_sidebar():
    
    t = time.time()

    st.sidebar.caption(st.session_state.app_datetime)

    st.sidebar.text_input(
        "ACBL player number", on_change=player_number_change, placeholder=st.session_state.player_number, key='player_number_input')

    st.sidebar.selectbox("Choose a club game.", index=0, options=[f"{k}, {v[2]}" for k, v in st.session_state.game_urls.items(
    )], on_change=club_session_id_change, key='club_session_ids_selectbox')  # options are event_id + event description

    st.sidebar.selectbox("Choose a tournament session.", index=None, options=[f"{k}, {v[2]}" for k, v in st.session_state.tournament_session_urls.items(
    )], on_change=tournament_session_id_change, key='tournament_session_ids_selectbox')  # options are event_id + event description

    if st.session_state.session_id is None:
        st.stop()

    if st.session_state.session_id in st.session_state.game_urls:
        launch_acbl_results_page = f"[ACBL Club Result Page]({st.session_state.game_urls[st.session_state.session_id][1]})"
    else:
        launch_acbl_results_page = f"[ACBL Tournament Result Page]({st.session_state.tournament_session_urls[st.session_state.session_id][1]})"
    st.sidebar.markdown(launch_acbl_results_page, unsafe_allow_html=True)

    # These files are releoaded each time for development purposes. Only takes a second.
    # todo: put filenames into a .json or .toml file?
    st.session_state.default_favorites_file = pathlib.Path(
        'default.favorites.json')
    st.session_state.player_number_custom_favorites_file = pathlib.Path(
        'favorites/'+st.session_state.player_number+'.favorites.json')
    st.session_state.debug_favorites_file = pathlib.Path(
        'favorites/debug.favorites.json')
    read_favorites()

    help_file = pathlib.Path('help.md')
    if help_file.exists():
        with open(help_file, 'r') as f:
            st.session_state.help = f.read()  # text string

    release_notes_file = pathlib.Path('release_notes.md')
    if release_notes_file.exists():
        with open(release_notes_file, 'r') as f:
            st.session_state.release_notes = f.read()  # text string

    if st.sidebar.button("Clear Conversation", key='clear_chat_button'):
        reset_messages()
        streamlitlib.move_focus()

    st.session_state.pdf_link = st.sidebar.empty()

    st.sidebar.divider()
    st.sidebar.write(
        'Below are favorite prompts. Either click a button below or enter a question in the prompt box at the bottom of the main section to the right.')

    if st.session_state.favorites is not None:
        st.sidebar.write('Favorite Prompts')

        # create dict of vetted prompts
        st.session_state.vetted_prompt_titles = {
            vp['title']: vp for k, vp in st.session_state.favorites['SelectBoxes']['Vetted_Prompts'].items()
        }
        st.session_state.vetted_prompts = {
            k: vp for k, vp in st.session_state.favorites['SelectBoxes']['Vetted_Prompts'].items()
        }

        # favorite buttons
        for k, button in st.session_state.favorites['Buttons'].items():
            if st.sidebar.button(button['title'], help=button['help'], key=k):
                # temp - re-read for every button click for realtime debugging.
                read_favorites()
                ups = []
                for up in st.session_state.favorites['Buttons'][k]['prompts']:
                    if up.startswith('@'):
                        box = st.session_state.vetted_prompts[up[1:]]
                        ups.extend(box['prompts']) # create list of lists in case prompts are dependent on previous prompts
                    else:
                        ups.append({'prompt':up})
                # 4 is arbitrary. clearning conversation so it doesn't become overwhelming to user or ai.
                if len(ups) > 4:
                    reset_messages()
                    ask_questions_without_context(
                        ups, st.session_state.ai_api)
                    #st.rerun() # this caused some systems to loop. not sure why.
                else:
                    ask_questions_without_context(ups, st.session_state.ai_api)

        # favorite prompts selectboxes
        if len(st.session_state.vetted_prompts):
            st.sidebar.selectbox("Vetted Prompts", index=None, options=st.session_state.vetted_prompt_titles.keys(),
                                    on_change=prompts_selectbox_change, key='prompts_selectbox')
            
        st.session_state.dataframe_tooltips = {
            col: tip for col, tip in st.session_state.favorites['ToolTips'].items()
        }

    if st.session_state.player_number_favorites is not None:
        st.sidebar.write(
            f"Player Number {st.session_state.player_number} Favorites")

        # player number favorite buttons
        for k, button in st.session_state.player_number_favorites['Buttons'].items():
            if st.sidebar.button(button['title'], help=button['help'], key=k):
                # temp - re-read for every button click for realtime debugging.
                read_favorites()
                ups = []
                for up in st.session_state.player_number_favorites['Buttons'][k]['prompts']:
                    if up.startswith('@'):
                        box = st.session_state.vetted_prompts[up[1:]]
                        ups.append(box['prompts']) # create list of lists in case prompts are dependent on previous prompts
                    else:
                        ups.append(up)
                ask_questions_without_context(ups, st.session_state.ai_api)

    with st.sidebar.expander('Developer Settings', False):

        if st.session_state.debug_favorites is not None:
            # favorite prompts selectboxes
            st.session_state.debug_player_number_names = st.session_state.debug_favorites[
                'SelectBoxes']['Player_Numbers']['options']
            if len(st.session_state.debug_player_number_names):
                # changed placeholder to player_number because when selectbox gets reset, possibly due to expander auto-collapsing, we don't want an unexpected value.
                # test player_number is not None else use debug_favorites['SelectBoxes']['Player_Numbers']['placeholder']?
                st.selectbox("Debug Player List", options=st.session_state.debug_player_number_names, placeholder=st.session_state.player_number, #.debug_favorites['SelectBoxes']['Player_Numbers']['placeholder'],
                                        on_change=debug_player_number_names_change, key='debug_player_number_names_selectbox')

        st.checkbox(
            "Ninja Coder Mode (Show SQL Queries)", on_change=show_sql_query_change, key='sql_query_checkbox')

        if len(st.session_state.ai_apis):
            st.selectbox("AI API Model Used for Prompts", index=st.session_state.ai_apis.index(st.session_state.ai_api),options=st.session_state.ai_apis,
                                    on_change=ai_api_selectbox_change, key='ai_api_selectbox')

        # Not at all fast to calculate. approximately .25 seconds per unique pbn overhead is minimum + .05 seconds per observation per unique pbn. e.g. time for 24 boards = 24 * (.25 + num of observations * .05).
        st.number_input("Single Dummy Random Trials", min_value=1, max_value=100,
                                value=st.session_state.sd_observations, on_change=sd_observations_changed, key='sd_observations_number_input')

    print_to_log_info('create_sidebar time:', time.time()-t)

def create_tab_bar():

    t = time.time()
    with st.container():

        chat_tab, data, dtypes, schema, commands_sql, URLs, system_prompt_tab, favorites, help, release_notes, about, debug = st.tabs(
            ['Chat', 'Data', 'dtypes', 'Schema', 'SQL', 'URLs', 'Sys Prompt', 'Favorites', 'Help', 'Release Notes', 'About', 'Debug'])
        streamlitlib.stick_it_good()

        with chat_tab:
            pass

        with data:
            if st.session_state.df is not None:
                # AgGrid unreliable in displaying within tab so using st.dataframe instead
                # todo: why? Neil's event 846812 causes id error. must be NaN? # .style.format({col:'{:,.2f}' for col in st.session_state.df.select_dtypes('float')}))
                streamlitlib.ShowDataFrameTable(st.session_state.df, key='data_tab_df', tooltips=st.session_state.dataframe_tooltips)
                #st.dataframe(st.session_state.df)

        with dtypes:
            # AgGrid unreliable in displaying within tab. Also issue with Series.
            # gave 'Serialization of dataframe to Arrow table was unsuccessful' .astype('string') was appended.
            st.dataframe(st.session_state.df.dtypes.astype('string'))

        with schema:
            if st.session_state.df_schema_string is not None:
                # st.dataframe(st.session_state.df_meta)
                st.write(st.session_state.df_schema_string)
                # todo: index column shows twice. once as index and once as column. fix.
                # st.divider()
                # st.dataframe(pd.concat(
                #    [st.session_state.df.columns.to_series(), st.session_state.df.dtypes.name], axis='columns')) # gave arrow conversion error until .name was appended.

        with commands_sql:
            st.header('SQL Commands')
            #st.write('SQL commands are not yet editable. Use the SQL commands to explore the data.')
            st.write(st.session_state.commands_sql)

        with URLs:
            st.write(
                f"Player number is {st.session_state.player_number}")
            st.divider()
            st.write('Club Game URLs')
            st.write(st.session_state.game_urls.values())
            st.write('Tournament Sessions')
            st.write(st.session_state.tournament_session_urls.values())

        with system_prompt_tab:
            st.header('System Prompt')
            # todo: make system prompt editable. useful for experimenting.
            #st.write('The system prompt is not yet editable.')
            st.write(st.session_state.augmented_system_prompt)

        with favorites:
            read_favorites()  # todo: update each time for debugging
            st.header(
                f"Default Favorites:{st.session_state.default_favorites_file}")
            if st.session_state.favorites is not None:
                st.write(st.session_state.favorites)
            st.divider()
            st.header(
                f"Player Number Custom Favorites:{st.session_state.player_number_custom_favorites_file}")
            if st.session_state.player_number_favorites is not None:
                st.write(st.session_state.player_number_favorites)
            if st.session_state.debug_favorites is not None:
                st.write(st.session_state.debug_favorites)

        with help:
            if st.session_state.help is None:
                st.write('Help not available.')
            else:
                st.markdown(st.session_state.help)

        with release_notes:
            if st.session_state.release_notes is None:
                st.write('No release notes available.')
            else:
                st.markdown(st.session_state.release_notes)

        with about:
            content = slash_about()
            st.write(content)
            app_info()

        with debug:
            st.header('Debug')
            st.write('Not yet implemented.')

    print_to_log_info('create_tab_bar time:', time.time()-t)

def create_main_section():

    # using streamlit's st.chat_input because it stays put at bottom, chat.openai.com style. # was chat_input
    if st.session_state.show_sql_query:
        if user_content := st.chat_input("Type your prompt here.", key='user_prompt_input'):
            ask_a_question_with_context(user_content) # don't think DEFAULT_LARGE_AI_MODEL is needed?
    # output all messages except the initial system message.
    # only system message
    t = time.time()
    if len(st.session_state.messages) == 1:
        # todo: put this message into config file.
        content = slash_about()
        streamlit_chat.message(f"Morty: {content}", key='create_main_section_about_message', logo=st.session_state.assistant_logo)
        if st.session_state.show_sql_query:
            streamlit_chat.message(
                f"Morty: Press the **Summarize** or **AI Predictions** button in the left sidebar or ask me questions using the **prompt box below**. Queries take about 10 seconds to complete.", key='chat_messages_user_no_messages', logo=st.session_state.assistant_logo)
        else:
            streamlit_chat.message(
                f"Morty: Press the **Summarize** or **AI Predictions** button in the left sidebar.", key='chat_messages_user_no_messages', logo=st.session_state.assistant_logo)
    else:
        with st.container():

            pdf_assets = []
            pdf_assets.append(f"# Bridge Game Postmortem Report Personalized for {st.session_state.player_number}")
            pdf_assets.append(f"### Created by http://postmortem.chat")
            pdf_assets.append(f"## Game Date: {st.session_state.game_date} Game ID: {st.session_state.session_id}")
            print_to_log_info('messages: len:', len(st.session_state.messages))
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "system":
                    assert i == 0, "First message should be system message."
                    continue
                if message["role"] == "user":
                    if st.session_state.show_sql_query:
                        streamlit_chat.message(
                            f"You: {message['content']}", is_user=True, key='chat_messages_user_'+str(i))
                        pdf_assets.append(f"You: {message['content']}")
                    user_prompt_help = ''
                    for k, prompt_sqls in st.session_state.vetted_prompts.items():
                        for prompt_sql in prompt_sqls['prompts']:
                            if message["content"] == prompt_keyword_replacements(prompt_sql['prompt']) or (prompt_sql['prompt'] == '' and message["content"] == prompt_keyword_replacements(prompt_sql['sql'])):
                                user_prompt_help = prompt_sqls['help']
                                break
                    continue
                elif message["role"] == "assistant":
                    # ```sql\nSELECT board_number, contract, COUNT(*) AS frequency\nFROM results\nGROUP BY board_number, contract\nORDER BY board_number, frequency DESC;\n```
                    if message['content'].startswith('/'):
                        slash_command = message['content'].split(
                            ' ', maxsplit=1)
                        assert len(slash_command), slash_command
                        if slash_command[0] == '/about':
                            # ['/about',content]
                            assert len(slash_command) == 2
                            streamlit_chat.message(
                                f"Morty: {slash_command[1]}", key='main.slash.'+str(i), logo=st.session_state.assistant_logo)
                            pdf_assets.append(f"🥸 Morty: {slash_command[1]}")
                        continue
                    match = re.match(
                        r'```sql\n(.*)\n```', message['content'])
                    print_to_log_info('message content:', message['content'], match)
                    if match is None:
                        # for unknown reasons, the sql query is returned in 'content'.
                        # hoping this is a SQL query
                        sql_query = message['content']
                        streamlit_chat.message(f"Morty: Oy, invalid SQL query: {sql_query}",
                                            key='main.invalid.'+str(i), logo=st.session_state.assistant_logo)
                        pdf_assets.append(f"🥸 Morty: Oy, invalid SQL query: {sql_query}")
                        continue
                    else:
                        # for unknown reasons, the sql query is returned embedded in a markdown code block.
                        sql_query = match.group(1)
                elif message['role'] == 'function':
                    sql_query = message['content']
                else:
                    assert False, message['role']
                if st.session_state.show_sql_query:
                    streamlit_chat.message(f"Ninja Coder: {sql_query}",
                                        key='main.embedded_sql.'+str(i), logo=st.session_state.guru_logo)
                    pdf_assets.append(f"Ninja Coder: {sql_query}")
                # use sql query as key. get last dataframe in list.
                assert len(
                    st.session_state.dataframes[sql_query]) > 0, "No dataframes for sql query."
                df = st.session_state.dataframes[sql_query][-1]
                if df.shape < (1, 1):
                    assistant_content = f"{user_prompt_help} -- Never happened."
                    streamlit_chat.message(
                        f"Morty: {assistant_content}", key='main.empty_dataframe.'+str(i), logo=st.session_state.assistant_logo)
                    pdf_assets.append(f"🥸 Morty: {assistant_content}")
                    continue
                if df.shape == (1, 1):
                    assistant_answer = str(df.columns[0]).replace(
                        'count_star()', 'count').replace('_', ' ')
                    assistant_scaler = df.iloc[0][0]
                    if assistant_scaler is pd.NA:
                        assistant_content = f"{assistant_answer} is None."
                    else:
                        assistant_content = f"{assistant_answer} is {assistant_scaler}."
                    streamlit_chat.message(
                        f"Morty: {user_prompt_help} {assistant_content}", key='main.dataframe_is_scaler.'+str(i), logo=st.session_state.assistant_logo)
                    pdf_assets.append(f"🥸 Morty: {user_prompt_help} {assistant_content}")
                    continue
                assistant_content = f"{user_prompt_help} Result is a dataframe of {len(df)} rows."
                streamlit_chat.message(
                    f"Morty: {assistant_content}", key='main.dataframe.'+str(i), logo=st.session_state.assistant_logo)
                pdf_assets.append(f"🥸 Morty: {assistant_content}")
                df.index.name = 'Row'
                st.session_state.df_unique_id += 1 # only needed because message dataframes aren't being released for some unknown reason.
                streamlitlib.ShowDataFrameTable(
                    df, key='main_messages_df_'+str(st.session_state.df_unique_id), color_column=None if len(df.columns) <= 1 else df.columns[1], tooltips=st.session_state.dataframe_tooltips) # only colorize if more than one column.
                pdf_assets.append(df)
                # else:
                #    st.dataframe(df.T.style.format(precision=2, thousands=""))

            if st.session_state.pdf_link.download_button(label="Download Personalized Report",
                    data=streamlitlib.create_pdf(pdf_assets, title=f"Bridge Game Postmortem Report Personalized for {st.session_state.player_number}"),
                    file_name = f"{st.session_state.session_id}-{st.session_state.player_number}-morty.pdf",
                    mime='application/octet-stream'):
                st.warning('Personalized report downloaded.')
            #pdf_base64_encoded = streamlitlib.create_pdf(pdf_assets)
            #download_pdf_html = f'<a href="data:application/octet-stream;base64,{pdf_base64_encoded.decode()}" download="{st.session_state.session_id}-{st.session_state.player_number}-morty.pdf">Download Personalized Report</a>'
            #st.session_state.pdf_link.markdown(download_pdf_html, unsafe_allow_html=True) # pdf_link is really a previously created st.sidebar.empty().

    # wish this would scroll to top of page but doesn't work.
    # js = '''
    # <script>
    #     var body = window.parent.document.querySelector(".main");
    #     console.log(body);
    #     body.scrollTop = 0;
    # </script>
    # '''
    # st.components.v1.html(js)

    streamlitlib.move_focus()
    print_to_log_info('create_main_section time:', time.time()-t)


def main():

#     AvatarStyle = [
#     "adventurer",
#     "adventurer-neutral",
#     "avataaars",
#     "avataaars-neutral",
#     "big-ears",
#     "big-ears-neutral",
#     "big-smile",
#     "bottts",
#     "bottts-neutral",
#     "croodles",
#     "croodles-neutral",
#     "fun-emoji",
#     "icons",
#     "identicon",
#     "initials",
#     "lorelei",
#     "lorelei-neutral",
#     "micah",
#     "miniavs",
#     "open-peeps",
#     "personas",
#     "pixel-art",
#     "pixel-art-neutral",
#     "shapes",
#     "thumbs",
# ]
#     for a in AvatarStyle:
#         streamlit_chat.message(
#             f"Hi. I'm Morty. Your friendly postmortem chatbot."+a, key='vacation_message_1'+a, avatar_style=a)


    if "player_number" not in st.session_state:
        # initialize values which will never change
        #st.set_page_config(page_title="Morty", page_icon=":robot_face:", layout="wide")
        # streamlitlib.widen_scrollbars()
        st.session_state.app_datetime = datetime.fromtimestamp(pathlib.Path(
            __file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        # in case there's no ai_apis.json file
        import platform
        if platform.system() == 'Windows': # ugh. this hack is required because torch somehow remembers the platform where the model was created. Must be a bug. Must lie to torch.
            pathlib.PosixPath = pathlib.WindowsPath
        else:
            pathlib.WindowsPath = pathlib.PosixPath
        st.session_state.ai_api = DEFAULT_AI_MODEL
        st.session_state.conn = duckdb.connect()
        st.session_state.show_sql_query = False
        st.session_state.player_number = None
        st.session_state.session_id = None
        st.session_state_tournament_session_id = None
        st.session_state.sd_observations = 10
        st.session_state.df_unique_id = 0 # only needed because message dataframes aren't being released for some unknown reason.
        st.session_state.assistant_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/main/assets/logo_assistant.gif?raw=true' # 🥸 todo: put into config. must have raw=true for github url.
        st.session_state.guru_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/main/assets/logo_guru.png?raw=true' # 🥷todo: put into config file. must have raw=true for github url.
        # causes streamlit connection error
        # if os.environ.get('STREAMLIT_ENV') is not None and os.environ.get('STREAMLIT_ENV') == 'development':
        #     if os.environ.get('STREAMLIT_QUERY_STRING') is not None:
        #         # todo: need to parse STREAMLIT_QUERY_STRING instead of hardcoding.
        #         if 'player_number' not in st.query_params:
        #             obsolete? st.experimental_set_query_params(player_number=2663279)
        # http://localhost:8501/?player_number=2663279
        if 'player_number' in st.query_params:
            player_number = st.query_params['player_number']
            if not isinstance(player_number, str):
                st.stop()
            #assert isinstance(player_number_l, list) and len(
            #    player_number_l) == 1, player_number_l
            #player_number = player_number_l[0]
            if not chat_initialize(player_number, None):
                st.stop()

    if st.session_state.player_number is None:
        st.sidebar.caption(f"App:{st.session_state.app_datetime}")
        if st.__version__ < '1.27.0':
            st.error('Please use http://postmortem.chat')
            st.stop()
        # temp!
        if False:
            streamlit_chat.message(
                "Hi. I'm Morty. Your friendly postmortem chatbot.", key='vacation_message_1', logo=st.session_state.assistant_logo)
            streamlit_chat.message(
                "I'm on a well deserved vacation while my overlord swaps out my chat API for something more economically sustainable. Should be back in a week or so. Meanwhile, happy prompting.", key='vacation_message_2', logo=st.session_state.assistant_logo)
            app_info()
            st.stop()
        st.sidebar.text_input(
            "Enter an ACBL player number", on_change=player_number_change, placeholder='2663279', key='player_number_input')
        streamlit_chat.message(
            "Hi. I'm Morty. Your friendly postmortem chatbot. I only want to chat about ACBL pair matchpoint games using a Mitchell movement and not shuffled.", key='intro_message_1', logo=st.session_state.assistant_logo)
        streamlit_chat.message(
            "I'm optimized for large screen devices such as a notebook or monitor. Do not use a smartphone.", key='intro_message_2', logo=st.session_state.assistant_logo)
        streamlit_chat.message(
            "To start our postmortem chat, I'll need an ACBL player number. I'll use it to find player's latest ACBL club game. It will be the subject of our chat.", key='intro_message_3', logo=st.session_state.assistant_logo)
        streamlit_chat.message(
            "Enter any ACBL player number in the left sidebar.", key='intro_message_4', logo=st.session_state.assistant_logo)
        streamlit_chat.message(
            "I'm just a Proof of Concept so don't double me.", key='intro_message_5', logo=st.session_state.assistant_logo)
        app_info()

    else:
         
        create_sidebar()

        create_tab_bar()

        create_main_section()


if __name__ == '__main__':
    main()