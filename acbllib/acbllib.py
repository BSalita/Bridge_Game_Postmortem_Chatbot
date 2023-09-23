# functions which are specific to acbl; downloading acbl webpages, api calls.

import pandas as pd
import re
import traceback
import requests
from bs4 import BeautifulSoup
import urllib
import time
import json


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
        event_id = int(href.split('/')[-1]) # extract event_id from href which is the last part of url
        my_results_details_data[event_id] = (url, detail_url, msg)
    return my_results_details_data


# get a single tournament session result
def get_tournament_session_results(session_id, acbl_api_key):
    headers = {'accept':'application/json', 'Authorization':acbl_api_key[len('Authorization: '):]}
    path = 'https://api.acbl.org/v1/tournament/session'
    query = {'id':session_id,'full_monty':1}
    params = urllib.parse.urlencode(query)
    url = path+'?'+params
    print('tournament session url:',url)
    response = requests.get(url, headers=headers)
    assert response.status_code == 200, [url, response.status_code]
    json_response = response.json()
    #json_pretty = json.dumps(json_response, indent=4)
    return json_response


# get a list of tournament session results
def get_tournament_sessions_from_acbl_number(acbl_number, acbl_api_key):
    url, json_responses = download_tournament_player_history(acbl_number, acbl_api_key)
    tournament_sessions_urls = {d['session_id']:(url, f"https://live.acbl.org/event/{d['session_id'].replace('-','/')}/summary", f"{d['date']}, {d['score_tournament_name']}, {d['score_event_name']}, {d['score_session_time_description']}, {d['percentage']}", d) for r in json_responses for d in r['data']} # https://live.acbl.org/event/NABC232/23FP/1/summary
    return tournament_sessions_urls


# get a single player's tournament history
def download_tournament_player_history(player_id, acbl_api_key):
    headers = {'accept':'application/json', 'Authorization':acbl_api_key[len('Authorization: '):]}
    path = 'https://api.acbl.org/v1/tournament/player/history_query'
    query = {'acbl_number':player_id,'page':1,'page_size':200,'start_date':'1900-01-01'}
    params = urllib.parse.urlencode(query)
    url = path+'?'+params
    sessions_count = 0
    except_count = 0
    json_responses = []
    while url:
        try:
            response = requests.get(url, headers=headers)
        except Exception as ex:
            print(f'Exception: count:{except_count} type:{type(ex).__name__} args:{ex.args}')
            if except_count > 5:
                print('Except count exceeded')
                break # skip url
            except_count += 1
            time.sleep(1) # just in case the exception is transient
            continue # retry url
        except KeyboardInterrupt as e:
            print(f"Error: {type(e).__name__} while processing file:{url}")
            print(traceback.format_exc())
            return None
        else:
            except_count = 0
        if response.status_code in [400,500,504]: # 500 is unknown response code. try skipping player
            print(f'Status Code:{response.status_code}: count:{len(json_responses)} skipping') # 4476921 - Thx Merle.
            # next_page_url = None
            # sessions_total = 0
            break
        assert response.status_code == 200, (url, response.status_code) # 401 is authorization error often because Persanal Access Token has expired.
        json_response = response.json()
        #json_pretty = json.dumps(json_response, indent=4)
        #print(json_pretty)
        json_responses.append(json_response)
        url = json_response['next_page_url']
    return path, json_responses


# get a list of player's tournament history
def download_tournament_players_history(player_ids, acbl_api_key, dirPath):
    start_time = time.time()
    get_count = 0 # total number of gets
    #canceled = False
    for n,player_id in enumerate(sorted(player_ids)):
        if player_id.startswith('tmp:') or player_id.startswith('#'): # somehow #* crept into player_id
            print(f'Skipping player_id:{player_id}')
            continue
        else:
            print(f'Processing player_id:{player_id}')
        if dirPath.exists():
            session_file_count = len(list(dirPath.glob('*.session.json')))
            print(f'dir exists: file count:{session_file_count} dir:{dirPath}')
            #if session_file_count == 0: # todo: ignore players who never played a tournament?
            #    print(f'dir empty -- skipping')
            #    continue
            #if session_file_count > 0: # todo: temp?
            #    print(f'dir not empty -- skipping')
            #    continue
        else:
            print(f'Creating dir:{dirPath}')
            dirPath.mkdir(parents=True,exist_ok=True)
            session_file_count = 0
        url, json_responses = download_tournament_player_history(player_id, acbl_api_key)
        if json_responses is None: # canceled
            break
        get_count = len(json_responses)
        if get_count == 0: # skip player_id's generating errors. e.g. player_id 5103045, 5103045, 5103053
            continue
        print(f"{n}/{len(player_ids)} gets:{get_count} rate:{round((time.time()-start_time)/get_count,2)} {player_id=}")
        #time.sleep(1) # throttle api calling. Maybe not needed as api is taking longer than 1s.
        sessions_count = 0
        for json_response in json_responses:
            sessions_total = json_response['total'] # is same for every page
            if sessions_total == session_file_count: # sometimes won't agree because identical sessions. revised results?
                print(f'File count correct: {dirPath}: terminating {player_id} early.')
                sessions_count = sessions_total
                break
            for data in json_response['data']:
                sessions_count += 1 # todo: oops, starts first one at 2. need to move
                session_id = data['session_id']
                filePath_sql = dirPath.joinpath(session_id+'.session.sql')
                filePath_json = dirPath.joinpath(session_id+'.session.json')
                if filePath_sql.exists(): # todo: and filePath_json.exists() and filePath_sql is newer than filePath_json
                    print(f'{sessions_count}/{sessions_total}: File exists: {filePath_sql}: skipping')
                    #if filePath_json.exists(): # json file is no longer needed?
                    #    print(f'Deleting JSON file: {filePath_json}')
                    #    filePath_json.unlink(missing_ok=True)
                    break # continue will skip file. break will move on to next player
                if filePath_json.exists():
                    print(f'{sessions_count}/{sessions_total}: File exists: {filePath_json}: skipping')
                    break # continue will skip file. break will move on to next player
                print(f'{sessions_count}/{sessions_total}: Writing:',filePath_json)
                with open(filePath_json,'w',encoding='UTF8') as f:
                    f.write(json.dumps(data, indent=4))
        if sessions_count != sessions_total:
            print(f'Session count mismatch: {dirPath}: variance:{sessions_count-sessions_total}')