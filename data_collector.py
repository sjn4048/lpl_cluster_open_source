import numpy as np
import requests

lst_cache = {}


def collect_data(tournament_lst: list, keys=None, min_occur: int = 20) -> (list, list, list, list):
    # raw data
    d = []
    for t in tournament_lst:
        t_lst = []
        if t in lst_cache:
            t_lst = lst_cache[t]
        else:
            headers = {
                'authority': 'www.scoregg.com',
                'accept': 'application/json, text/javascript, */*; q=0.01',
                'accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
                'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
                'origin': 'https://www.scoregg.com',
                'referer': 'https://www.scoregg.com/data/player?tournamentID=',
                'sec-ch-ua': '"Google Chrome";v="111", "Not(A:Brand";v="8", "Chromium";v="111"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
                'x-requested-with': 'XMLHttpRequest',
            }
            for page in range(1, 100000):
                post_data = {
                    'api_path': '/services/gamingDatabase/match_data_ssdb_list.php',
                    'method': 'post',
                    'platform': 'web',
                    'api_version': '9.9.9',
                    'language_id': '1',
                    'type': 'player',
                    'order_type': 'KDA',
                    'order_value': 'DESC',
                    'team_name': '',
                    'player_name': '',
                    'positionID': '',
                    'page': page,
                    'tournament_id': t
                }
                with requests.post('https://www.scoregg.com/services/api_url.php', headers=headers,
                                   data=post_data) as r:
                    r.encoding = 'utf8'
                    resp_data = r.json()['data']['data']['list']
                    if resp_data:
                        t_lst += resp_data
                    else:
                        lst_cache[t] = t_lst
                        break
        d += t_lst

    use_keys_full = {'KDA', 'PLAYS_TIMES', 'OFFERED_RATE', 'AVERAGE_KILLS', 'AVERAGE_ASSISTS', 'AVERAGE_DEATHS',
                     'MINUTE_ECONOMIC', 'MINUTE_HITS', 'MINUTE_DAMAGEDEALT', 'DAMAGEDEALT_RATE', 'MINUTE_DAMAGETAKEN',
                     'DAMAGETAKEN_RATE', 'MINUTE_WARDSPLACED', 'MINUTE_WARDKILLED', 'MVP', 'f_score', 'total_kills',
                     'total_deaths', 'total_assists'}

    use_keys_middle = {'OFFERED_RATE': '参团率', 'AVERAGE_KILLS': '平均击杀',
                       'AVERAGE_ASSISTS': '平均助攻', 'AVERAGE_DEATHS': '平均死亡', 'MINUTE_ECONOMIC': '分均经济',
                       'MINUTE_HITS': '分均补刀', 'MINUTE_DAMAGEDEALT': '分均伤害', 'DAMAGEDEALT_RATE': '伤害占比',
                       'MINUTE_DAMAGETAKEN': '分均承伤', 'DAMAGETAKEN_RATE': '承伤占比',
                       'MINUTE_WARDSPLACED': '分均插眼', 'MINUTE_WARDKILLED': '分均排眼'}

    use_keys_mini = {'KDA': 'KDA', 'OFFERED_RATE': '参团率',
                     'MINUTE_DAMAGEDEALT': '分均伤害', 'DAMAGEDEALT_RATE': '伤害占比',
                     'MINUTE_DAMAGETAKEN': '分均承伤', 'DAMAGETAKEN_RATE': '承伤占比'}

    if keys == 'mini':
        use_keys = use_keys_mini
    else:
        use_keys = use_keys_middle

    d = [x for x in d if int(x['PLAYS_TIMES']) >= min_occur]

    numerics_ = [{k: v for k, v in p.items() if k in use_keys} for p in d]
    names_ = [p['player_name'] for p in d]
    labels_ = [p['position'] for p in d]

    feature_names_ = [use_keys[k] for k in d[0] if k in use_keys]

    return numerics_, names_, labels_, feature_names_


def preprocess_data(train_data_raw_, normalize: bool = True, ranked: bool = False):
    def normalize_(_v: np.ndarray) -> np.ndarray:
        stats = (_v.mean(axis=0), _v.std(axis=0))
        return (_v - stats[0]) / stats[1]

    def ranked_(arr: np.ndarray) -> np.ndarray:
        sorted_index = np.argsort(np.argsort(arr, axis=0), axis=0)
        return sorted_index

    train_data_ = np.array([[float(v) for k, v in p.items()] for p in train_data_raw_])
    if ranked:
        train_data_ = ranked_(train_data_)
    if normalize:
        train_data_ = normalize_(train_data_)

    return train_data_


def get_tournament_data(tournaments: list, keys: str = 'all'):
    tournament_keys = {
        'S13-spring-LPL': 447,
        'S13-summer-LPL': 552,
        'S13-spring-LCK': 449,
        'S13-spring-LEC': 499,
        'S13-spring-LCS': 448,
        'S9-Worlds': 139,
        'S8-Worlds': 95
    }

    min_occur = 10
    tournaments = [tournament_keys[x] for x in tournaments]
    train_raw_data, _, _, _ = collect_data(tournaments, keys=keys, min_occur=min_occur)
    valid_raw_data, valid_names, valid_labels_gt, feature_names = collect_data(tournaments, keys=keys,
                                                                               min_occur=min_occur)
    # data preprocess
    train_data = preprocess_data(train_raw_data, normalize=True)
    valid_data = preprocess_data(valid_raw_data, normalize=True)
    return train_data, valid_data, valid_names, valid_labels_gt, feature_names
