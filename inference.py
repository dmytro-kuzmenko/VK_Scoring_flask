import pandas as pd
import numpy as np
import networkx as nx
import joblib

features_to_use = ['followers_count', 'has_mobile', 'has_photo', 'status', 'city', 'country', \
                   'occupation', 'skype', 'religion_id', 'smoking', 'life_main', 'relatives', 'alcohol', \
                   'university_name', 'instagram', 'sex']

"""
    Utils
"""


def np_divide(a, b, replacement=np.nan) -> float:
    """Wrapper for convenient np.divide"""
    if b == 0:
        return replacement
    else:
        return a / b


def pd_qcut(series: pd.Series, qcuts_cnt: int) -> pd.Series:
    """Wrapper for convenient pandas.qcut"""
    if series.dtype != object:

        series_dropna = series.dropna()
        series_dropna_nunique = series_dropna.nunique()
        if series_dropna_nunique <= qcuts_cnt:
            qcuts_cnt = series_dropna_nunique - 1
        else:
            qcuts_cnt = qcuts_cnt
        precision = 3
        qcut_series = pd.qcut(
            series_dropna,
            q=qcuts_cnt,
            duplicates='drop',
            precision=precision
        )

        return qcut_series

    return series


def get_iv(local_df: pd.DataFrame, global_df: pd.Series, target: str) -> float:
    """Calculate the IV"""
    global_target_vc = global_df[target].value_counts()
    local_target_vc = local_df[target].value_counts()

    if (local_target_vc.get(1, 0) == 0) or (local_target_vc.get(0, 0) == 0):
        iv = 0
        return iv

    distribution_of_goods = np_divide(local_target_vc.get(1, 0), global_target_vc.get(1, 0))
    distribution_of_bads = np_divide(local_target_vc.get(0, 0), global_target_vc.get(0, 0))
    woe = np.log(distribution_of_goods / distribution_of_bads)
    iv = (distribution_of_goods - distribution_of_bads) * woe

    return iv


def create_dag(df: pd.DataFrame, feature_raw_values: str, feature_encoded_values: str, target: str) -> tuple:
    dag = nx.DiGraph()

    precision = 3
    feature_categories = df[feature_encoded_values].value_counts().sort_index().index.values
    for category_idx, category in enumerate(feature_categories):

        category_left = round(category.left, precision)
        category_right = round(category.right, precision)
        category_filter_left_part = df[feature_raw_values] > category_left
        category_filter = category_filter_left_part & (df[feature_raw_values] <= (category_right + 0.001))
        category_iv = get_iv(df[category_filter], df, target)

        dag.add_edge(category_left, category_right, weight=category_iv)

        for next_category in feature_categories[category_idx:]:

            category_left = round(category.left, precision)
            next_category_right = round(next_category.right, precision)
            next_category_filter = category_filter_left_part & (df[feature_raw_values] <= (next_category_right + 0.001))
            next_category_iv = get_iv(df[next_category_filter], df, target)

            dag.add_edge(category_left, next_category_right, weight=next_category_iv)

    source = round(feature_categories[0].left, precision)
    target = round(feature_categories[-1].right, precision)

    return dag, source, target


def get_path_weight(dag: nx.DiGraph, path: tuple) -> float:

    path_iv = 0
    for idx in range(len(path) - 1):
        dag_edge = dag.get_edge_data(path[idx], path[idx + 1])
        dag_edge_weight = dag_edge['weight']
        path_iv += dag_edge_weight

    return path_iv


def get_longest_path(df: pd.DataFrame, dag: nx.DiGraph, source: float, target: float) -> float:
    """Identify longest path which has the highest IV"""

    all_simple_paths = nx.all_simple_paths(dag, source, target, 6)
    all_simple_paths = sorted(all_simple_paths, key=len, reverse=True)

    longest_path = max(
        (path for path in all_simple_paths),
        key=lambda path: get_path_weight(dag, path)
    )

    return longest_path


class FeatureEncoder():

    def __init__(self, input_row, feature_list):
        self.input_row = input_row
        self.output_row = input_row.copy()
        self.feature_list = feature_list

        self.impact_dict = \
            {
                'city': (True, 0.005),
                'country': (True, 0.001),
                'university_name': (True, 0.0015),

                'followers_count': (False, 0),
                'has_mobile': (False, 0),
                'has_photo': (False, 0),
                'status': (False, 0),
                'occupation': (False, 0),
                'skype': (False, 0),
                'religion_id': (False, 0),
                'smoking': (False, 0),
                'life_main': (False, 0),
                'relatives': (False, 0),
                'alcohol': (False, 0),
                'instagram': (False, 0),
                'sex': (False, 0),
            }

    def encode_followers_count(self, feature='followers_count'):

        feature_value = self.input_row[feature].values[0]

        if feature_value > 427:
            self.output_row[feature] = 6
        elif 119 < feature_value <= 427:
            self.output_row[feature] = 5
        elif 46 < feature_value <= 119:
            self.output_row[feature] = 4
        elif 21 < feature_value <= 46:
            self.output_row[feature] = 3
        elif 3 < feature_value <= 21:
            self.output_row[feature] = 2
        elif 0 <= feature_value <= 3:
            self.output_row[feature] = 1

    def encode_city(self, input_row=None, feature='city'):

        input_row = self.input_row

        mapping = {
            'Санкт-Петербург': 1,

            'non-impactful': 2,

            'Уфа': 3,
            'Нижний Новгород': 3,
            'Москва': 3,

            'Новосибирск': 4,
            np.nan: 4,
            'nan': 4,
            'Екатеринбург': 4,
        }

        input_row[feature] = input_row[feature].map(mapping)
        self.output_row[feature] = input_row[feature].map(lambda x: 4 if pd.isna(x) else x)

    def encode_country(self, input_row=None, feature='country'):

        input_row = self.input_row

        mapping = {
            'Россия': 1,
            'Италия': 1,

            'Испания': 2,
            'non-impactful': 2,
            'Финляндия': 2,
            'Германия': 2,
            'Канада': 2,
            'Эстония': 2,
            'США': 2,
            'Беларусь': 2,
            'Израиль': 2,
            'Франция': 2,

            np.nan: 3,
            'nan': 3,
            'Казахстан': 3,
            'Украина': 3,
            'Великобритания': 3,
            'Латвия': 3,
            'Азербайджан': 3,
            'Узбекистан': 3,
        }

        input_row[feature] = input_row[feature].map(mapping)
        self.output_row[feature] = input_row[feature].map(lambda x: 2 if pd.isna(x) else x)

    def encode_occupation(self, input_row=None, feature='occupation'):

        input_row = self.input_row

        mapping = {
            'work': 1,

            'school': 2,
            'university': 2,

            np.nan: 3,
            'nan': 3,
        }

        self.output_row[feature] = input_row[feature].map(mapping)

    def encode_religion_id(self, input_row=None, feature='religion_id'):

        input_row = self.input_row

        mapping = {
            102: 1,
            200: 1,
            139: 1,
            124: 1,
            129: 1,

            np.nan: 1,
            'nan': 1,

            107: 2,
            201: 2,
            167: 2,
            1: 2,
            101: 2,
        }

        self.output_row[feature] = input_row[feature].map(mapping)

    def encode_smoking(self, input_row=None, feature='smoking'):

        input_row = self.input_row

        mapping = {
            1: 1,
            3: 1,
            5: 1,
            2: 1,
            4: 1,

            np.nan: 2,
            'nan': 2,
        }

        self.output_row[feature] = input_row[feature].map(mapping)

    def encode_life_main(self, input_row=None, feature='life_main'):

        input_row = self.input_row

        mapping = {
            1: 1,
            5: 1,
            8: 1,
            7: 1,
            4: 1,
            6: 1,
            3: 1,
            2: 1,

            np.nan: 2,
            'nan': 2,
        }

        self.output_row[feature] = input_row[feature].map(mapping)

    def encode_alcohol(self, input_row=None, feature='alcohol'):

        input_row = self.input_row

        mapping = {
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,

            np.nan: 2,
            'nan': 2,
        }

        self.output_row[feature] = input_row[feature].map(mapping)

    def encode_university_name(self, input_row=None, feature='university_name'):

        input_row = self.input_row

        mapping = {
            'РГПУ им. А. И. Герцена': 1,
            np.nan: 1,
            'nan': 1,

            'СПбПУ Петра Великого (Политех)': 2,
            'СПбГЭУ': 2,
            'non-impactful': 2,
            'СПбГУ': 2,
            'МГУ': 2,
        }

        input_row[feature] = input_row[feature].map(mapping)
        self.output_row[feature] = input_row[feature].map(lambda x: 2 if pd.isna(x) else x)

    def encode_sex(self, input_row=None, feature='sex'):

        input_row = self.input_row

        mapping = {
            1: 1,
            2: 2,
            0: 2,
        }

        self.output_row[feature] = input_row[feature].map(mapping)

    def encode_features(self):

        self.encode_followers_count()
        self.encode_city()
        self.encode_country()
        self.encode_occupation()
        self.encode_religion_id()
        self.encode_smoking()
        self.encode_life_main()
        self.encode_alcohol()
        self.encode_university_name()
        self.encode_sex()


# local testing
if __name__ == '__main__':
    fe = FeatureEncoder(pd.DataFrame(test_dict, index=[0]), features_to_use)
    fe.encode_features()
    encoded_vector = fe.output_row

    model = joblib.load('./xgb_16f_074-AUC.joblib')
    print(model.predict_proba(encoded_vector)[:, 0])