import logging
categorical_features = ['napa_code']

# numerical_features = ['bmi',
# 'math_5_ration',
# 'yehidot_chemistry',
# 'weight',
# 'dapar_horaot',
# 'mea_svivat_ibud',
# 'dapar',
# 'height',
# 'dapar_haskamuti',
# 'yehidot_math',
# 'bagrut_sum_units',
# 'comp_5_ration',
# 'bagrut_max_units',
# 'yehidot_computers',
# 'mea_svivat_afaala',
# 'yehidot_physics',
# 'dapar_atzuranit',
# 'profil',
# 'mea_misgeret',
# 'mea_madad_pikud',
# 'seif_likuy_max_severity',
# 'dapar_amilulit',
# 'seif_likuy_sum_severity']

numerical_features = ['bmi', 'height', 'weight', 'profil',
                      'seif_likuy_max_severity', 'seif_likuy_sum_severity', 'seif_likuy_total',
                      'bagrut_max_units', 'bagrut_sum_units', 'bagrut_total',
                      'yehidot_physics', 'yehidot_chemistry', 'yehidot_english', 'yehidot_math',
                      'yehidot_computers',
                      'dapar', 'dapar_horaot', 'dapar_amilulit', 'dapar_haskamuti', 'dapar_atzuranit',
                      'mea_madad_keshev_mitmasheh', 'mea_svivat_ahzaka', 'mea_madad_hashkaa',
                      'mea_svivat_tipul', 'mea_svivat_afaala', 'mea_svivat_irgun', 'mea_svivat_ibud',
                      'mea_madad_avodat_zevet', 'mea_bagrut', 'mea_svivat_adraha', 'mea_madad_keshev_selectivi',
                      'mea_madad_pikud', 'mea_svivat_sade', 'mea_misgeret',
                      'city_sotzio', 'sotzio', 'math_5_ration', 'comp_5_ration']


project_path = '/Users/netalorberbom/Library/CloudStorage/OneDrive-Payoneer/personal/msc/analytics_project'
target_feature = 'target'














# target_name = 'אשכול תפקידי מקצועות המחשב_חיילת מקצועות המחשוב'
# folder_name = 'mihshuv'
# target_name = 'אשכול תפקידי מקצועות המחשב_אשכול מקצועות המחשב'
# folder_name = 'mahshev'
show_plot = False
include_bagrut = True
include_likui = True
algo_name = 'xgboost'

search_space = {
    "classifier__subsample": [0.75, 1],
    "classifier__colsample_bytree": [0.75, 1],
    "classifier__max_depth": [2, 4, 6],
    "classifier__lambda": [0, 0.1, 1, 3],
    "classifier__alpha": [0, 0.1, 1, 3],
    # "classifier__min_child_weight": [1, 6],
    "classifier__learning_rate": [0.05, 0.1, 0.3],
    "classifier__n_estimators": [50, 100, 200, 300, 400]}  # , 10, 50]}

logging.basicConfig(
    # format='%(funcName).10s %(message)s',
    level=logging.INFO,
)

logger = logging.getLogger()