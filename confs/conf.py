categorical_features = ['napa_code']

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
# target_name = 'אשכול תפקידי מקצועות המחשב_חיילת מקצועות התקשוב'
# folder_name = 'tikshuv'
# target_name = 'אשכול תפקידי מקצועות המחשב_חיילת מקצועות המחשוב'
# folder_name = 'mihshuv'
target_name = 'אשכול תפקידי מקצועות המחשב_אשכול מקצועות המחשב'
folder_name = 'mahshev'
show_plot = False
include_bagrut = True
include_likui = True
algo_name = 'xgboost'
