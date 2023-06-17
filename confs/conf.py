categorical_features = ['napa_code']
numerical_features = ['mea_madad_pikud', 'seif_likuy_total',
'mea_madad_avodat_zevet',
'weight',
'seif_likuy_sum_severity',
'bmi',
'seif_likuy_max_severity',
'profil',
'yehidot_english',
'mea_madad_keshev_mitmasheh',
'mea_svivat_ahzaka',
'bagrut_total',
'dapar_amilulit',
'mea_svivat_irgun',
'dapar_atzuranit',
'bagrut_max_units',
'dapar_horaot',
'dapar_haskamuti',
'yehidot_physics',
'mea_svivat_ibud',
'bagrut_sum_units',
'dapar',
'mea_svivat_afaala',
'yehidot_math',
'yehidot_computers']

# numerical_features = ['bmi', 'height', 'weight', 'profil',
#                       'seif_likuy_max_severity', 'seif_likuy_sum_severity', 'seif_likuy_total',
#                       'bagrut_max_units', 'bagrut_sum_units', 'bagrut_total',
#                       'yehidot_physics', 'yehidot_chemistry', 'yehidot_english', 'yehidot_math',
#                       'yehidot_computers',
#                       'dapar', 'dapar_horaot', 'dapar_amilulit', 'dapar_haskamuti', 'dapar_atzuranit',
#                       'mea_madad_keshev_mitmasheh', 'mea_svivat_ahzaka', 'mea_madad_hashkaa',
#                       'mea_svivat_tipul', 'mea_svivat_afaala', 'mea_svivat_irgun', 'mea_svivat_ibud',
#                       'mea_madad_avodat_zevet', 'mea_bagrut', 'mea_svivat_adraha', 'mea_madad_keshev_selectivi',
#                       'mea_madad_pikud', 'mea_svivat_sade', 'mea_misgeret',
#                       'city_sotzio', 'sotzio']


project_path = '/Users/netalorberbom/Library/CloudStorage/OneDrive-Payoneer/personal/msc/analytics_project'
target_feature = 'target'
target_name = 'אשכול תפקידי מקצועות המחשב_אשכול מקצועות המחשב'
folder_name = 'mahshev'
show_plot = False
include_bagrut = True
include_likui = True
algo_name = 'xgboost'
