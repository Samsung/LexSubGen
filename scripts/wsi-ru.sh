
# ELMo - bts-rnc train -> ARI == 0.456406
python lexsubgen/evaluations/wsi.py solve  --dataset-config-path configs/dataset_readers/wsi/russe-bts-rnc-train.jsonnet --substgen-config-path configs/subst_generators/wsi/elmo_ru.jsonnet --clusterizer-config-path configs/clusterizers/agglo_ru.jsonnet --run-dir debug/elmo-ru --force --experiment-name='wsi-bts-rnc' --run-name='elmo-ru' --verbose=True --batch-size 100

# ELMo - bts-rnc public test -> ARI == 0.501776
python lexsubgen/evaluations/wsi.py solve  --dataset-config-path configs/dataset_readers/wsi/russe-bts-rnc-public-test.jsonnet --substgen-config-path configs/subst_generators/wsi/elmo_bcomb_lms_ru.jsonnet --clusterizer-config-path configs/clusterizers/agglo_silhouette_score_ru.jsonnet --run-dir debug/elmo-bcomb-lms-ru --force --experiment-name='wsi-bts-rnc' --run-name='elmo-bcomb-lms-ru' --verbose=True --batch-size 100
