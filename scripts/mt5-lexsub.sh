python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/mt5_rusemeval.jsonnet --dataset-config-path configs/dataset_readers/lexsub/rusemeval.jsonnet --run-dir='debug/mt5-lexsub/rusemeval_mt5' --force --experiment-name='mt5-lexsub' --run-name='rusemeval_mt5'
python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/mt5_germeval.jsonnet --dataset-config-path configs/dataset_readers/lexsub/germeval_train.jsonnet --run-dir='debug/mt5-lexsub/germeval_train_mt5' --force --experiment-name='mt5-lexsub' --run-name='germeval_train_mt5'
python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/mt5_semevall_all.jsonnet --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet --run-dir='debug/mt5-lexsub/semeval_all_mt5' --force --experiment-name='mt5-lexsub' --run-name='semeval_all_mt5'
python lexsubgen/evaluations/lexsub.py solve --substgen-config-path configs/subst_generators/lexsub/mt5_coinco.jsonnet --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet --run-dir='debug/mt5-lexsub/coinco_mt5' --force --experiment-name='mt5-lexsub' --run-name='coinco_mt5'