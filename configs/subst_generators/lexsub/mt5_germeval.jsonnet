local pre_processing = import '../pre_processors/mt5_germeval.jsonnet';
local prob_estimator = import '../../prob_estimators/lexsub/mt5.jsonnet';
local post_processing = import '../post_processors/hanta_ge.jsonnet';

{
    class_name: "SubstituteGenerator",
    pre_processing: pre_processing,
    prob_estimator: prob_estimator,
    post_processing: post_processing,
    top_k: 10
}