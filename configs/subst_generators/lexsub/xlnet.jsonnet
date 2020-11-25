local pre_processing = import '../pre_processors/xlnet_preprocessor.jsonnet';
local prob_estimator = import '../../prob_estimators/lexsub/xlnet.jsonnet';
local post_processing = import '../post_processors/lower_nltk_spacy.jsonnet';

{
    class_name: "SubstituteGenerator",
    pre_processing: pre_processing,
    prob_estimator: prob_estimator,
    post_processing: post_processing,
    top_k: 10
}
