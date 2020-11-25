local pre_processing = import '../pre_processors/bert_preprocessor.jsonnet';
local prob_estimator = import '../../prob_estimators/lexsub/bert_embs.jsonnet';
local post_processing = import '../post_processors/lower_nltk_spacy.jsonnet';

{
    class_name: "SubstituteGenerator",
    pre_processing: pre_processing,
    prob_estimator: prob_estimator,
    post_processing: post_processing,
    top_k: 10
}
