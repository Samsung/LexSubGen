local prob_estimator = import '../../prob_estimators/lexsub/elmo.jsonnet';
local post_processing = import '../post_processors/lower_nltk_spacy.jsonnet';

{
    class_name: "SubstituteGenerator",
    pre_processing: [
        {
            class_name: "pre_processors.base_preprocessors.AddPunctPreprocessor"
        }
    ],
    prob_estimator: prob_estimator,
    post_processing: post_processing,
    top_k: 10,
}
