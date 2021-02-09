local prob_estimator = import '../../prob_estimators/lexsub/roberta_masked.jsonnet';
local post_processing = import '../post_processors/lower_nltk_spacy.jsonnet';

{
    class_name: "SubstituteGenerator",
    pre_processing: [
        {
            class_name: "pre_processors.base_preprocessors.AndPreprocessor"
        }
    ],
    prob_estimator: prob_estimator,
    post_processing: [
        {class_name: "post_processors.roberta_postproc.RobertaPostProcessor", strategy: "drop_subwords"},
        {
            class_name: 'post_processors.base_postprocessor.LowercasePostProcessor'
        },
        {
            class_name: "post_processors.target_excluder.TargetExcluder",
            lemmatizer: "spacy_old"
        },
    ],
    top_k: 10,
}
