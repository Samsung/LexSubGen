local post_processing = import '../post_processors/lower_nltk_spacy.jsonnet';
local prob_estimator = import '../../prob_estimators/lexsub/xlnet_embs.jsonnet';

{
    class_name: "SubstituteGenerator",
    pre_processing: [
        {
            class_name: "pre_processors.base_preprocessors.AddPunctPreprocessor"
        },
        {
            class_name: "pre_processors.base_preprocessors.PadTextPreprocessor",
            text: "Bla bla bla bla bla bla bla bla " +
                  "bla bla bla bla bla bla bla bla. "
        },
        {
            class_name: "pre_processors.base_preprocessors.PadTextPreprocessor",
            special_end_token: "<eod> "
        }
    ],
    prob_estimator: prob_estimator,
    post_processing: post_processing,
    top_k: 10
}
