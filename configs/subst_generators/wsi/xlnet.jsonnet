local prob_estimator = import '../../prob_estimators/wsi/xlnet.jsonnet';
local post_processing = import '../post_processors/spacy_old_sum.jsonnet';

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
    top_k: 200
}
