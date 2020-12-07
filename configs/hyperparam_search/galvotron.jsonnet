local reader = import '../dataset_readers/lexsub/semeval_all.jsonnet';
local post_processing = import '../subst_generators/post_processors/spacy_max.jsonnet';

{
    class_name: "evaluations.lexsub.LexSubEvaluation",
    dataset_reader: reader,
    batch_size: 50,
    substitute_generator: {
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
        prob_estimator: {
            class_name: "prob_estimators.combiner.BcombCombiner",
            prob_estimators: [
                {
                    class_name: "prob_estimators.xlnet_estimator.XLNetProbEstimator",
                    masked: false,
                    model_name: "xlnet-large-cased",
                    temperature: 1.0,
                    use_input_mask: true,
                    cuda_device: 0
                },
                {
                    class_name: "prob_estimators.xlnet_estimator.XLNetProbEstimator",
                    masked: true,
                    model_name: "xlnet-large-cased",
                    temperature: {
                        class_name: "LinspaceHyperparam",
                        start: 0.1,
                        end: 2.5,
                        size: 7,
                        name: "masked_model_temp"
                    },
                    use_input_mask: true,
                    cuda_device: 0
                },
                {
                    class_name: "prob_estimators.xlnet_estimator.XLNetProbEstimator",
                    masked: true,
                    model_name: "xlnet-large-cased",
                    embedding_similarity: true,
                    temperature: {
                        class_name: "LogspaceHyperparam",
                        start: -2.0,
                        end: 0.1,
                        size: 8,
                        base: 10.0,
                        name: "embs_temp"
                    },
                    use_input_mask: true,
                    cuda_device: 0
                }
            ],
            k: 4.0,
            s: 1.0,
            beta: {
                class_name: "Hyperparam",
                name: "beta",
                values: [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 1.75],
            }
        },
        post_processing: post_processing,
        top_k: 10,
    }
}
