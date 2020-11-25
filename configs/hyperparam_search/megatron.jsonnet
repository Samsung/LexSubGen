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
                class_name: "pre_processors.base_preprocessors.AndPreprocessor"
            },
            {
                class_name: "pre_processors.base_preprocessors.AddPunctPreprocessor"
            }
        ],
        prob_estimator: {
        class_name: "prob_estimators.combiner.BcombCombiner",
        prob_estimators: [
            {
                class_name: "prob_estimators.bert_estimator.BertProbEstimator",
                mask_type: "not_masked",
                model_name: "bert-large-cased",
                temperature: 1.0,
                use_attention_mask: true,
                cuda_device: 0
            },
            {
                class_name: "prob_estimators.bert_estimator.BertProbEstimator",
                mask_type: "masked",
                model_name: "bert-large-cased",
                temperature: {
                    class_name: "LinspaceHyperparam",
                    start: 0.5,
                    end: 1.5,
                    size: 5,
                    name: "masked_model_temp"
                },
                use_attention_mask: true,
                cuda_device: 0
            },
            {
                class_name: "prob_estimators.bert_estimator.BertProbEstimator",
                model_name: "bert-large-cased",
                embedding_similarity: true,
                temperature: {
                    class_name: "LinspaceHyperparam",
                    start: 0.05,
                    end: 1.5,
                    size: 10,
                    name: "embs_temp"
                },
                use_subword_mean: true,
                cuda_device: 0
            }
        ],
        k: {
            class_name: "LogspaceHyperparam",
            start: 0.0,
            end: 2.0,
            size: 5,
            base: 10.0,
            name: "k"
        },
        s: 1.0,
        beta: {
            class_name: "Hyperparam",
            name: "beta",
            values: [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 1.75],
        },
    },
        post_processing: post_processing,
        top_k: 10,
    }
}
