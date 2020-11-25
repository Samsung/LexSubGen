{
    class_name: "prob_estimators.combiner.BcombCombiner",
    prob_estimators: [
        {
            class_name: "prob_estimators.bert_estimator.BertProbEstimator",
            mask_type: "not_masked",
            model_name: "bert-large-cased",
            embedding_similarity: false,
            temperature: 1.0,
            use_attention_mask: true,
            cuda_device: 0,
            verbose: false
        },
        {
            class_name: "prob_estimators.bert_estimator.BertProbEstimator",
            mask_type: "not_masked",
            model_name: "bert-large-cased",
            embedding_similarity: true,
            temperature: 0.1,
            use_attention_mask: true,
            use_subword_mean: true,
            cuda_device: 0,
            verbose: false
        }
    ],
    verbose: false
}
