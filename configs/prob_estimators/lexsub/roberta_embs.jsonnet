{
    class_name: "prob_estimators.combiner.BcombCombiner",
    prob_estimators: [
        {
            class_name: "prob_estimators.roberta_estimator.RobertaProbEstimator",
            mask_type: "masked",
            model_name: "roberta-large",
            embedding_similarity: false,
            temperature: 1.0,
            use_attention_mask: true,
            cuda_device: 0,
            verbose: false
        },
        {
            class_name: "prob_estimators.roberta_estimator.RobertaProbEstimator",
            mask_type: "not_masked",
            model_name: "roberta-large",
            embedding_similarity: true,
            sim_func: "cosine",
            unk_word_embedding: "first_subtoken",
            temperature: 0.025,
            use_attention_mask: true,
            cuda_device: 0,
            verbose: false
        }
    ],
    verbose: false,
    k: 4.0,
    s: 1.05,
    beta: 0.0
}