{
    class_name: "prob_estimators.combiner.BcombCombiner",
    prob_estimators: [
        {
            class_name: "prob_estimators.xlnet_estimator.XLNetProbEstimator",
            masked: false,
            model_name: "xlnet-large-cased",
            embedding_similarity: false,
            temperature: 1.0,
            use_input_mask: true,
            multi_subword: false,
            cuda_device: 0,
            verbose: false
        },
        {
            class_name: "prob_estimators.xlnet_estimator.XLNetProbEstimator",
            masked: false,
            model_name: "xlnet-large-cased",
            embedding_similarity: true,
            temperature: 5.0,
            use_input_mask: true,
            multi_subword: false,
            cuda_device: 0,
            use_subword_mean: true,
            verbose: false
        }
    ],
    verbose: false,
    k: 4.0,
    s: 1.05,
    beta: 2.0
}
