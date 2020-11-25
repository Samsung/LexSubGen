{
    class_name: "prob_estimators.combiner.BcombCombiner",
    prob_estimators: [
        {
            class_name: "prob_estimators.elmo_estimator.ElmoProbEstimator",
            model_name: "elmo-en",
            cutoff_vocab: 150000,
            add_bias: false,
            embedding_similarity: false,
            direction: "forward",
            temperature: 1.0,
            cuda_device: 0,
        },
        {
            class_name: "prob_estimators.elmo_estimator.ElmoProbEstimator",
            model_name: "elmo-en",
            cutoff_vocab: 150000,
            add_bias: false,
            embedding_similarity: false,
            direction: "backward",
            temperature: 1.0,
            cuda_device: 0,
        },
        {
            class_name: "prob_estimators.elmo_estimator.ElmoProbEstimator",
            model_name: "elmo-en",
            cutoff_vocab: 150000,
            add_bias: false,
            embedding_similarity: true,
            direction: "both",
            temperature: 0.1,
            cuda_device: 0,
        }
    ],
    k: 4.0,
    s: 1.05,
    beta: 2.0,
}