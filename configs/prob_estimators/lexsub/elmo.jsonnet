{
    class_name: "prob_estimators.combiner.BcombLmsCombiner",
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
        }
    ],
    alpha: 1.0,
    beta: 0.5,
}
