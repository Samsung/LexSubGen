{
    class_name: "prob_estimators.combiner.BcombCombiner",
    prob_estimators: [
        {
            class_name: "prob_estimators.context2vec_estimator.Context2VecProbEstimator",
            model_name: "c2v_ukwac",
            embedding_similarity: false,
            temperature: 1.0,
            verbose: false
        },
        {
            class_name: "prob_estimators.context2vec_estimator.Context2VecProbEstimator",
            model_name: "c2v_ukwac",
            embedding_similarity: true,
            temperature: 1.0,
            verbose: false
        }
    ],
    verbose: false
}