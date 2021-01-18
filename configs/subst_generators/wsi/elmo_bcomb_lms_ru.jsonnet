{
    class_name: "SubstituteGenerator",
    prob_estimator: {
        class_name: "prob_estimators.combiner.BcombLmsCombiner",
        prob_estimators: [
            {
                class_name: "prob_estimators.elmo_estimator.ElmoProbEstimator",
                model_name: "elmo-ru-news",
                weights_path: null,
                cutoff_vocab: 250000,
                add_bias: false,
                embedding_similarity: false,
                direction: "forward",
                temperature: 1.0,
                cuda_device: 0,
            },
            {
                class_name: "prob_estimators.elmo_estimator.ElmoProbEstimator",
                model_name: "elmo-ru-news",
                weights_path: null,
                cutoff_vocab: 250000,
                add_bias: false,
                embedding_similarity: false,
                direction: "backward",
                temperature: 1.0,
                cuda_device: 0,
            }
        ],
        alpha: 1.0,
        beta: 2.0
    },
    substitute_handler: {
        class_name: "post_processors.substitute_handler.SubstituteHandler",
        lemmatizer: "pymorphy-ru",
        exclude_target: true
    },
    top_k: 200
}