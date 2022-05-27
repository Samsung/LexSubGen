local topk = 15;

{
    class_name: "prob_estimators.mt5_estimator.MT5ProbEstimator",
    model_name: "google/mt5-large",
    cuda_device: 0,
    num_beams: topk,
    num_return_sequences: topk,
    handler_span: "max_length", 
    verbose: false,
    eos_token: "<extra_id_1>"
}