local generator = import '../../subst_generators/lexsub/elmo_embs.jsonnet';
local reader = import '../../dataset_readers/lexsub/coinco.jsonnet';

{
    class_name: "evaluations.lexsub.LexSubEvaluation",
    substitute_generator: generator,
    dataset_reader: reader,
    batch_size: 100,
    verbose: false
}
