local generator = import '../../subst_generators/lexsub/semeval_xlnet_embs.jsonnet';
local reader = import '../../dataset_readers/lexsub/semeval_all.jsonnet';

{
    class_name: "evaluations.lexsub.LexSubEvaluation",
    substitute_generator: generator,
    dataset_reader: reader,
    batch_size: 50,
    verbose: false
}
