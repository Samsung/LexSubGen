local generator = import '../subst_generators/lexsub/xlnet.jsonnet';

{
    host: "106.109.129.52",
    port: 5002,
    name: "XLNet",
    verbose: false,
    substitute_generator: generator
}
