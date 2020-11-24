[
    {
        class_name: 'post_processors.base_postprocessor.LowercasePostProcessor'
    },
    {
        class_name: "post_processors.lemmatizer.Lemmatizer",
        lemmatizer: "nltk",
        strategy: "max"
    },
    {
        class_name: "post_processors.target_excluder.TargetExcluder",
        lemmatizer: "spacy_old"
    }
]
