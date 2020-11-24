function processModelsResponse(response) {
    let responseStatus = response.status;
    let models = [];

    if (responseStatus === 200) {
        delete response.status;
        delete response.name;
        delete response.is_parallel;
        models = Object.keys(response).map(idx => response[idx]);
    } else {
        console.log("Something goes wrong! Response code:", responseStatus);
    }
    return models;
}

function processSingleSubstResponse(response) {
    return response[0];
}

function processBatchSubstResponse(batch, batchSize) {
    let response = [];
    for (let i = 0; i < batchSize; i++) {
        response.push(batch[i]);
    }
    return response;
}

function processSingleEvaluationResponse(response) {
    delete response.name;
    return response;
}

function processDatasetLoading(response) {
    let datasetResponse = {
        status: response.status,
        sentences: [],
        goldSubsts: [],
        posTags: [],
        annotations: [],
        candidates: [],
        wordnetRelations: []
    };

    if (datasetResponse.status === 200) {
        Object.keys(response).map(
            // eslint-disable-next-line array-callback-return
            (key) => {
                if (key !== "name" && key !== "status") {
                    let sample = response[key];
                    if (!sample) {
                        return null;
                    }
                    datasetResponse.sentences.push(sample.context);
                    datasetResponse.goldSubsts.push(sample.gold_substitutes);
                    datasetResponse.posTags.push(sample.pos_tag);
                    datasetResponse.annotations.push(sample.annotations);
                    datasetResponse.candidates.push(sample.candidates);
                    datasetResponse.wordnetRelations.push(sample.gold_wordnet_relations);
                }
            }
        );
    }
    return datasetResponse;
}

export {
    processModelsResponse,
    processSingleSubstResponse, processBatchSubstResponse,
    processSingleEvaluationResponse, processDatasetLoading
};
