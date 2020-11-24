import {processSentences, requestParams} from "../utils";
import {getModelsInfo, removeGenerator} from "./state";
import {processModelsResponse} from "../responses";
import axios from "axios";

export function getSubstitutes(
    sentences,
    posTags,
    goldSubstitutes,
    goldWeights,
    candidates,
    handler,
    initHandler
) {
    let processed = processSentences(sentences);
    let payload = {
        "sentences": processed.tokensLists,
        "target_ids": processed.targetIds,
        "gold_substs": goldSubstitutes,
        "gold_weights": goldWeights,
        "candidates": candidates,
        "pos_tags": posTags
    };
    let response = getModelsInfo(() => {});
    response.then(
        async (r) => {
            let isParallelRequests = r.is_parallel;
            let modelsInfo = processModelsResponse(r);
            let numberOfActive = 0;
            let model2idx = {};
            for (let i = 0; i < modelsInfo.length; i++) {
                if (modelsInfo[i].is_active) {
                    model2idx[modelsInfo[i].name] = numberOfActive;
                    numberOfActive++;
                }
            }
            initHandler(numberOfActive);

            for (let i = 0; i < modelsInfo.length; i++) {
                let model = modelsInfo[i];
                if (model.is_active) {
                    let requestUrl = `http://${model.host}:${model.port}/get_subst`;
                    if (isParallelRequests) {
                        console.log("Requesting to model:", i + 1);
                        let substResponse = substitutesRequest(requestUrl, payload);
                        // eslint-disable-next-line
                        substResponse.then(res => {
                            handler(res.data, model2idx[model.name]);
                            console.log("Response from model: ", i + 1, res.data);
                        });
                    } else {
                        console.log("Requesting to model:", i + 1);
                        let substResponse = await substitutesRequest(requestUrl, payload);
                        // eslint-disable-next-line no-unused-vars
                        let removeResponse = await removeGenerator(model);
                        handler(substResponse.data, model2idx[model.name]);
                        console.log("Response from model:", i + 1, substResponse.data);
                    }
                }
            }
        }
    );
}

export function substitutesRequest(url, data) {
    try {
        return axios.post(url, data, requestParams);
    } catch (error) {
        console.log("Request 'Get Substitutes' error:", error);
    }
}
