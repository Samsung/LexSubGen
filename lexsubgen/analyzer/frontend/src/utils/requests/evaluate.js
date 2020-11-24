import axios from "axios";

import { processModelsResponse } from "../responses";
import { requestParams } from "../utils";
import { removeGenerator, getModelsInfo } from "./state";

export function getProgress(modelHost, modelPort, handler) {
    console.log("Requesting progress from", modelHost, modelPort);
    let requestUrl = `http://${modelHost}:${modelPort}/get_progress`;
    let response = progressRequest(requestUrl);
    return handler(response);
}

export async function progressRequest(url) {
    try {
        let response = await axios.get(url, requestParams);
        return response.data;
    } catch (error) {
        console.log("Request 'Get Progress' error:", error);
    }

}

export function evaluate(datasetName, taskName, handler, initHandler) {
    let payload = {
        "dataset_name": datasetName,
        "task_name": taskName
    };
    let models = getModelsInfo(() => {
    });
    models.then(
        async (r) => {
            let isParallelRequests = r.is_parallel;
            let modelsInfo = processModelsResponse(r);
            let activeModelsInfo = [];
            let model2idx = {};
            for (let i = 0; i < modelsInfo.length; i++) {
                if (modelsInfo[i].is_active) {
                    model2idx[modelsInfo[i].name] = activeModelsInfo.length;
                    activeModelsInfo.push(modelsInfo[i]);
                }
            }
            initHandler(activeModelsInfo);

            for (let i = 0; i < modelsInfo.length; i++) {
                let model = modelsInfo[i];
                if (model.is_active) {
                    let requestUrl = `http://${model.host}:${model.port}/evaluate`;
                    if (isParallelRequests) {
                        console.log("Requesting to model:", i + 1);
                        let evaluateResponse = evaluateRequest(requestUrl, payload);
                        // eslint-disable-next-line
                        evaluateResponse.then(res => {
                            handler(res.data, model2idx[model.name]);
                            console.log("Response from model: ", i + 1, res.data);
                        });
                    } else {
                        // Synchronous requests (Sequential)
                        console.log("Requesting to model:", i + 1);
                        let evaluateResponse = await evaluateRequest(requestUrl, payload);
                        // eslint-disable-next-line no-unused-vars
                        let removeResponse = await removeGenerator(model);
                        handler(evaluateResponse.data, model2idx[model.name]);
                        console.log("Response from model:", i + 1, evaluateResponse.data);
                    }
                }
            }
        }
    );
}

export function evaluateRequest(url, data) {
    try {
        return axios.post(url, data, requestParams);
    } catch (error) {
        console.log("Request 'Evaluate' error:", error);
    }
}
