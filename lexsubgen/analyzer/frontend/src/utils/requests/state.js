import {baseURL, requestParams} from "../utils";
import axios from "axios";

export function switchRequest(refreshCallback) {
    let requestUrl = `${baseURL}/switch_parallel`;
    axios.get(requestUrl, requestParams)
         .then(response => { refreshCallback(); })
         .catch(error => {
            console.log("Request 'Switch Requests' error:", error);
         });
}

export async function getModelsInfo(handler) {
    let requestUrl = `${baseURL}/get_models`;

    try {
        const response = await axios.get(requestUrl, requestParams);
        let data = response.data;
        handler(data);
        return data;
    } catch (error) {
        console.log("Request 'Get Models Info' error", error);
    }
}

export function changeModelStatus(modelIdx, refreshCallback) {
    let requestUrl = `${baseURL}/change_status`;
    let payload = { "model_idx": modelIdx };
    axios.post(requestUrl, payload, requestParams)
         .then(response => {
          refreshCallback();
         })
         .catch(error => {
            console.log("Request 'Change Model Status' error:", error);
         });
}

export async function pingModel(modelIdx) {
    let models = await getModelsInfo(() => {});
    let model = models[modelIdx];
    console.log(model);
    let host = model.host;
    let port = model.port;
    let url = `http://${host}:${port}/ping`;
    let status = false;
    try {
        await axios.get(url, requestParams);
        status =  true;
    } catch (e) {
        console.log(e);
    }
 return status;
}

export function addModel(configFileName, refreshCallback) {
    let requestUrl = `${baseURL}/add_model`;
    let payload = {
        "config_path": configFileName
    };
    return axios
        .post(requestUrl, payload, requestParams)
        .then(response => {
            let data = response.data;
            console.log(data);
            refreshCallback(data);
        })
        .catch(error => {
            console.log("Request 'Add Model' error:", error);
        });
}

export function removeModel(modelIdx, refreshCallback) {
    let requestUrl = `${baseURL}/remove_model`;
    let payload = {
        "model_idx": modelIdx
    };
    axios.post(requestUrl, payload, requestParams)
         .then(response => {
             let data = response.data;
             console.log(data);
             refreshCallback();
         })
         .catch(error => {
             console.log("Request 'Remove Model' error:", error);
         });
}

export function getWordnetRelations(handler) {
    let requestUrl = `${baseURL}/wordnet_relations`;
    axios.get(requestUrl, requestParams)
         .then(response => {
             let data = response.data;
             let relations = data.relations;
             handler(relations);
         })
         .catch(error => {
            console.log("Request 'WordNet Relations' error:", error);
         });
}

export function removeGenerator(model) {
    let requestUrl = `http://${model.host}:${model.port}/remove_generator`;
    try {
        return axios.get(requestUrl, requestParams);
    } catch (error) {
        console.log("Request 'Remove Generator' error:", error);
    }
}

export async function getDatasetNames(isCustom, handler) {
    let requestUrl = `${baseURL}/dataset_names`;
    let payload = { "is_custom": isCustom };
    let response = await datasetNamesRequest(requestUrl, payload);
    handler(response.dataset_names);
}

export async function datasetNamesRequest(url, data) {
    try {
        let response = await axios.post(url, data, requestParams);
        return response.data;
    } catch (error) {
        console.log("Request 'Get Custom Dataset Names' error:", error);
    }
}
