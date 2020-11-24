import {baseURL, requestParams} from "../utils";
import axios from "axios";
import {processDatasetLoading} from "../responses";

export async function datasetUpload(uploadedFile) {
    let requestUrl = `${baseURL}/upload_dataset`;
    let formData = new FormData();
    let datasetName = uploadedFile.name;
    let content = await uploadedFile.text();
    formData.append("datasetName", datasetName);
    formData.append("datasetContent", content);
    let requestParams = {
        headers: {
            "Content-type": "multipart/form-data"
        }
    };
    try {
        let response = await fileUpload(requestUrl, formData, requestParams);
        return response.data;
    } catch (error) {
        console.log("Request 'Dataset Upload' error:", error);
    }
}

export async function configUpload(uploadedFile) {
    let requestUrl = `${baseURL}/upload_config`;
    let formData = new FormData();
    let configName = uploadedFile.name;
    let content = await uploadedFile.text();
    formData.append("configName", configName);
    formData.append("configContent", content);
    let requestParams = {
        headers: {
            "Content-type": "multipart/form-data"
        }
    };
    try {
        let response = await fileUpload(requestUrl, formData, requestParams);
        return response.data;
    } catch (error) {
        console.log("Request 'Config Upload' error:", error);
    }
}

export function fileUpload(url, data, params) {
    try {
        return axios.post(url, data, params);
    } catch (error) {
        console.log("Request 'File Upload' error:", error);
    }
}

export async function loadDataset(datasetName, handler) {
    let requestUrl = `${baseURL}/load_dataset`;
    let payload = {
        "dataset_name": datasetName
    };
    let response = await loadDatasetRequest(requestUrl, payload, requestParams);
    response = processDatasetLoading(response);
    return handler(response);
}

export async function loadDatasetRequest(url, data, params) {
    try {
        let response = await axios.post(url, data, params);
        return response.data;
    } catch (error) {
        console.log("Request 'Load Dataset' error:", error);
    }
}
