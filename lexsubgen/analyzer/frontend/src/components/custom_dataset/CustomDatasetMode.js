import React, { Component } from "react";

import DatasetPicker from "./DatasetPicker";
import SubstitutesResponse from "./SubstitutesResponse";

import { loadDataset } from "../../utils/requests/upload";
import { getSubstitutes } from "../../utils/requests/substitutes";
import { processBatchSubstResponse } from "../../utils/responses";

export default class CustomDatasetMode extends Component {
    constructor(props) {
        super(props);
        this.state = {
            datasetName: "",
            dataset: {},
            substitutesData: null
        };
        this.handleDatasetPick = this.handleDatasetPick.bind(this);
        this.handleDatasetLoad = this.handleDatasetLoad.bind(this);
        this.getSamplesFromDataset = this.getSamplesFromDataset.bind(this);

        this.initHandler = this.initHandler.bind(this);
        this.handleSubstitutes = this.handleSubstitutes.bind(this);
    }

    async initHandler(numberOfModels) {
        let emptyData = [];
        for (let i = 0; i < this.state.dataset.sentences.length; i++) {
            let newSubstResponseState = [];
            // For gold substitutes row in table
            let wordnetRels = null;
            if (this.state.dataset.wordnetRels)
                wordnetRels = this.state.dataset.wordnetRels[i];
            let annotations = null;
            if (this.state.dataset.annotations) {
                annotations = this.state.dataset.annotations[i];
            }
            let goldSubstitutes = this.state.dataset.goldSubsts[i].map(
                (w, idx) => {
                    return {
                        word: w,
                        rank: null,
                        score: annotations ? annotations[idx] : null,
                        wordnet_relation: wordnetRels ? wordnetRels[idx] : null
                    }
                }
            );
            newSubstResponseState.push({
                modelName: "Gold Standard",
                response: {
                    generated_substitutes: goldSubstitutes
                }
            });
            for (let j = 0; j < numberOfModels; j++) {
               newSubstResponseState.push({});
            }
            emptyData.push(newSubstResponseState);
        }
        await this.setState(
            {
                substitutesData: emptyData
            },
            () => {
                console.log("Initialized emtpy substitutesData:", this.state.substitutesData);
            }
        );
    }

    handleSubstitutes(batchOfSubstitutes, modelIndex) {
        let modelName = batchOfSubstitutes.name;
        let responseStatus = batchOfSubstitutes.status;
        if (responseStatus === 200) {
            let currentSubstitutes = this.state.substitutesData;
            let numberOfSentences = this.state.dataset.sentences.length;
            let newSubstitutes = processBatchSubstResponse(batchOfSubstitutes, numberOfSentences);
            for (let i = 0; i < numberOfSentences; i++) {
                 currentSubstitutes[i][modelIndex + 1] = {
                    modelName: modelName,
                    response: newSubstitutes[i]
                 };
            }
            this.setState(
                {
                    substitutesResponse: currentSubstitutes
                },
                () => {
                    console.log("Successfully processed response:",  this.state.substitutesData);
                }
            );
        } else {
            this.props.msgHandler(
                `Get "Substitutes Request" error: status code ${responseStatus}`,
                <div>
                    <p>{batchOfSubstitutes.err_msg}</p>
                    <p>{batchOfSubstitutes.traceback}</p>
                </div>,
                "red-text"
            );
        }
    }

    async getSamplesFromDataset(datasetName) {
        await this.setState({ substitutesData: null });
        let loadStatus = await loadDataset(datasetName, this.handleDatasetLoad);
        if (!loadStatus) {
            return;
        }
        let { sentences, posTags, goldSubsts, annotations, candidates }  = this.state.dataset;
        getSubstitutes(
            sentences,
            posTags,
            goldSubsts,
            annotations,
            candidates,
            this.handleSubstitutes,
            this.initHandler
        );
    }

    handleDatasetLoad(response) {
        if (response.status === 500) {
            this.props.msgHandler(
                "Error occurred while loading Dataset!!!",
                response.err_msg,
                "red-text"
            );
            return false;
        }
        this.setState(
            {
                dataset: {
                    goldSubsts: response.goldSubsts,
                    sentences: response.sentences,
                    wordnetRels: response.wordnetRelations,
                    posTags: response.posTags,
                    annotations: response.annotations,
                    candidates: response.candidates,
                }
            },
            () => {
                console.log("Successfully loaded:", this.state.dataset);
            }
        );
        return true;
    }

    handleDatasetPick(dataset) {
        if (dataset === this.state.datasetName) {
            return;
        }
        this.setState(
            {
                datasetName: dataset
            },
            () => {
                console.log("Successfully picked", this.state.datasetName, "dataset!");
                this.getSamplesFromDataset(this.state.datasetName);
            }
        );
    }

    render() {
        return (
            <div
                className={"lead"}
                style={{display: this.props.visible ? "block" : "none"}}
            >
                <h1>
                    Custom Dataset mode.
                </h1>
                <div
                    className={"container"}
                    style={{width: "40%"}}
                >
                    <p key={"1"} className={"text-left"}>
                        1. Choose Dataset from existing datasets or upload your Custom Dataset.
                    </p>
                    <p key={"2"} className={"text-left"}>
                        2. Click on this button to choose or to add Dataset.
                    </p>
                </div>
                <DatasetPicker
                    onDatasetPick={this.handleDatasetPick}
                    msgHandler={this.props.msgHandler}
                    withUploading={true}
                    custom
                />
                <SubstitutesResponse
                    datasetName={this.state.datasetName}
                    msgHandler={this.props.msgHandler}
                    sentences={this.state.dataset.sentences}
                    responseData={this.state.substitutesData}
                />
            </div>
        );
    }
}