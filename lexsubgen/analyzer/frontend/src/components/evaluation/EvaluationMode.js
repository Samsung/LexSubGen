import React, { Component } from "react";
import DatasetPicker from "../custom_dataset/DatasetPicker";
import EvaluationResult from "./EvaluationResult";
import { evaluate } from "../../utils/requests/evaluate";
import { processSingleEvaluationResponse } from "../../utils/responses";



export default class EvaluationMode extends Component {
    constructor(props) {
        super(props);
        this.state = {
            datasetName: "",
            evaluationResults: null,
        };
        this.handleDatasetPick = this.handleDatasetPick.bind(this);
        this.initHandler = this.initHandler.bind(this);
        this.handleEvaluateResponse = this.handleEvaluateResponse.bind(this);
    }

    initHandler(response) {
        this.setState(
            {
                evaluationResults: response
            },
            () => {
                console.log("Initialize Evaluate Response:", this.state.evaluationResults);
            }
        );
    }

    handleEvaluateResponse(response, idx) {
        let modelName = response.name;
        let responseStatus = response.status;
        if (responseStatus === 200) {
            let currentEvalResponse = this.state.evaluationResults;
            let newMetrics = processSingleEvaluationResponse(response);
            currentEvalResponse[idx] = {
                modelName: modelName,
                response: newMetrics,
            };
            this.setState(
                {
                    evaluationResults: currentEvalResponse
                },
                () => {
                    console.log(this.state.evaluationResults);
                }
            );
        } else {
            this.props.msgHandler(
                `Get "Evaluation Request" error: status code ${responseStatus}`,
                <div>
                    <p>{response.err_msg}</p>
                    <p>{response.traceback}</p>
                </div>,
                "red-text"
            );
        }
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
                evaluate(dataset, "lexsub", this.handleEvaluateResponse, this.initHandler)
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
                    Evaluation mode.
                </h1>
                <div
                    className={"container"}
                    style={{width: "40%"}}
                >
                    <p key={"1"} className={"text-left"}>
                        1. Choose Dataset from existing datasets.
                    </p>
                    <p key={"2"} className={"text-left"}>
                        2. Click on this button to choose Dataset. After that, App will run Evaluation.
                    </p>
                </div>
                <DatasetPicker
                    withUploading={false}
                    onDatasetPick={this.handleDatasetPick}
                    msgHandler={this.props.msgHandler}
                />
                <EvaluationResult
                    datasetName={this.state.datasetName}
                    data={this.state.evaluationResults}
                />
            </div>
        );
    }
}