import React, { Component } from "react";
import {
    MDBTable, MDBTableBody, MDBTableHead,
    MDBIcon
} from "mdbreact";

import EvaluationProgress from "./EvaluationProgress";

export default class EvaluationResult extends Component {
    constructor(props) {
        super(props);
        this.state = {};
        this.renderTableHeader = this.renderTableHeader.bind(this);
        this.renderTableBody = this.renderTableBody.bind(this);
        this.renderTableRow = this.renderTableRow.bind(this);
    }

    renderTableRow(model) {
        if (model.response) {
            let metrics = model.response;
            return Object.keys(metrics).map(
                (key, idx) => {
                    if (key === "name" || key === "status") {
                        return null;
                    } else {
                        return (
                            <th key={"metric" + idx}>
                                {metrics[key]}
                            </th>
                        );
                    }
                }
            );
        } else {
            return (
                <th
                >
                    <EvaluationProgress
                        host={model.host}
                        port={model.port}
                    />
                </th>
            );
        }
    }

    renderTableHeader(data) {
        if (!data) {
            return null;
        }
        let anyoneReady = false;
        let metrics = [];
        for (let i = 0; i < data.length; i++) {
            let response = data[i];
            if (response.modelName) {
                anyoneReady = true;
                metrics = response.response;
                break;
            }
        }
        if (anyoneReady) {
            return Object.keys(metrics).map(
                (key, idx) => {
                    if (key === "name" || key === "status") {
                        return null;
                    } else {
                        return (
                            <th
                                key={"metric" + idx}
                                className={"custom-sentence-header"}
                            >
                                {key}
                            </th>
                        );
                    }
                }
            );
        } else {
            return (
                <th
                    key={"progress"}
                    className={"custom-sentence-header"}
                >
                    Evaluation Progress
                </th>
            );
        }

    }

    renderTableBody(data) {
        if (!data) {
            return null;
        }
        return data.map(
            (model, idx) => {
                return (
                    <tr
                        key={idx}
                    >
                        <th
                            className={"text-left"}
                        >
                            {model.modelName || model.name}
                        </th>
                        {this.renderTableRow(model)}
                    </tr>
                );
            }
        );
    }

    render() {
        if (!this.props.datasetName) {
            return null;
        }
        return(
            <div
                className={"evaluation-response"}
            >
                <p
                    className={"my-2 input-sentence"}
                >
                    You choose <b>{this.props.datasetName}</b> dataset for evaluation.
                </p>
                <MDBTable
                    id={"evaluation-metrics-table"}
                >
                    <MDBTableHead
                    >
                        <tr>
                            <th
                                className={"custom-sentence-header"}
                            >
                                <MDBIcon
                                    className={"model-icon"}
                                    icon={"server"}
                                />
                                Model Name
                            </th>
                            { this.renderTableHeader(this.props.data) }
                        </tr>
                    </MDBTableHead>
                    <MDBTableBody>
                        { this.renderTableBody(this.props.data) }
                    </MDBTableBody>
                </MDBTable>
            </div>
        );
    }
}