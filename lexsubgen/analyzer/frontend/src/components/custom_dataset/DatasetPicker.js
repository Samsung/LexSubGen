import React, {Component} from "react";
import {MDBDropdown, MDBDropdownToggle,} from "mdbreact";

import { getDatasetNames } from "../../utils/requests/state";

import DatasetsInfo from "./DatasetsInfo";

export default class DatasetPicker extends Component {
    constructor(props) {
        super(props);
        this.state = {
            datasetNames: null,
        };
        this.refreshInfo = this.refreshInfo.bind(this);
        this.handleDatasetNames = this.handleDatasetNames.bind(this);
        this.toggle = this.toggle.bind(this);
    }

    toggle() {
        this.refreshInfo();
    }

    handleDatasetNames(response) {
        this.setState(
            {
                datasetNames: response
            },
            () => {
                console.log("Dataset Names Response:", this.state.datasetNames);
            }
        );
    }

    refreshInfo() {
        getDatasetNames(this.props.custom, this.handleDatasetNames);
    }

    render() {
        return (
            <div id="dataset_picker">
                <MDBDropdown
                    dropright
                >
                    <MDBDropdownToggle
                        caret
                        color={"primary"}
                        onClick={this.toggle}
                    >
                        Choose dataset
                    </MDBDropdownToggle>
                    <DatasetsInfo
                        withUploading={this.props.withUploading}
                        handler={this.props.onDatasetPick}
                        refresher={this.refreshInfo}
                        datasets={this.state.datasetNames}
                        onDatasetInfoChange={this.refreshInfo}
                        msgHandler={this.props.msgHandler}
                    />
                </MDBDropdown>
            </div>
        );
    }
}