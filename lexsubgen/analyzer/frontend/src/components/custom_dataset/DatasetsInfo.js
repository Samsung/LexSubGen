import React, {Component} from "react";
import {MDBDropdownItem, MDBDropdownMenu} from "mdbreact";

import DatasetUploader from "./DatasetUploader";
import Spinner from "../Spinner";


export default class DatasetsInfo extends Component {
    constructor(props) {
        super(props);
        this.state = {};

        this.renderItems = this.renderItems.bind(this);
        this.renderUploader = this.renderUploader.bind(this);
    }

    renderUploader() {
        if (!this.props.withUploading) {
            return null;
        }
        return ([
            <MDBDropdownItem
                key={"divider"}
                divider
            />,
            <DatasetUploader
                key={"uploader"}
                refresher={this.props.refresher}
                msgHandler={this.props.msgHandler}
            />
        ]);
    }

    renderItems() {
        if (!this.props.datasets) {
            return (
                <div
                    className={"text-center"}
                >
                    <Spinner/>
                </div>
            );
        }
        if (!this.props.datasets.length) {
            return (
                <div
                    className={"text-center"}
                >
                    <p
                        className={"lead"}
                    >
                        Datasets not found!
                    </p>
                    <p
                        className={"lead"}
                    >
                        Upload your first Dataset.
                    </p>
                </div>
            );
        }
        return this.props.datasets.map(
            (dataset, index) => {
                return (
                    <MDBDropdownItem
                        key={index}
                        onClick={
                            () => {
                                this.props.handler(dataset);
                            }
                        }
                    >
                        {index + 1}. {dataset}
                    </MDBDropdownItem>
                );
            }
        );
    }

    render() {
        return (
            <MDBDropdownMenu
                className={"force-scroll"}
                flip
                basic
            >
                { this.renderItems() }
                { this.renderUploader() }
            </MDBDropdownMenu>
        );
    }
}