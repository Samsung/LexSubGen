import React, { Component } from "react";

import { MDBIcon } from "mdbreact";

import { datasetUpload } from "../../utils/requests/upload";


export default class DatasetUploader extends Component {
    constructor(props) {
        super(props);
        this.state = {
            selectedFile: null
        };
        this.handleSelect = this.handleSelect.bind(this);
        this.handleFileUpload = this.handleFileUpload.bind(this);
    }

    async handleFileUpload(event){
        event.preventDefault();
        console.log("Uploading file:", this.state.selectedFile);
        let uploadedResponse = await datasetUpload(this.state.selectedFile);
        if (uploadedResponse.status === 500) {
            this.props.msgHandler(
                "Invalid Dataset file!!!",
                uploadedResponse.err_msg,
                "red-text"
            );
        }
        console.log("Successfully upload:", uploadedResponse);
        this.props.refresher();
    }

    async handleSelect(event) {
        let selectedFile = event.target.files;
        if (!selectedFile.length) {
            return;
        }
        let fileName = selectedFile[0];
        await this.setState(
            {
                selectedFile: fileName
            },
            () => {
                console.log("Selected file:", this.state.selectedFile);
            }
        );
        let uploadBtn = document.getElementById("dataset-upload-btn");
        if (uploadBtn) {
            uploadBtn.click();
        }
    }

    render() {
        return (
            <form
                className={"text-center"}
                onSubmit={this.handleFileUpload}
            >
                <input
                    type={"file"}
                    id={"file"}
                    className={"input-file"}
                    onChange={this.handleSelect}
                />
                <label
                    className={"btn btn-sm btn-primary"}
                    htmlFor={"file"}
                >
                    <MDBIcon
                        style={{
                            paddingRight: "5px",
                            fontSize: "1rem"
                        }}
                        icon="file-upload"
                    />
                    Upload dataset
                </label>
                <button
                    id={"dataset-upload-btn"}
                    type={"submit"}
                    style={{display: "none"}}
                />
            </form>
        )
    }
}