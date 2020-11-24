import React, { Component } from "react";
import { MDBBtn, MDBIcon } from "mdbreact";

import Spinner from "../Spinner";

import { addModel } from "../../utils/requests/state";
import { configUpload } from "../../utils/requests/upload";


export default class ModelConfigSelector extends Component {
    constructor(props) {
        super(props);
        this.state = {
            selectedFile: null,
            isLoading: false
        };
        this.handleSelect = this.handleSelect.bind(this);
        this.handleFileUpload = this.handleFileUpload.bind(this);
        this.handleAdd = this.handleAdd.bind(this);
        this.renderAddBtn = this.renderAddBtn.bind(this);
    }

    async handleFileUpload(event){
        event.preventDefault();
        console.log("Uploading file:", this.state.selectedFile);
        let uploadedResponse = await configUpload(this.state.selectedFile);
        if (uploadedResponse.status === 500) {
            this.props.msgHandler(
                "Invalid Config file!!!",
                uploadedResponse.err_msg,
                "red-text"
            );
        }
        console.log("Successfully upload:", uploadedResponse);
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
        let uploadBtn = document.getElementById("config-upload-btn");
        if (uploadBtn) {
            uploadBtn.click();
        }
    }

    handleAdd() {
        if (!this.state.selectedFile) {
            return;
        }
        this.setState({ isLoading: true });
        addModel(this.state.selectedFile.name, (response) => {
            this.props.onConfigSelect();
            this.setState(
                {
                    selectedFile: null,
                    isLoading: false
                },
            () => {
                    if (response.status === 500) {
                        this.props.msgHandler(
                            response.err_msg,
                            "Please, check that your config is properly formatted and compiled!",
                            "red-text",
                            -1
                        );
                    } else {
                        console.log("Successfully added model.");
                    }
                }
            );
        });
    }

    renderAddBtn() {
        if (!this.state.isLoading) {
            return (
                <MDBBtn
                    color={"primary"}
                    onClick={this.handleAdd}
                >
                    <MDBIcon
                        style={{
                            paddingRight: "5px",
                            fontSize: "1rem"
                        }}
                        icon="plus"
                    />
                    Add model
                </MDBBtn>
            );
        } else {
            return (
                <Spinner />
            );
        }
    }

    render() {
        return (
            <>
                { this.renderAddBtn() }
                <form
                    className={"text-center"}
                    onSubmit={this.handleFileUpload}
                >
                    <input
                        type={"file"}
                        name={"file"}
                        id={"file"}
                        className={"input-file"}
                        onChange={this.handleSelect}
                    />
                    <label
                        className={"btn btn-primary"}
                        htmlFor={"file"}
                    >
                        <MDBIcon
                            style={{
                                paddingRight: "5px",
                                fontSize: "1rem"
                            }}
                            icon="file-upload"
                        />
                        { this.state.selectedFile ? this.state.selectedFile.name : "Choose a file" }
                        <button
                            id={"config-upload-btn"}
                            type={"submit"}
                            style={{display: "none"}}
                        />
                    </label>
                </form>
            </>
        );
    }
}