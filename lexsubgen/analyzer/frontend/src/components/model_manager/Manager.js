import React, { Component } from "react";
import {
    MDBBtn, MDBModal, MDBModalBody,
    MDBModalHeader, MDBModalFooter, MDBIcon
} from "mdbreact";

import ModelsInfo from "./ModelsInfo";
import ModelConfigSelector from "./ModelConfigSelector";
import Switcher from "./Switcher";
import { getModelsInfo } from "../../utils/requests/state";
import { processModelsResponse } from "../../utils/responses";


export default class Manager extends Component {
    constructor(props) {
        super(props);
        this.state = {
            isOpen: false,
            isParallel: false,
            models: []
        };
        this.toggle = this.toggle.bind(this);
        this.refreshInfo = this.refreshInfo.bind(this);
        this.handleModelsRequest = this.handleModelsRequest.bind(this);
    }

    handleModelsRequest(response) {
        let isParallel = response.is_parallel;
        let models = processModelsResponse(response);
        this.setState(
            {
                models: models,
                isParallel: isParallel
            },
            () => {
                console.log("Response:", this.state.models);
            }
        );
    }

    refreshInfo() {
        getModelsInfo(this.handleModelsRequest);
    }

    toggle() {
        if (!this.state.isOpen) {
            this.refreshInfo();
        }
        this.setState(
            {
                isOpen: !this.state.isOpen
            },
            () => {
                console.log("Successfully toggling Models Manager!")
            }
        )
    }

    render() {
        return (
            <div className={"navbar-brand black-text"}>
                <MDBBtn
                    id={this.props.id}
                    color={this.props.btnColor}
                    onClick={this.toggle}
                >
                    <MDBIcon
                        icon={"cubes"}
                        style={{
                            paddingRight: "5px",
                            fontSize: "1.3rem"
                        }}
                    />
                    { this.props.modeText }
                </MDBBtn>
                <div
                    className={"manager"}
                >
                    <MDBModal
                        isOpen={this.state.isOpen}
                        toggle={this.toggle}
                        size={"fluid"}
                    >
                        <MDBModalHeader toggle={this.toggle}>
                            Model Manager dialog
                            <MDBBtn
                                color={"primary"}
                                size={"sm"}
                                style={{
                                    marginLeft: "10%"
                                }}
                                onClick={this.refreshInfo}
                            >
                                <MDBIcon
                                    icon={"redo"}
                                    style={{
                                        paddingRight: "5px"
                                    }}
                                />
                                Refresh info
                            </MDBBtn>
                        </MDBModalHeader>
                        <MDBModalBody>
                            <ModelsInfo
                                models={this.state.models}
                                onModelInfoChange={this.refreshInfo}
                            />
                        </MDBModalBody>
                        <MDBModalFooter>
                            <Switcher
                                checked={this.state.isParallel}
                                onSwitch={this.refreshInfo}
                            />
                            <ModelConfigSelector
                                onConfigSelect={this.refreshInfo}
                                msgHandler={this.props.msgHandler}
                            />
                            <MDBBtn
                                color={"danger"}
                                onClick={this.toggle}
                            >
                                Close
                            </MDBBtn>
                        </MDBModalFooter>
                    </MDBModal>
                </div>
            </div>
        );
    }
}