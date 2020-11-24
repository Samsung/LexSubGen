import React, { Component } from "react";
import { MDBNavbar, MDBNavbarBrand } from "mdbreact";

import ModeChanger from "./ModeChanger";
import Manager from "./model_manager/Manager";


export default class Navbar extends Component {
    render() {
        return (
            <MDBNavbar id="navbar" dark expand="md">
                <img 
                    id={"logo"}
                    alt={"Samsung"}
                    src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Samsung_Logo.svg/512px-Samsung_Logo.svg.png"
                />
                <MDBNavbarBrand
                    id={"logo-text"}
                >
                    SRR R&D
                </MDBNavbarBrand>

                <ModeChanger
                    id={"evaluation-btn"}
                    active={this.props.currentMode === "evaluation"}
                    btnColor={"primary"}
                    iconName={"balance-scale"}
                    onModeChange={this.props.modeHandler}
                    modeName={"evaluation"}
                    modeText={"Evaluation"}
                    msgHandler={this.props.msgHandler}
                />

                <ModeChanger
                    id={"custom-sentence-btn"}
                    active={this.props.currentMode === "custom_sentence"}
                    btnColor={"primary"}
                    iconName={"user-edit"}
                    onModeChange={this.props.modeHandler}
                    modeName={"custom_sentence"}
                    modeText={"Custom sentence"}
                    msgHandler={this.props.msgHandler}
                />

                <ModeChanger
                    id={"custom-dataset-btn"}
                    active={this.props.currentMode === "custom_dataset"}
                    btnColor={"primary"}
                    iconName={"file-upload"}
                    onModeChange={this.props.modeHandler}
                    modeName={"custom_dataset"}
                    modeText={"Samples from dataset"}
                    msgHandler={this.props.msgHandler}
                />

                <Manager
                    id={"manager-btn"}
                    btnColor={"primary"}
                    modeText={"Model manager"}
                    msgHandler={this.props.msgHandler}
                />
            </MDBNavbar>
        );
    }
}
