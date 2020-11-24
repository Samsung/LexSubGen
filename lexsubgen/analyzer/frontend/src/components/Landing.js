import React, {Component} from "react";

import Navbar from "./Navbar";
import CustomSentenceMode from "./custom_sentence/CustomSentenceMode";
import CustomDatasetMode from "./custom_dataset/CustomDatasetMode";
import EvaluationMode from "./evaluation/EvaluationMode";
import InfoMessage from "./InfoMessage";
import ScrollToTopArrow from "./ScrollToTopArrow";


export default class Landing extends Component {
    constructor(props) {
        super(props);
        this.state = {
            currentMode: "custom_dataset",
        };
        this.modeChangeHandler = this.modeChangeHandler.bind(this);
        this.changeInfoMsgState = this.changeInfoMsgState.bind(this);
        this.renderCurrentMode = this.renderCurrentMode.bind(this);
    }

    modeChangeHandler(mode) {
        this.setState(
            {
                currentMode: mode,
                infoMsgTitle: "",
                infoMsgText: "",
                titleColor: ""
            },
            () => {
                console.log("Mode changed to: " + this.state.currentMode);
            }
        )
    }

    changeInfoMsgState(newTitle, newText, titleColor="black-text", timeout=-1) {
        this.setState(
            {
                infoMsgTitle: newTitle,
                infoMsgText: newText,
                titleColor: titleColor
            },
            () => {
                console.log("New Message:", this.state.infoMsgTitle, this.state.infoMsgText);
            }
        );
        let showMsgBtn = document.getElementById("show-msg-btn");
        if (showMsgBtn) {
            showMsgBtn.click();
        }
        if (timeout === -1) {
            return;
        }
        setTimeout(
            () => {
                let closeMsgBtn = document.getElementById("close-msg-btn");
                if (closeMsgBtn) {
                    closeMsgBtn.click();
                }
            },
            timeout * 1000
        );
    }

    renderCurrentMode() {
        let currentMode = this.state.currentMode;
        return ([
            <EvaluationMode
                key={"evaluation-mode"}
                visible={currentMode === "evaluation"}
                msgHandler={this.changeInfoMsgState}
            />,
            <CustomSentenceMode
                key={"custom-sentence-mode"}
                visible={currentMode === "custom_sentence"}
                msgHandler={this.changeInfoMsgState}
            />,
            <CustomDatasetMode
                key={"custom-dataset-mode"}
                visible={currentMode === "custom_dataset"}
                msgHandler={this.changeInfoMsgState}
            />
        ]);
    }

    render() {
        return (
            <>
                <Navbar
                    key={"navbar"}
                    msgHandler={this.changeInfoMsgState}
                    modeHandler={this.modeChangeHandler}
                    currentMode={this.state.currentMode}
                />
                <ScrollToTopArrow />

                <div
                    id={"landing"}
                    key={"actions"}
                    className={"mt-3 text-center"}
                >
                    <h2 className="h1 display-3">Lexical Substitutions demo</h2>
                    <hr className="my-2"/>
                    <InfoMessage
                        msgTitle={this.state.infoMsgTitle}
                        msgText={this.state.infoMsgText}
                        titleColor={this.state.titleColor}
                    />
                    { this.renderCurrentMode() }
                </div>
            </>
        );
    }
}
