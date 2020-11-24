import React, { Component } from "react";

import Input from "./Input";
import SubstitutesTable from "./SubstitutesTable";

import { getTargetWord } from "../../utils/utils";
import { getSubstitutes } from "../../utils/requests/substitutes";
import { processSingleSubstResponse} from "../../utils/responses";


export default  class CustomSentenceMode extends  Component {
    constructor(props) {
        super(props);
        this.state = {
            customSentence: "",
            targetWordPosTag: "n",
            substitutesResponse: []
        };
        this.sentenceInputHandler = this.sentenceInputHandler.bind(this);
        this.posTagHandler = this.posTagHandler.bind(this);
        this.submitSentence = this.submitSentence.bind(this);
        this.handleSubstitutes = this.handleSubstitutes.bind(this);
        this.handleInitSubstTable = this.handleInitSubstTable.bind(this);
    }

    handleInitSubstTable(numberOfModels) {
        let newSubstResponseState = [];
        for (let i = 0; i < numberOfModels; i++) {
            newSubstResponseState.push({});
        }
        this.setState(
            {
                substitutesResponse: newSubstResponseState
            },
            () => {
                console.log("Initialize Substitute Response:", this.state.substitutesResponse);
            }
        );
    }

    handleSubstitutes(response, idx) {
        let modelName = response.name;
        let responseStatus = response.status;
        if (responseStatus === 200) {
            let currentSubstitutes = this.state.substitutesResponse;
            let newSubstitutes = processSingleSubstResponse(response);
            currentSubstitutes[idx] = {
                modelName: modelName,
                response: newSubstitutes
            };
            this.setState(
                {
                    substitutesResponse: currentSubstitutes
                },
                () => {
                    console.log(this.state.substitutesResponse);
                }
            );
        } else {
            this.props.msgHandler(
                `Get "Substitutes Request" error: status code ${responseStatus}`,
                <div>
                    <p>{response.err_msg}</p>
                    <p>{response.traceback}</p>
                </div>,
                "red-text"
            );
        }
    }

    submitSentence() {
        this.setState({
            substitutesResponse: []
        });
        if (!this.state.customSentence) {
            return;
        }
        let submittedSentence = this.state.customSentence;
        let result = getTargetWord(submittedSentence);
        if (!result) {
            this.props.msgHandler(
                "Invalid sentence input!!!",
                "Don't forget to put \"@\" around target word.",
                "red-text"
            );
            return;
        } else if (result.length === 2) {
            this.props.msgHandler(
                "Empty target word!!!",
                `You input sentence with empty target word:\n ${submittedSentence}`,
                "orange-text"
            );
            return;
        }
        getSubstitutes(
            [this.state.customSentence],
            [this.state.targetWordPosTag],
            null,
            null,
            null,
            this.handleSubstitutes,
            this.handleInitSubstTable
        );
        console.log("Successfully submitted:", submittedSentence);
    }

    sentenceInputHandler(sentence) {
        this.setState(
            {
                customSentence: sentence
            },
            this.submitSentence
        );
    }

    posTagHandler(posTag) {
        this.setState(
            {
                targetWordPosTag: posTag
            },
            () => {
                console.log("You choose POS-tag:", this.state.targetWordPosTag)
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
                    Custom Sentence mode.
                </h1>
                <div
                    className={"container"}
                    style={{width: "40%"}}
                >
                    <p key={"1"} className={"text-left"}>
                        1. You can type any sentence bellow and get substitutes from models.
                    </p>
                    <p key={"2"} className={"text-left"}>
                        2. Mark target word by "<b>@</b>" (e.g. : "Hello <b>@</b>world<b>@</b> !").
                    </p>
                    <p key={"3"} className={"text-left"}>
                        3. Choose POS-tag of target word for better WordNet Relation functionality.
                    </p>
                </div>
                <Input
                    onInput={this.sentenceInputHandler}
                    onPosTag={this.posTagHandler}
                />
                <SubstitutesTable
                    substitutesData={this.state.substitutesResponse}
                    sentence={this.state.customSentence}
                    showTargetInfo={true}
                />
            </div>
        )
    }
}