import React, { Component } from "react";
import {
    MDBInput, MDBContainer, MDBBtn, MDBBtnGroup,
    MDBDropdown, MDBDropdownToggle, MDBDropdownMenu, MDBDropdownItem
} from "mdbreact";


export default class Input extends Component {
    constructor(props) {
        super(props);
        this.state = {
            sentence: ""
        };
        this.handleButtonPress = this.handleButtonPress.bind(this);
        this.handleInput = this.handleInput.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
        this.renderPosTagPicker = this.renderPosTagPicker.bind(this);
    }

    renderPosTagPicker() {
        let posTagInfo = {
            "n": "N - Noun",
            "v": "V - Verb",
            "a": "A - Adjective",
            "r": "R - Adverb"
        };

        let dropdownItems = Object.keys(posTagInfo).map(
            (posTag) => {
                return (
                    <MDBDropdownItem
                        key={posTag}
                        onClick={() => this.props.onPosTag(posTag)}
                    >
                        { posTagInfo[posTag] }
                    </MDBDropdownItem>
                );
            }
        );
        return (
            <MDBDropdown dropright={true}>
                <MDBDropdownToggle caret color="primary">
                    Choose POS-tag
                </MDBDropdownToggle>
                <MDBDropdownMenu basic>
                    { dropdownItems }
                </MDBDropdownMenu>
            </MDBDropdown>
        );
    }

    handleInput(event) {
        this.setState(
            {
                sentence: event.target.value
            },
            () => {
                console.log("Input sentence: " + this.state.sentence);
            }
        );
    }

    handleSubmit() {
        let submittedSentence = this.state.sentence;
        if (!submittedSentence) {
            return;
        }
        this.props.onInput(this.state.sentence);
        this.setState(
            {
                sentence: "",
            },
            () => {
                console.log("Submitting:", submittedSentence);
            }
        )
    }

    handleButtonPress(event) {
        let pressedButton = event.key;
        if (pressedButton === "Enter" && this.state.sentence) {
            this.handleSubmit();
        }
    }

    render() {
        return (
            <MDBContainer
                 id={"custom-input-form"}
            >
                <MDBInput
                    icon={"keyboard"}
                    value={this.state.sentence}
                    label="Type sentence here"
                    onKeyDown={this.handleButtonPress}
                    onChange={this.handleInput}
                />
                <MDBBtnGroup>
                    <MDBBtn
                        color={"primary"}
                        onClick={this.handleSubmit}
                        style={{marginBottom: "5%"}}
                    >
                        Submit
                    </MDBBtn>
                    { this.renderPosTagPicker() }
                </MDBBtnGroup>
            </MDBContainer>
        );
    }
}