import React, { Component } from "react";
import {
    MDBContainer, MDBBtn, MDBModal, MDBModalBody,
    MDBModalHeader, MDBModalFooter
} from "mdbreact";


export default class InfoMessage extends Component {
    constructor(props) {
        super(props);
        this.state = {
            isVisible: false
        };
        this.toggle = this.toggle.bind(this);
    }

    toggle() {
        this.setState(
            {
                isVisible: !this.state.isVisible
            },
            () => {
                console.log("Successfully togging InfoMessage!");
            }
        );
    }

    render() {
        return (
            <MDBContainer>
                <MDBBtn
                    id={"show-msg-btn"}
                    onClick={this.toggle}
                    style={{display: "none"}}
                >
                    { this.props.msgTitle }
                </MDBBtn>
                <MDBModal
                    isOpen={this.state.isVisible}
                    toggle={this.toggle}
                    backdrop={false}
                    side
                    position={"top-right"}
                >
                    <MDBModalHeader
                        className={this.props.titleColor}
                        toggle={this.toggle}
                    >
                        { this.props.msgTitle }
                    </MDBModalHeader>
                    <MDBModalBody>
                        { this.props.msgText }
                    </MDBModalBody>
                    <MDBModalFooter>
                        <MDBBtn
                            id={"close-msg-btn"}
                            color={"danger"}
                            onClick={this.toggle}
                        >
                            Close
                        </MDBBtn>
                    </MDBModalFooter>
                </MDBModal>
            </MDBContainer>
        );
    }
}