import React, { Component } from "react";
import { MDBNavbarBrand, MDBBtn, MDBIcon } from "mdbreact";

export default class ModeChanger extends Component {
    constructor(props) {
        super(props);
        this.state = {};
    }

    render() {
        return (
            <MDBNavbarBrand>
                <MDBBtn
                    id={this.props.id}
                    color={this.props.btnColor}
                    className={this.props.active ? "active-mode" : ""}
                    onClick={
                        () => {
                            this.props.onModeChange(this.props.modeName);
                        }
                    }
                >
                    <MDBIcon
                        icon={this.props.iconName}
                        style={{
                            paddingRight: "5px",
                            fontSize: "1rem"
                        }}
                    />
                    { this.props.modeText }
                </MDBBtn>
            </MDBNavbarBrand>
        );
    }
}