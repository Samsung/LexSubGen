import React, { Component } from "react";

import { switchRequest } from "../../utils/requests/state";

export default class Switcher extends Component {
    constructor(props) {
        super(props);
        this.handleSwitch = this.handleSwitch.bind(this);
    }

    handleSwitch() {
        switchRequest(this.props.onSwitch);
    }

    render() {
        return (
            <div
                className={"custom-control custom-switch"}
            >
                <input
                    type={"checkbox"}
                    checked={this.props.checked}
                    onChange={this.handleSwitch}
                    className={"custom-control-input"}
                    id={"customSwitches"}
                />
                <label
                    className={"custom-control-label"}
                    htmlFor={"customSwitches"}
                >
                    {this.props.checked ? "Parallel " : "Sequential "}
                    requests mode
                </label>
            </div>
        );
    }
}
