import React, { Component } from "react";

export default class Footer extends Component {
    render() {
        let currentYear = new Date().getFullYear();
        return (
                <p>
                    &copy; Copyright {currentYear}, Samsung Electronics
                </p>
        )
    }
}
