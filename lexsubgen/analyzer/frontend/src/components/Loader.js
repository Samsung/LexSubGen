import React, { Component } from "react";


export default class Loader extends Component {
    render() {
        return (
            <div
                className={"spinner-grow text-" + this.props.color}
                role={"status"}
            >
                <span className="sr-only">Loading...</span>
            </div>
        );
    }
}