import React, { Component } from "react";
import { MDBProgress } from "mdbreact";
import { getProgress } from "../../utils/requests/evaluate";


export default class EvaluationProgress extends Component {
    constructor(props) {
        super(props);
        this.state = {
            percent: 0
        };
        this.handleProgress = this.handleProgress.bind(this);
    }

    componentDidMount() {
        this.interval = setInterval(
            () => {
                getProgress(this.props.host, this.props.port, this.handleProgress);
                console.log("Current progress:", this.state.percent)
            },
            2000
        );
    }

    componentWillUnmount() {
        clearInterval(this.interval);
    }

    handleProgress(response) {
        this.setState(
            {
                percent: response.progress
            },
            () => {
                console.log("Progress:", this.state.percent);
            }
        );
    }

    render() {
        return (
            <>
                <span>
                    Progress: {this.state.percent} %
                </span>
                <MDBProgress
                    animated={true}
                    value={Number(this.state.percent)}
                />
            </>
        );
    }
}