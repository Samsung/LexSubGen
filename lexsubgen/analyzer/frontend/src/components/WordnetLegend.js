import React, { Component } from "react";
import Spinner from "./Spinner";
import { getWordnetRelations } from "../utils/requests/state";
import { MDBIcon } from "mdbreact";

export default class WordnetLegend extends Component {
    constructor(props) {
        super(props);
        this.state = {
          relations: null
        };
        this.renderLegend = this.renderLegend.bind(this);
        this.handleRelations = this.handleRelations.bind(this);
    }

    handleRelations(relations) {
        this.setState(
            {
                relations: relations
            },
            () => {
                console.log("Received WordNet relations:", relations);
            }
        );
    }

    componentDidMount() {
        getWordnetRelations(this.handleRelations);
    }

    renderLegend() {
        if (!this.state.relations) {
            return (
                <Spinner />
            );
        }
        return this.state.relations.map(
            (relation) => {
                return (
                    <button
                        key={relation}
                        className={"relation legend " + relation}
                    >
                        {relation}
                    </button>
                );
            }
        );
    }

    render() {
        return (
            <div
                className={"wordnet-container"}
            >
                <h2>
                    <MDBIcon
                        icon={"globe"}
                        style={{marginRight: "1%"}}
                    />
                    WordNet Legend
                </h2>
                <hr className="my-2"/>
                <div
                    className={"wordnet-legend"}
                >
                    <div>
                        { this.renderLegend() }
                    </div>
                </div>
                <hr className="my-2"/>
            </div>
        );
    }
}
