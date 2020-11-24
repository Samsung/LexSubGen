import React, { Component } from "react";

import SubstitutesTable from "../custom_sentence/SubstitutesTable";
import WordnetLegend from "../WordnetLegend";

export default class SubstitutesResponse extends Component {
    constructor(props) {
        super(props);
        this.state = {
            showTargetInfo: false,
            showGoldInfo: true,
            showWordnetLegend: false,
            showCandidateInfo: false,
        };
        this.renderSubstitutesTables = this.renderSubstitutesTables.bind(this);
        this.renderCandidateInfoBtn = this.renderCandidateInfoBtn.bind(this);
        this.renderWordnetBtn = this.renderWordnetBtn.bind(this);
        this.renderRankInfoBtn = this.renderRankInfoBtn.bind(this);
        this.renderGoldInfoBtn = this.renderGoldInfoBtn.bind(this);
        this.renderControlButtons = this.renderControlButtons.bind(this);
    }

    renderSubstitutesTables(data) {
        return data.map(
            (sample, idx) => {
                return (
                    <SubstitutesTable
                        key={idx}
                        substitutesData={sample}
                        showMetrics={true}
                        sentence={this.props.sentences[idx]}
                        showTargetInfo={this.state.showTargetInfo}
                        showGoldInfo={this.state.showGoldInfo}
                        showCandidateInfo={this.state.showCandidateInfo}
                    />
                );
            }
        );
    }

    renderWordnetBtn() {
        let className = "show-wordnet-btn custom-tooltip";
        if (this.state.showWordnetLegend) {
            className += " visible";
        }
        return (
            <div
                className={className}
                onClick={
                    () => this.setState({showWordnetLegend : !this.state.showWordnetLegend})
                }
            >
                <i className={"fa fa-globe"}/>
                <span
                    className={"tooltip-text left-tooltip"}
                >
                    WordNet Legend
                </span>
            </div>
        );
    }

    renderRankInfoBtn() {
        let className = "show-target-info-btn custom-tooltip";
        if (this.state.showTargetInfo) {
            className += " visible";
        }
        return (
            <div
                className={className}
                onClick={
                    () => this.setState({showTargetInfo : !this.state.showTargetInfo})
                }
            >
                <i className={"fa fa-crosshairs"}/>
                <span
                    className={"tooltip-text left-tooltip"}
                >
                    Target Rank
                </span>
            </div>
        );
    }

    renderGoldInfoBtn() {
        let className = "show-gold-info-btn custom-tooltip";
        if (this.state.showGoldInfo) {
            className += " visible";
        }
        return (
            <div
                className={className}
                onClick={
                    () => this.setState({
                        showGoldInfo : !this.state.showGoldInfo,
                        showCandidateInfo : !this.state.showCandidateInfo,
                    })
                }
            >
                <i className={"fa fa-scroll"}/>
                <span
                    className={"tooltip-text left-tooltip"}
                >
                    Gold Ranks
                </span>
            </div>
        );
    }

    renderCandidateInfoBtn() {
        let className = "show-candidate-info-btn custom-tooltip";
        if (this.state.showCandidateInfo) {
            className += " visible";
        }
        return (
            <div
                className={className}
                onClick={
                    () => this.setState({
                        showCandidateInfo : !this.state.showCandidateInfo,
                        showGoldInfo: !this.state.showGoldInfo
                    })
                }
            >
                <i className={"fa fa-users"}/>
                <span
                    className={"tooltip-text left-tooltip"}
                >
                    Candidate Ranks
                </span>
            </div>
        );
    }

    renderControlButtons() {
        return (
            <div className={"btn-group-vertical control-buttons"}>
                { this.renderCandidateInfoBtn() }
                { this.renderGoldInfoBtn() }
                { this.renderRankInfoBtn() }
                { this.renderWordnetBtn() }
            </div>
        );
    }

    render() {
        if (!this.props.responseData) {
            return null;
        }
        return (
            <div style={{marginTop: "1%"}}>
                <h2 className="display-5">
                    You choose <b>{this.props.datasetName}</b> dataset, which contains {this.props.responseData.length} samples.
                </h2>
                { this.renderControlButtons() }
                {
                    !this.state.showWordnetLegend ?
                        null :
                        <WordnetLegend />
                }
                { this.renderSubstitutesTables(this.props.responseData) }
            </div>
        );
    }
}