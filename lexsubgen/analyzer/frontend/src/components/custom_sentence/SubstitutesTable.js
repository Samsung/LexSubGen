import React, { Component } from "react";
import {
    MDBTable, MDBTableBody, MDBTableHead,
    MDBIcon
} from "mdbreact";

import Loader from "../Loader";

import { getTargetWord } from "../../utils/utils";


export default class SubstitutesTable extends Component {
    constructor(props) {
        super(props);
        this.state = {};
        this.renderTableBody = this.renderTableBody.bind(this);
        this.renderTargetInfo = this.renderTargetInfo.bind(this);
        this.renderGoldInfo = this.renderGoldInfo.bind(this);
        this.renderMetric = this.renderMetric.bind(this);

        this.renderSubstituteScore = this.renderSubstituteScore.bind(this);
        this.renderSubstitutes = this.renderSubstitutes.bind(this);

        this.renderInputSentence = this.renderInputSentence.bind(this);
    }

    renderMetric(response) {
        let metric = null;
        if (this.props.showGoldInfo) {
            metric = response.recall_at_ten;
        } else if (this.props.showCandidateInfo) {
            metric = response.gap;
        }
        return metric;
    }

    renderTargetInfo(info) {
        if (!info) {
            return "No Target Word Info!";
        }
        return (
            <span>
                Rank: <b>{
                    info.rank === -1 ? "OOV" : info.rank
                }</b>
            </span>
        );
    }

    renderGoldInfo(info) {
        if (!info) {
            return null;
        }
        return info.map(
            (gold, idx) => {
                return (
                    <div
                        key={`gold-${idx}`}
                        className={"substitute relation"}
                    >
                        { gold.word }
                        { this.renderSubstituteScore(gold.rank, true)}
                    </div>
                );
            }
        );
    }

    renderSubstituteScore(score, isRank=false) {
        if (score === null) {
            return null;
        }
        if (isRank) {
            return (
                <b
                    className={"substitute-rank"}
                >
                    { score > 0 ? score: "OOV" }
                </b>
            );
        }
        return (
            <sup
                className={"substitute-score"}
            >
                <b>
                { score > 0 ? score: "<1" }
                </b>
            </sup>
        );
    }

    renderSubstitutes(response, isGold=false) {
        let substitutes;
        if (this.props.showCandidateInfo && !isGold) {
            substitutes = response.ranked_candidates;
        } else {
            substitutes = response.generated_substitutes;
        }
        if (!substitutes) {
            return null;
        }
        return substitutes.map(
            (substitute, idx) => {
                let substituteClass = "relation custom-tooltip " + substitute.wordnet_relation;
                let substituteContainerClass = "substitute";
                if (substitute.tp || isGold) {
                    substituteContainerClass += " true-positive";
                }
                let tooltipClass = "tooltip-text top-tooltip " + substitute.wordnet_relation;
                return (
                    <div
                        key={idx}
                        className={substituteContainerClass}
                    >
                        <div
                            className={substituteClass}
                        >
                            { substitute.word }
                            <span
                                className={tooltipClass}
                            >
                                { substitute.wordnet_relation }
                            </span>
                        </div>
                        {
                            this.renderSubstituteScore(
                                this.props.showCandidateInfo && !isGold ? substitute.rank : substitute.score,
                                isGold || this.props.showCandidateInfo
                            )
                        }
                    </div>
                );
            }
        );
    }

    renderTableBody(data) {
        return data.map(
            (model, idx) => {
                return (
                    <tr key={idx}>
                        <th
                            className={
                                model.modelName ?
                                    "model-name-column" :
                                    "text-center"
                            }
                        >
                            {
                                model.modelName ?
                                    model.modelName :
                                    <Loader color={"success"}/>
                            }
                        </th>
                        {
                            this.props.showMetrics ?
                                <th
                                    className={"text-center"}
                                >
                                    {
                                        model.modelName ?
                                            this.renderMetric(model.response) :
                                            <Loader color={"info"}/>
                                    }
                                </th> : null
                        }
                        <th
                            className={
                                model.modelName ?
                                    "substitute-column" :
                                    "text-center"
                            }
                        >
                            {
                                model.modelName ?
                                    this.renderSubstitutes(
                                        model.response,
                                        model.modelName === "Gold Standard"
                                    ) :
                                    <Loader color={"primary"}/>
                            }
                        </th>
                        {
                            !this.props.showGoldInfo ?
                                null :
                                <th
                                    className={
                                        model.modelName ?
                                            "gold-standard-column" :
                                            "text-center"
                                    }
                                >
                                {
                                    model.modelName ?
                                        this.renderGoldInfo(model.response.gold_substitutes) :
                                        <Loader color={"warning"}/>
                                }
                                </th>
                        }
                        {
                            !this.props.showTargetInfo ?
                                null :
                                <th className={"text-center"}>
                                {
                                    model.modelName ?
                                        this.renderTargetInfo(model.response.target_word) :
                                        <Loader color={"danger"}/>
                                }
                                </th>

                        }
                    </tr>
                );
            }
        );
    }

    renderInputSentence(sentence) {
        let targetWord = getTargetWord(sentence);
        if (!targetWord) {
            return sentence;
        }
        let parts = sentence.split(targetWord);
        targetWord = targetWord.slice(1, -1);
        let leftContext = parts[0];
        let rightContext = parts[1];
        return ([
            <span key={"left-context"}>
                { leftContext }
            </span>,
            <span key={"target"} className={"target-word"}>
                { targetWord }
            </span>,
            <span key={"right-context"}>
                { rightContext }
            </span>,
        ]);
    }

    render() {
        if (!this.props.substitutesData.length) {
            return null;
        }
        return (
            <div className={"custom-sentence-response"}>
                <p className={"my-2 input-sentence"}>
                    { this.renderInputSentence(this.props.sentence) }
                </p>
                <MDBTable>
                    <MDBTableHead>
                        <tr>
                            <th className={"custom-sentence-header"}>
                                <MDBIcon
                                    className={"model-icon"}
                                    icon={"server"}
                                />
                                Model Name
                            </th>
                            {
                                !this.props.showMetrics ? null :
                                <th
                                    className={"custom-sentence-header text-center"}
                                >
                                    <MDBIcon
                                        className={"metric-icon"}
                                        icon={"calculator"}
                                    />
                                    {
                                        this.props.showCandidateInfo ? "GAP" : "R@10"
                                    }
                                </th>
                            }
                            {
                                this.props.showCandidateInfo ?
                                    <th className={"custom-sentence-header"}>
                                        <MDBIcon
                                            className={"candidate-icon"}
                                            icon={"users"}
                                        />
                                        Candidate Ranks
                                    </th> :
                                    <th className={"custom-sentence-header"}>
                                        <MDBIcon
                                            className={"substitute-icon"}
                                            icon={"comment-dots"}
                                        />
                                        Generated Substitutes
                                    </th>
                            }
                            {
                                this.props.showGoldInfo ?
                                    <th className={"custom-sentence-header"}>
                                        <MDBIcon
                                            className={"gold-standard-icon"}
                                            icon={"scroll"}
                                        />
                                        Gold Word Ranks
                                    </th> :
                                    null
                            }
                            {
                                this.props.showTargetInfo ?
                                    <th className={"custom-sentence-header"}>
                                        <MDBIcon
                                            className={"target-word-icon"}
                                            icon={"crosshairs"}
                                        />
                                        Target Word Rank
                                    </th> :
                                    null
                            }
                        </tr>
                    </MDBTableHead>
                    <MDBTableBody
                        class={"substitute-table-body"}
                    >
                        { this.renderTableBody(this.props.substitutesData) }
                    </MDBTableBody>
                </MDBTable>
            </div>
        );
    }
}