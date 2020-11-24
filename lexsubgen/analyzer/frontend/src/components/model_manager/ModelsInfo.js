import React, { Component } from "react";
import {MDBTable, MDBTableBody, MDBTableHead, MDBBtn, MDBIcon} from "mdbreact";

import { changeModelStatus, removeModel, pingModel } from "../../utils/requests/state";
import Spinner from "../Spinner";

export default class ModelsInfo extends Component {
    constructor(props) {
        super(props);
        this.state = {
            hoverIdx: -1,
            modelsInfo: Array(this.props.models.length).fill("#939393"),
        };
        this.renderModelInfo = this.renderModelInfo.bind(this);
        this.renderControlBtns = this.renderControlBtns.bind(this);

        this.getModelStatus = this.getModelStatus.bind(this);
        this.handleModelChangeStatus = this.handleModelChangeStatus.bind(this);
        this.handleModelRemove = this.handleModelRemove.bind(this);
        this.handleModelPing = this.handleModelPing.bind(this);
    }

    handleModelChangeStatus(modelIdx) {
        changeModelStatus(modelIdx, this.props.onModelInfoChange);
    }

    handleModelRemove(modelIdx) {
        removeModel(modelIdx, this.props.onModelInfoChange);
    }

    handleModelPing(modelIdx) {
        let oldPings = this.state.modelsInfo;
        let isOk = pingModel(modelIdx);
        if (isOk) {
            oldPings[modelIdx] = "#4ac52f";
        } else {
            oldPings[modelIdx] = "red";
        }
        this.setState({modelsInfo: oldPings});
        this.props.onModelInfoChange();
        setTimeout(
            () => {
                let oldPings = this.state.modelsInfo;
                oldPings[modelIdx] = "#939393";
                this.setState({modelsInfo: oldPings});
                this.props.onModelInfoChange();
            },
            1000
        );
    }

    getModelStatus(status) {
        if (status) {
            return (
                <MDBBtn
                    size={"sm"}
                    color={"success"}
                >
                    Activated
                </MDBBtn>
            );
        } else {
            return (
                <MDBBtn
                    size={"sm"}
                    color={"danger"}
                >
                    Deactivated
                </MDBBtn>
            );
        }
    }

    renderControlBtns(idx) {
        return ([
            <MDBIcon
                key={"ping"}
                icon={"signal"}
                className={"ping-icon"}
                style={{
                    color: this.state.modelsInfo[idx]
                }}
                onClick={() => this.handleModelPing(idx)}
            />,
            <MDBIcon
                key={"remove"}
                icon={"trash"}
                className={"model-delete-icon"}
                onClick={() => this.handleModelRemove(idx)}
            />,
        ]);
    }

    renderModelInfo(models) {
        return models.map(
            (model, idx) => {
                return (
                    <tr key={idx}>
                        <th className={"text-center"}>{idx + 1}</th>
                        {/*<th>{model.config_path}</th>*/}
                        <th>{model.name}</th>
                        <th>{model.host}</th>
                        <th>{model.port}</th>
                        <th
                            className={"model-status"}
                            onClick={() => this.handleModelChangeStatus(idx)}
                        >
                            { this.getModelStatus(model.is_active) }
                        </th>
                        <th>
                            { this.renderControlBtns(idx) }
                        </th>
                    </tr>
                );
            }
        );
    }

    render() {
        if (!this.props.models.length) {
            return (
                <Spinner />
            );
        }
        return (
            <MDBTable>
                <MDBTableHead>
                    <tr>
                        <th className={"text-center"}>#</th>
                        {/*<th>Config</th>*/}
                        <th>Model Name</th>
                        <th>Host</th>
                        <th>Port</th>
                        <th>Status</th>
                        <th>Control</th>
                    </tr>
                </MDBTableHead>
                <MDBTableBody>
                    { this.renderModelInfo(this.props.models) }
                </MDBTableBody>
            </MDBTable>
        );
    }
}