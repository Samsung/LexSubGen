import React, { Component } from "react";

export default class ScrollToTopArrow extends Component {
    constructor(props) {
        super(props);
        this.scrollToTop = this.scrollToTop.bind(this);
    }

    componentDidMount() {
        let maxScrollValue = 0.9 * window.innerHeight;
        let btnToTop = document.getElementById("scroll-to-top-btn");
        window.addEventListener("scroll", () => {
            if (document.scrollingElement.scrollTop > maxScrollValue) {
                btnToTop.style.display = "block";
            } else {
                btnToTop.style.display = "none";
            }
        });
    }

    scrollToTop() {
        document.scrollingElement.scrollTop = 0;
    }

    render() {
        return (
            <div
                id={"scroll-to-top-btn"}
                onClick={this.scrollToTop}
            >
                <button
                    className={"btn btn-lg btn-primary"}
                >
                    <i className="fa fa-arrow-up" />
                </button>
            </div>
        );
    }
}
