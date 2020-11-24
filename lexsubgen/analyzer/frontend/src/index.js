import React from "react";
import ReactDOM from "react-dom";
import App from "./components/App";
import Footer from "./components/Footer";

import "@fortawesome/fontawesome-free/css/all.min.css";
import "bootstrap-css-only/css/bootstrap.min.css";
import "mdbreact/dist/css/mdb.css";

import "./styles/index.css";
import "./styles/modelManager.css";
import "./styles/customSentenceMode.css";
import "./styles/substituteTable.css";
import "./styles/evaluationMode.css";
import "./styles/wordnet.css";

ReactDOM.render(<App />, document.getElementById('root'));
ReactDOM.render(<Footer />, document.getElementById('footer'));
