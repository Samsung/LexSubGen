// const baseURL = "http://" + window.location.host;
const baseURL = "http://kafka:5000";
const requestParams = {
    "headers": {
        "Content-type": "application/json"
    }
};


function getTargetWord(s) {
    let match = s.match(/@.*@/g);
    let result = "";
    if (match) {
        result = match[0];
    }
    return result;
}

function processSentences(sentences) {
    let targetIds = [];
    let targetWords = [];
    let tokensLists = [];
    let processedSentences = sentences.map(
        (sentence) => {
            let targetWord = getTargetWord(sentence);
            targetWords.push(targetWord.slice(1, -1));
            return sentence.replace(targetWord, "<TARGET>");
        }
    );
    processedSentences.map(
        // eslint-disable-next-line array-callback-return
        (sentence, idx) => {
            let tokens = sentence.split(" ");
            let targetIdx = tokens.indexOf("<TARGET>");
            targetIds.push(targetIdx);
            let leftContext = tokens.slice(0, targetIdx);
            let rightContext = tokens.slice(targetIdx + 1);
            let targetTokens = targetWords[idx].split(" ");
            tokensLists.push(leftContext.concat(targetTokens, rightContext));
        }
    );
    return {
        targetIds: targetIds,
        tokensLists: tokensLists,
        targetWords: targetWords
    }
}

export {
    baseURL, requestParams,
    getTargetWord, processSentences
};
