const API_URL = "http://127.0.0.1:8000";

const codeInput = document.getElementById("codeInput");
const language = document.getElementById("language");

const reviewBtn = document.getElementById("reviewBtn");
const clearBtn = document.getElementById("clearBtn");
const copyBtn = document.getElementById("copyBtn");

const lineNumbers = document.getElementById("lineNumbers");
const lineCount = document.getElementById("lineCount");
const charCount = document.getElementById("charCount");

const loadingSection = document.getElementById("loadingSection");
const resultsSection = document.getElementById("resultsSection");
const errorSection = document.getElementById("errorSection");

const errorMessage = document.getElementById("errorMessage");


/* -----------------------------------------
   Editor statistics + line numbers
----------------------------------------- */

function updateEditorStats() {

    const code = codeInput.value;

    const lines = code.length === 0
        ? 1
        : code.split("\n").length;

    const characters = code.length;

    /* Update editor statistics */

    lineCount.textContent =
        `${lines} ${lines === 1 ? "line" : "lines"}`;

    charCount.textContent =
        `${characters} ${characters === 1 ? "character" : "characters"}`;


    /* Generate one line number per line */

    const numbers = Array.from(
        { length: lines },
        (_, index) => index + 1
    ).join("\n");

    lineNumbers.textContent = numbers;
}


/* Update line numbers while typing */

codeInput.addEventListener(
    "input",
    updateEditorStats
);


/* Keep line numbers vertically aligned
   when the code editor is scrolled */

codeInput.addEventListener(
    "scroll",
    () => {
        lineNumbers.scrollTop = codeInput.scrollTop;
    }
);


/* -----------------------------------------
   Clear editor
----------------------------------------- */

clearBtn.addEventListener(
    "click",
    () => {

        codeInput.value = "";

        updateEditorStats();

        resultsSection.classList.add("hidden");
        errorSection.classList.add("hidden");

        codeInput.focus();
    }
);


/* -----------------------------------------
   Display error
----------------------------------------- */

function showError(message) {

    errorMessage.textContent = message;

    errorSection.classList.remove("hidden");

    resultsSection.classList.add("hidden");
}


/* -----------------------------------------
   Loading state
----------------------------------------- */

function setLoading(isLoading) {

    if (isLoading) {

        loadingSection.classList.remove("hidden");

        reviewBtn.disabled = true;

        reviewBtn.innerHTML =
            `<span class="btn-icon">◌</span> Analyzing...`;

    } else {

        loadingSection.classList.add("hidden");

        reviewBtn.disabled = false;

        reviewBtn.innerHTML =
            `<span class="btn-icon">⚡</span> Review Code`;
    }
}


/* -----------------------------------------
   Format review item
----------------------------------------- */

function renderItems(
    containerId,
    items,
    emptyMessage
) {

    const container =
        document.getElementById(containerId);

    container.innerHTML = "";

    if (!items || items.length === 0) {

        container.innerHTML =
            `<div class="empty-result">${emptyMessage}</div>`;

        return;
    }


    items.forEach(item => {

        const div = document.createElement("div");

        div.className = "issue-item";


        const text =
            typeof item === "string"
                ? item
                : JSON.stringify(item);


        const lower =
            text.toLowerCase();


        if (lower.includes("severity: high")) {
            div.classList.add("high");
        }

        else if (lower.includes("severity: medium")) {
            div.classList.add("medium");
        }

        else if (lower.includes("severity: low")) {
            div.classList.add("low");
        }


        div.textContent = text;

        container.appendChild(div);
    });
}


/* -----------------------------------------
   Render improvements
----------------------------------------- */

function renderImprovements(items) {

    const container =
        document.getElementById("improvements");

    container.innerHTML = "";

    if (!items || items.length === 0) {

        container.innerHTML =
            `<div class="empty-result">
                No additional improvements suggested.
            </div>`;

        return;
    }


    items.forEach(item => {

        const div =
            document.createElement("div");

        div.className =
            "improvement-item";

        div.textContent =
            typeof item === "string"
                ? item
                : JSON.stringify(item);

        container.appendChild(div);
    });
}


/* -----------------------------------------
   Render complete result
----------------------------------------- */

function renderResults(data) {

    document.getElementById("summary")
        .textContent =
        data.summary || "No summary available.";


    document.getElementById("issueCount")
        .textContent =
        data.issues?.length || 0;

    document.getElementById("securityCount")
        .textContent =
        data.security?.length || 0;

    document.getElementById("performanceCount")
        .textContent =
        data.performance?.length || 0;

    document.getElementById("maintainabilityCount")
        .textContent =
        data.maintainability?.length || 0;


    renderItems(
        "issues",
        data.issues,
        "No issues detected."
    );

    renderItems(
        "security",
        data.security,
        "No security issues detected."
    );

    renderItems(
        "performance",
        data.performance,
        "No performance issues detected."
    );

    renderItems(
        "maintainability",
        data.maintainability,
        "No maintainability issues detected."
    );


    renderImprovements(
        data.improvements
    );


    document.getElementById("improvedCode")
        .textContent =
        data.improved_code ||
        "No improved code was generated.";


    document.getElementById("finalVerdict")
        .textContent =
        data.final_verdict ||
        "No final verdict available.";


    const metadata =
        data.metadata || {};


    document.getElementById("modelName")
        .textContent =
        metadata.model || "—";


    document.getElementById("retrievalTime")
        .textContent =
        metadata.retrieval_time
            ? `${metadata.retrieval_time.toFixed(2)}s`
            : "—";


    document.getElementById("llmLatency")
        .textContent =
        metadata.llm_latency
            ? `${metadata.llm_latency.toFixed(2)}s`
            : "—";


    document.getElementById("totalTime")
        .textContent =
        metadata.total_service_time
            ? `${metadata.total_service_time.toFixed(2)}s`
            : "—";


    resultsSection.classList.remove("hidden");

    errorSection.classList.add("hidden");


    window.scrollTo({
        top: resultsSection.offsetTop - 30,
        behavior: "smooth"
    });
}


/* -----------------------------------------
   Review code
----------------------------------------- */

reviewBtn.addEventListener(
    "click",
    async () => {

        const code =
            codeInput.value.trim();

        const selectedLanguage =
            language.value;


        if (!code) {

            showError(
                "Please paste some code before starting the review."
            );

            codeInput.focus();

            return;
        }


        setLoading(true);

        errorSection.classList.add("hidden");


        try {

            const response =
                await fetch(
                    `${API_URL}/review`,
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json",
                            "Accept":
                                "application/json"
                        },

                        body: JSON.stringify({
                            code: code,
                            language: selectedLanguage
                        })
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                const message =
                    data?.detail
                        ? JSON.stringify(data.detail)
                        : "The API returned an error.";

                throw new Error(message);
            }


            if (!data.success) {

                throw new Error(
                    data.error ||
                    "The AI review could not be completed."
                );
            }


            renderResults(data);

        }

        catch (error) {

            console.error(
                "Review request failed:",
                error
            );

            showError(
                error.message ||
                "Unable to connect to the AI Code Reviewer."
            );

        }

        finally {

            setLoading(false);
        }
    }
);


/* -----------------------------------------
   Copy improved code
----------------------------------------- */

copyBtn.addEventListener(
    "click",
    async () => {

        const code =
            document.getElementById(
                "improvedCode"
            ).textContent;


        if (!code) {
            return;
        }


        try {

            await navigator.clipboard.writeText(code);

            copyBtn.textContent =
                "Copied ✓";


            setTimeout(
                () => {
                    copyBtn.textContent =
                        "Copy Code";
                },
                1800
            );

        }

        catch (error) {

            console.error(
                "Copy failed:",
                error
            );

            copyBtn.textContent =
                "Copy failed";
        }
    }
);


/* -----------------------------------------
   Initial state
----------------------------------------- */

updateEditorStats();