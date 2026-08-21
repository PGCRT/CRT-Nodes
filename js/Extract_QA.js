import { app } from "../../scripts/app.js";

const EXT_NAME = "CRT.ExtractQA.DynamicTarget";

function findTargetWidget(node) {
    return node.widgets?.find((w) => w.name === "target" || w.name === "target_is_question" || w.name.startsWith("target:"));
}

function updateWidgetLabel(widget, node) {
    const isQuestion = !!widget.value;
    // Keep the input key stable ("target") for prompt serialization,
    // use label for the dynamic display text.
    widget.name = "target";
    widget.label = isQuestion ? "target: user (question) → answer it" : "target: assistant (answer) → create question";
    node?.setDirtyCanvas(true, true);
}

const ExtractQAExtension = {
    name: EXT_NAME,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ExtractQA") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);

            const w = findTargetWidget(this);
            if (!w) return;

            // Legacy workflows may have stored the old combo value as "user"/"assistant"
            if (typeof w.value === "string") {
                w.value = w.value === "user";
            }
            // Ensure canonical name for serialization
            w.name = "target";

            const origCb = w.callback;
            w.callback = (value) => {
                if (origCb) origCb.call(this, value);
                updateWidgetLabel(w, this);
            };

            updateWidgetLabel(w, this);
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            origConfigure?.apply(this, arguments);
            const w = findTargetWidget(this);
            if (w) {
                if (typeof w.value === "string") w.value = w.value === "user";
                w.name = "target";
                updateWidgetLabel(w, this);
            }
        };
    },
};

app.registerExtension(ExtractQAExtension);
