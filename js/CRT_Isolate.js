import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "CRT.Isolate",

    async nodeCreated(node) {
        if (node.comfyClass === "CRT_IsolateInput") {
            node.color   = "#0000eb";
            node.bgcolor = "#0400ff";
        }
        if (node.comfyClass === "CRT_IsolateOutput") {
            node.color   = "#eb0000";
            node.bgcolor = "#ff0000";
        }
    },
});
