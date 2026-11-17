import {
    parseMermaidToExcalidraw
} from "@excalidraw/mermaid-to-excalidraw";

import {
    convertToExcalidrawElements
} from "@excalidraw/excalidraw";


async function main() {

    try {

        // 1. Read Mermaid file
        console.log("Reading diagram.mmd...");

        const response = await fetch("/diagram.mmd");

        if (!response.ok) {
            throw new Error("Could not load diagram.mmd");
        }

        const definition = await response.text();


        // 2. Parse Mermaid
        console.log("Parsing Mermaid...");

        const {
            elements,
            files
        } = await parseMermaidToExcalidraw(definition);


        // 3. Convert to Excalidraw elements
        console.log("Converting elements...");

        const converted =
            convertToExcalidrawElements(elements);

        console.log("Excalidraw Elements:", converted);
        console.log("Element count:", converted.length);


        // 4. Create Excalidraw document
        const excalidrawData = {

            type: "excalidraw",

            version: 2,

            source: "StudyMate",

            elements: converted,

            appState: {
                viewBackgroundColor: "#ffffff",
                gridSize: null
            },

            files: files || {}

        };


        // 5. Save directly into StudyMate/Engine
        console.log("Saving diagram...");

        const saveResponse = await fetch("/save-diagram", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(excalidrawData)

        });


        if (!saveResponse.ok) {
            throw new Error(await saveResponse.text());
        }

        const result = await saveResponse.json();

        console.log("SUCCESS!");
        console.log("Saved file:", result.path);


    } catch (error) {

        console.error("Conversion failed:", error);

    }

}


// 6. Execute
main();
