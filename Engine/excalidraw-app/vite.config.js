import { defineConfig } from "vite";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "node:url";

const projectDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
    plugins: [
        {
            name: "save-excalidraw",

            configureServer(server) {
                server.middlewares.use("/save-diagram", (req, res) => {

                    if (req.method !== "POST") {
                        res.statusCode = 405;
                        return res.end("Method not allowed");
                    }

                    let body = "";

                    req.on("data", chunk => {
                        body += chunk;
                    });

                    req.on("end", () => {
                        try {
                            const data = JSON.parse(body);

                            // Save directly to StudyMate/Engine
                            const outputPath = path.resolve(
                                projectDir,
                                "../Output/diagram.excalidraw"
                            );

                            // Ensure the destination folder exists
                            fs.mkdirSync(path.dirname(outputPath), {
                                recursive: true
                            });

                            // Write Excalidraw JSON file
                            fs.writeFileSync(
                                outputPath,
                                JSON.stringify(data, null, 2),
                                "utf8"
                            );

                            res.setHeader(
                                "Content-Type",
                                "application/json"
                            );

                            res.end(JSON.stringify({
                                success: true,
                                path: outputPath
                            }));

                            console.log("Saved:", outputPath);

                        } catch (error) {
                            console.error("Save failed:", error);

                            res.statusCode = 500;
                            res.end(error.message);
                        }
                    });
                });
            }
        }
    ]
});