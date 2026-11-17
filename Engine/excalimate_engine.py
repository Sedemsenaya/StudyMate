import asyncio
import json
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


MCP_URL = "http://127.0.0.1:4000/mcp"

EXCALIDRAW_FILE = Path(
    r"C:\Users\Colli\PycharmProjects\StudyMate\Engine\Output\diagram.excalidraw"
)


async def main():

    # 1. Load Excalidraw file
    with open(EXCALIDRAW_FILE, "r", encoding="utf-8") as f:
        diagram = json.load(f)

    elements = diagram["elements"]

    ids = [
        el["id"]
        for el in elements
        if not el.get("isDeleted")
    ]

    print(f"Loaded {len(elements)} elements")

    # 2. Connect to Excalimate
    async with streamable_http_client(MCP_URL) as (
        read_stream,
        write_stream
    ):

        async with ClientSession(
            read_stream,
            write_stream
        ) as session:

            await session.initialize()

            tools = await session.list_tools()

            print("\nAVAILABLE EXCALIMATE TOOLS")

            for tool in tools.tools:
                print(f"\n{tool.name}")
                print(tool.description)
                print(json.dumps(tool.input_schema, indent=2))

            # 3. Create scene
            result = await session.call_tool(
                "create_scene",
                {"elements": elements}
            )

            print("\nScene created:")
            print(result)

            # 4. Automatically animate
            result = await session.call_tool(
                "auto_animate",
                {
                    "scope": {"elementIds": ids},
                    "style": {"intensity": "balanced"}
                }
            )

            print("\nAnimation generated:")
            print(result)

            # 5. Set clip duration
            result = await session.call_tool(
                "set_clip_range",
                {
                    "start": 0,
                    "end": 15000
                }
            )

            print("\nClip range configured:")
            print(result)

            # 6. Save checkpoint
            result = await session.call_tool(
                "save_checkpoint",
                {"id": "diagram"}
            )

            print("\nCheckpoint saved:")
            print(result)




if __name__ == "__main__":
    asyncio.run(main())
