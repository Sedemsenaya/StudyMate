"""
StudyMate: diagram.excalidraw -> Excalimate Auto Animate -> diagram.mp4

No MCP server needed: drives app.excalimate.com directly with Playwright.
Button names below were read from the live app (Nov 2026).

Setup (once):
    pip install playwright
    playwright install chromium

Run:
    python excalimate_to_mp4.py
    python excalimate_to_mp4.py path\\to\\other.excalidraw
"""
import asyncio
import sys
import re
from pathlib import Path

from playwright.async_api import async_playwright

APP_URL = "https://app.excalimate.com"

# DEFAULT_FILE = Path(
#     r"C:\Users\Colli\PycharmProjects\StudyMate\Engine\Output\diagram.excalidraw"
# )
# DEFAULT_FILE = Path(
#     r"C:\Users\Colli\PycharmProjects\StudyMate\Engine\Output\scene_01.excalidraw"
# )
DEFAULT_FILE = Path(
    r"C:\Users\Colli\PycharmProjects\StudyMate\Engine\visualization\scenes\scene_03.excalidraw"
)

HEADLESS = False   # watch the first run; set True once it works
QUALITY = "High"   # "Low" | "Medium" | "High" | "Very High"


async def excalidraw_to_mp4(src: Path, out: Path) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            accept_downloads=True,
            viewport={"width": 1600, "height": 900},
        )
        page = await context.new_page()

        # 1. Open the editor and dismiss the privacy banner
        await page.goto(APP_URL)
        consent = page.get_by_role("button", name="Use required only")
        if await consent.is_visible(timeout=5000):
            await consent.click()

        # 2. Import the .excalidraw file through the "Choose File" drop zone
        file_inputs = page.locator('input[type="file"]')
        if await file_inputs.count() > 0:
            await file_inputs.first.set_input_files(str(src))
        else:
            async with page.expect_file_chooser() as fc:
                await page.get_by_role("button", name="Choose File").click()
            await (await fc.value).set_files(str(src))
        print(f"[ok] imported {src.name}")

        # 3. Auto Animate -> review dialog -> Apply order
        await page.get_by_role("button", name="Auto Animate").click()
        await page.get_by_role("button", name="Apply order").click()
        await page.get_by_text("Animation order applied").wait_for(timeout=15000)
        print("[ok] auto-animated")

        # 4. Export -> Video tab -> MP4 -> quality -> Export MP4
        await page.get_by_role("button", name="Export", exact=True).click()
        dialog = page.get_by_role("dialog", name="Export")
        await dialog.get_by_role("button", name=re.compile(r"^MP4")).click()
        await dialog.get_by_role("button", name=re.compile(r"^MP4")).click()
        await dialog.get_by_role("button", name=QUALITY, exact=True).click()
        print("[..] rendering MP4 (this can take a while)")

        async with page.expect_download(timeout=10 * 60 * 1000) as dl:
            await dialog.get_by_role("button", name="Export MP4").click()
        await (await dl.value).save_as(out)
        print(f"[ok] wrote {out}")

        await browser.close()


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    asyncio.run(excalidraw_to_mp4(src, src.with_suffix(".mp4")))