"""
pinterest_scraper_lazyload.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scrape 200 image‑src URLs from three Pinterest searches that lazily render
only the pins inside the viewport.

Strategy
--------
1.  Scroll the page step‑by‑step (1.5 × viewport height each time).
2.  After every scroll, collect *all* <img> elements currently mounted
    under #mweb-unauth-container.
3.  Stop when 200 unique srcs are found, or when a configurable number of
    consecutive scrolls yields no new images (to avoid an infinite loop).
4.  Save the results to CSV files named mens_outfits.csv, women_outfits.csv.

Dependencies
------------
playwright  |  pip install playwright
bs4         |  pip install beautifulsoup4

Run once to install the browser binaries:
    playwright install chromium
"""

from pathlib import Path
import csv
from playwright.sync_api import sync_playwright
from ..config import DATA_DIR

# --------------------------------------------------------------------------
# SEARCH PAGES
# --------------------------------------------------------------------------
SEARCH_CONFIGS = [
    {
        "name": "mens_outfits",
        "url": "https://www.pinterest.com/search/pins/?q=mens%20outfits&rs=ac"
               "&len=3&source_id=ac_QrLmKXEK&eq=men&etslf=6256",
        "csv_file": str(DATA_DIR / "mens_outfits.csv"),
    },
    {
        "name": "women_outfits",
        "url": "https://www.pinterest.com/search/pins/?q=women%20outfits&rs=ac"
               "&len=5&source_id=ac_gxMviWR9&eq=women&etslf=9847",
        "csv_file": str(DATA_DIR / "women_outfits.csv"),
    },
]

# --------------------------------------------------------------------------
# SCRAPER
# --------------------------------------------------------------------------
TARGET_PER_PAGE = 200
SCROLL_Y_FACTOR = 1.5       # 1.5 × viewport height each step
SCROLL_PAUSE_MS = 1200      # wait time after each scroll
MAX_IDLE_SCROLLS = 20       # stop if this many scrolls add no new images


def write_csv(path: Path, rows):
    """Write one-column CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["src"], *[[r] for r in rows]])


def collect_visible_imgs(page):
    """Return list of srcs currently mounted in the DOM."""
    return page.eval_on_selector_all(
        "#mweb-unauth-container img[src], #mweb-unauth-container img[data-src]",
        "els => els.map(e => e.getAttribute('src') || e.getAttribute('data-src'))"
    )


def harvest_pinterest(page, target=TARGET_PER_PAGE):
    collected = set(collect_visible_imgs(page))
    print(f"  ↳ seeded with {len(collected)} imgs on first view")

    idle_scrolls = 0
    viewport_height = page.viewport_size["height"]

    while len(collected) < target and idle_scrolls < MAX_IDLE_SCROLLS:
        page.evaluate(
            "(y) => window.scrollBy(0, y)",
            viewport_height * SCROLL_Y_FACTOR
        )
        page.wait_for_timeout(SCROLL_PAUSE_MS)

        before = len(collected)
        collected.update(collect_visible_imgs(page))
        gained = len(collected) - before

        if gained == 0:
            idle_scrolls += 1
            print(f"  — no new imgs (idle {idle_scrolls}/{MAX_IDLE_SCROLLS})")
        else:
            idle_scrolls = 0
            print(f"  +{gained} new (total {len(collected)}/{target})")

    return list(collected)[:target]


def main():
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/123.0.0.0 Safari/537.36"),
        )

        for cfg in SEARCH_CONFIGS:
            print(f"▶ Scraping '{cfg['name']}' …")
            page = ctx.new_page()
            page.goto(cfg["url"], wait_until="networkidle")

            img_srcs = harvest_pinterest(page, TARGET_PER_PAGE)

            if len(img_srcs) < TARGET_PER_PAGE:
                print(f"  • reached end of feed with only {len(img_srcs)} images")

            write_csv(cfg["csv_file"], img_srcs)
            print(f"✔ saved {len(img_srcs)} → {cfg['csv_file']}\n")
            page.close()

        ctx.close()
        browser.close()


if __name__ == "__main__":
    main()

""" Had to replace 236x and 60x60 with 736x in all of the image links to get the full resolution images """