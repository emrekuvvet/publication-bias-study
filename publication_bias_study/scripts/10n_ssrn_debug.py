"""
Debug: camoufox non-headless (passes Cloudflare Turnstile).
coreBundle.js is patched with optional chaining so Firefox JS errors don't crash the driver.
"""
import asyncio, re, time, unicodedata
from camoufox.async_api import AsyncCamoufox

TITLES = [
    "An Equilibrium Model with Buy and Hold Investors",
    "Change is Good or the Disposition Effect Among Mutual Fund Managers",
    "Does Investor Recognition Predict Excess Returns?",
]

CF_WORDS = {"just a moment", "un instant", "un momento", "einen moment", "ett ögonblick", "attendez", "loading"}

async def is_cf_page(page):
    """Return True if we're still on a Cloudflare challenge page."""
    try:
        title = await page.title()
        title_lower = title.lower()
        if any(w in title_lower for w in CF_WORDS):
            return True
        # Also check if Turnstile widget is the only content
        has_links = await page.evaluate(
            "() => !!document.querySelector('a[href*=\"abstract_id\"], input[name=\"txtKey\"], "
            ".title a')"
        )
        return not has_links
    except Exception:
        return True

async def wait_for_ssrn(page, timeout=90):
    """Wait until actual SSRN content appears (not a CF challenge page)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not await is_cf_page(page):
            return True
        title = await page.title()
        print(f"  waiting SSRN ({int(deadline-time.time())}s left) title={title[:50]}")
        await page.wait_for_timeout(2500)
    return False

async def main():
    async with AsyncCamoufox(headless=False, geoip=True) as browser:
        page = await browser.new_page()
        page.on("pageerror", lambda e: None)  # suppress JS errors

        # Load SSRN search page once and wait for form
        search_base = "https://papers.ssrn.com/sol3/results.cfm"
        print(f"Navigating to SSRN search...")
        try:
            await page.goto(search_base, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            print(f"goto: {e}")

        try:
            await page.wait_for_selector("input#term", timeout=90000)
            print(f"SSRN loaded! title: {await page.title()}")
        except Exception as e:
            print(f"Search form not found: {e}")

        # Dismiss cookie consent
        try:
            await page.locator("#onetrust-accept-btn-handler").click(timeout=5000)
            await page.wait_for_timeout(800)
            print("Cookie dismissed")
        except Exception:
            await page.evaluate("document.getElementById('onetrust-consent-sdk')?.remove()")

        for title in TITLES:
            print(f"\n{'='*60}")
            print(f"Searching: {title}")

            try:
                search_inp = page.locator("input#term").first
                await search_inp.click(timeout=5000)
                await search_inp.fill("")
                await search_inp.type(title[:120], delay=20)
                await search_inp.press("Enter")
                await page.wait_for_load_state("domcontentloaded", timeout=20000)
                await page.wait_for_timeout(2000)
                print(f"  Result page title: {await page.title()}")
            except Exception as e:
                print(f"  Search error: {e}")
                continue

            links = await page.evaluate("""
                () => [...document.querySelectorAll('a[href*="abstract_id"], a[href*="/sol3/papers.cfm"]')]
                    .slice(0,5).map(a => ({text: a.innerText.trim().slice(0,80), href: a.href}))
            """)
            print(f"  Links: {links}")

            headings = await page.evaluate("""
                () => [...document.querySelectorAll('h2,h3,h4,.paper-title')].slice(0,5)
                    .map(h => h.innerText.trim().slice(0,80))
            """)
            print(f"  Headings: {headings}")

            # Go back to search form for next query
            try:
                await page.go_back(wait_until="domcontentloaded", timeout=15000)
                await page.wait_for_timeout(1500)
            except Exception:
                await page.goto(search_base, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_selector("input#term", timeout=30000)

        await page.wait_for_timeout(5000)

asyncio.run(main())

