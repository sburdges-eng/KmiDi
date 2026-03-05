#!/usr/bin/env python3
"""
Automated UI Test Script using Playwright
Tests all buttons, functions, and interactions in the iDAW UI

Install dependencies:
    pip install playwright
    playwright install chromium

Run:
    python scripts/test-ui-automated.py
"""

import asyncio
import sys
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

TEST_RESULTS = {
    'passed': [],
    'failed': [],
    'skipped': []
}


def log_test(name, passed, error=None):
    if passed:
        TEST_RESULTS['passed'].append(name)
        print(f"✅ PASS: {name}")
    else:
        TEST_RESULTS['failed'].append({
            'name': name,
            'error': str(error) if error else 'Unknown error',
        })
        print(f"❌ FAIL: {name} - {error}")


def log_skip(name, reason):
    TEST_RESULTS['skipped'].append({'name': name, 'reason': reason})
    print(f"⏭️  SKIP: {name} - {reason}")


async def test_app_header(page):
    """Test header buttons and controls"""
    print("\n=== Testing App Header ===")

    try:
        # Test Side Toggle Button
        toggle_btn = page.locator('button').filter(has_text=page.locator('text=/Side [AB]/'))
        if await toggle_btn.count() > 0:
            await toggle_btn.first.click()
            await page.wait_for_timeout(500)
            log_test("Toggle Side A/B", True)

            # Toggle back
            await toggle_btn.first.click()
            await page.wait_for_timeout(500)
            log_test("Toggle back to Side A", True)
        else:
            log_skip("Toggle Side A/B", "Button not found")

        # Test Error Dismiss (if exists)
        error_dismiss = page.locator('.bg-accent-error button, .lyric-error button').first
        if await error_dismiss.count() > 0:
            await error_dismiss.click()
            await page.wait_for_timeout(300)
            log_test("Dismiss error message", True)
    except Exception as e:
        log_test("App Header Tests", False, e)


async def test_side_a(page):
    """Test Side A (Professional DAW)"""
    print("\n=== Testing Side A (Professional DAW) ===")

    try:
        # Switch to Side A
        toggle_btn = page.locator('button').filter(has_text='Side B')
        if await toggle_btn.count() > 0:
            await toggle_btn.first.click()
            await page.wait_for_timeout(500)

        # Test Generate Music button
        generate_btn = page.locator('button').filter(has_text='Test Generate Music')
        if await generate_btn.count() > 0:
            if not await generate_btn.first.is_disabled():
                await generate_btn.first.click()
                await page.wait_for_timeout(2000)  # Wait for API call
                log_test("Test Generate Music (Side A)", True)
            else:
                log_skip("Test Generate Music (Side A)", "Button is disabled")
        else:
            log_skip("Test Generate Music (Side A)", "Button not found")
    except Exception as e:
        log_test("Side A Tests", False, e)


async def test_side_b(page):
    """Test Side B (Therapeutic Interface)"""
    print("\n=== Testing Side B (Therapeutic Interface) ===")

    try:
        # Switch to Side B
        toggle_btn = page.locator('button').filter(has_text='Side A')
        if await toggle_btn.count() > 0:
            await toggle_btn.first.click()
            await page.wait_for_timeout(500)

        # Test Load Emotions button
        load_emotions = page.locator('button').filter(has_text='Load Emotions')
        if await load_emotions.count() > 0:
            if not await load_emotions.first.is_disabled():
                await load_emotions.first.click()
                await page.wait_for_timeout(2000)  # Wait for API call
                log_test("Load Emotions", True)
            else:
                log_skip("Load Emotions", "Button is disabled")

        # Test Emotion Wheel
        await page.wait_for_timeout(1000)
        base_emotion_btns = page.locator('.emotion-wheel-grid-base button')
        if await base_emotion_btns.count() > 0:
            print("\n--- Testing Emotion Wheel ---")
            await base_emotion_btns.first.click()
            await page.wait_for_timeout(500)
            log_test("Select base emotion", True)

            # Select intensity
            intensity_btns = page.locator('.emotion-wheel-grid-intensity button')
            if await intensity_btns.count() > 0:
                await intensity_btns.first.click()
                await page.wait_for_timeout(500)
                log_test("Select intensity level", True)

                # Select sub-emotion
                sub_btns = page.locator('.emotion-wheel-grid-sub button')
                if await sub_btns.count() > 0:
                    await sub_btns.first.click()
                    await page.wait_for_timeout(500)
                    log_test("Select sub-emotion", True)

                    # Test Clear button
                    clear_btn = page.locator('.emotion-wheel-clear-btn')
                    if await clear_btn.count() > 0:
                        await clear_btn.first.click()
                        await page.wait_for_timeout(300)
                        log_test("Clear emotion selection", True)

        # Test Generate Music button (requires emotion)
        generate_btn = page.locator('button').filter(
            has_text='Generate Music',
        ).filter(has_not=page.locator(':disabled'))
        if await generate_btn.count() > 0:
            await generate_btn.first.click()
            await page.wait_for_timeout(2000)
            log_test("Generate Music (Side B)", True)
        else:
            log_skip("Generate Music (Side B)", "Requires emotion selection")

        # Test Interrogator button
        interrogate_btn = page.locator('button').filter(has_text='Start Interrogation')
        if await interrogate_btn.count() > 0:
            if not await interrogate_btn.first.is_disabled():
                await interrogate_btn.first.click()
                await page.wait_for_timeout(2000)
                log_test("Start Interrogation", True)
            else:
                log_skip("Start Interrogation", "Button is disabled")
    except Exception as e:
        log_test("Side B Tests", False, e)


async def test_lyric_panel(page):
    """Test Lyric Panel functions"""
    print("\n=== Testing Lyric Panel ===")

    try:
        # Test Save Lyrics button
        save_btn = page.locator('button').filter(has_text='Save Lyrics')
        if await save_btn.count() > 0:
            # First, type some lyrics
            textarea = page.locator('.lyric-textarea')
            if await textarea.count() > 0:
                await textarea.first.fill("Test lyrics\nLine 2\nLine 3")
                await page.wait_for_timeout(300)
                await save_btn.first.click()
                await page.wait_for_timeout(1000)
                log_test("Save lyrics", True)

        # Test Refresh button
        refresh_btn = page.locator('button').filter(has_text='Refresh')
        if await refresh_btn.count() > 0:
            await refresh_btn.first.click()
            await page.wait_for_timeout(1000)
            log_test("Refresh lyrics", True)

        # Test Clear button
        clear_btn = page.locator('button').filter(has_text='Clear')
        if await clear_btn.count() > 0:
            await clear_btn.first.click()
            await page.wait_for_timeout(1000)
            log_test("Clear lyrics", True)

        # Test Load from File button (may require Tauri)
        load_file_btn = page.locator('button').filter(has_text='Load .txt/.lrc')
        if await load_file_btn.count() > 0:
            log_test("Load lyrics from file button exists", True)
            # Don't actually trigger file dialog in automated test
    except Exception as e:
        log_test("Lyric Panel Tests", False, e)


async def test_guide_nav(page):
    """Test Guide Navigation"""
    print("\n=== Testing Guide Navigation ===")

    try:
        # Test search input
        search_input = page.locator('.guide-search')
        if await search_input.count() > 0:
            await search_input.first.fill("test")
            await page.wait_for_timeout(500)
            log_test("Search guides", True)

            # Clear search
            await search_input.first.fill("")
            await page.wait_for_timeout(500)

        # Test topic filter buttons
        topic_pills = page.locator('.topic-pill')
        if await topic_pills.count() > 0:
            # Click "All" button
            all_btn = page.locator('.topic-pill').filter(has_text='All')
            if await all_btn.count() > 0:
                await all_btn.first.click()
                await page.wait_for_timeout(300)
                log_test("Filter guides by topic (All)", True)

            # Click first specific topic
            if await topic_pills.count() > 1:
                await topic_pills.nth(1).click()
                await page.wait_for_timeout(300)
                log_test("Filter guides by specific topic", True)

        # Test guide card buttons
        guide_cards = page.locator('.guide-card')
        if await guide_cards.count() > 0:
            first_card = guide_cards.first

            # Test Preview button
            preview_btn = first_card.locator('.preview-btn')
            if await preview_btn.count() > 0:
                await preview_btn.first.click()
                await page.wait_for_timeout(500)
                log_test("Preview guide", True)

            # Test Copy path button
            copy_btn = first_card.locator('.copy-btn')
            if await copy_btn.count() > 0:
                await copy_btn.first.click()
                await page.wait_for_timeout(300)
                log_test("Copy guide path", True)
    except Exception as e:
        log_test("Guide Navigation Tests", False, e)


async def test_spectocloud_panel(page):
    """Test SpectoCloud Panel"""
    print("\n=== Testing SpectoCloud Panel ===")

    try:
        # Test Load Humanizer Config button
        load_config_btn = page.locator('button').filter(has_text='Load Humanizer Config')
        if await load_config_btn.count() > 0:
            await load_config_btn.first.click()
            await page.wait_for_timeout(1000)
            log_test("Load Humanizer Config", True)

        # Test preset buttons
        presets = ['preview', 'standard', 'high']
        for preset in presets:
            preset_btn = page.locator('button').filter(
                has_text=preset,
                has_not=page.locator('.emotion-wheel-btn'))
            if await preset_btn.count() > 0:
                await preset_btn.first.click()
                await page.wait_for_timeout(300)
                log_test(f"Select preset: {preset}", True)

        # Test mode dropdown
        mode_select = page.locator('select').first
        if await mode_select.count() > 0:
            await mode_select.select_option('animation')
            await page.wait_for_timeout(300)
            log_test("Change mode to animation", True)

            await mode_select.select_option('static')
            await page.wait_for_timeout(300)
            log_test("Change mode to static", True)

        # Test render button
        render_btn = page.locator('button').filter(has_text='Render')
        if await render_btn.count() > 0:
            if not await render_btn.first.is_disabled():
                await render_btn.first.click()
                await page.wait_for_timeout(2000)
                log_test("Render SpectoCloud", True)
            else:
                log_skip("Render SpectoCloud", "Button is disabled")
    except Exception as e:
        log_test("SpectoCloud Panel Tests", False, e)


async def run_all_tests():
    """Run all UI tests"""
    print("🚀 Starting Comprehensive UI Tests...\n")
    print("Make sure the dev server is running at http://localhost:1420\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Set to True for headless
        context = await browser.new_context()
        page = await context.new_page()

        try:
            # Navigate to the app
            print("Navigating to http://localhost:1420...")
            await page.goto('http://localhost:1420', wait_until='networkidle', timeout=10000)
            await page.wait_for_timeout(2000)  # Wait for React to render

            # Run all test suites
            await test_app_header(page)
            await test_side_a(page)
            await test_side_b(page)
            await test_lyric_panel(page)
            await test_guide_nav(page)
            await test_spectocloud_panel(page)

            # Print summary
            print("\n" + "=" * 50)
            print("📊 TEST SUMMARY")
            print("=" * 50)
            print(f"✅ Passed: {len(TEST_RESULTS['passed'])}")
            print(f"❌ Failed: {len(TEST_RESULTS['failed'])}")
            print(f"⏭️  Skipped: {len(TEST_RESULTS['skipped'])}")

            if TEST_RESULTS['failed']:
                print("\n❌ Failed Tests:")
                for f in TEST_RESULTS['failed']:
                    print(f"  - {f['name']}: {f['error']}")

            if TEST_RESULTS['skipped']:
                print("\n⏭️  Skipped Tests:")
                for s in TEST_RESULTS['skipped']:
                    print(f"  - {s['name']}: {s['reason']}")

            print("\n✅ All tests completed!")

            # Keep browser open for a moment to see results
            await page.wait_for_timeout(3000)

        except PlaywrightTimeout:
            print("❌ Timeout: Could not connect to http://localhost:1420")
            print("   Make sure the dev server is running: npm run dev:react")
        except Exception as e:
            print(f"❌ Test suite error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await browser.close()

if __name__ == '__main__':
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
