#!/usr/bin/env python3
"""
Scrape all discussion pages (1 to 13) from Kaggle Gemma 4 Developer Agent Competition
via Chrome Remote Debugging Protocol (CDP).
"""

import json
import re
import time
import urllib.request
import websocket
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_JSON = ROOT_DIR / "documents" / "all_discussions_scraped.json"

def get_kaggle_cdp_ws():
    targets = json.loads(urllib.request.urlopen("http://127.0.0.1:9222/json").read())
    kaggle_pages = [t for t in targets if "kaggle.com" in t.get("url", "") and t.get("type") == "page"]
    if not kaggle_pages:
        raise RuntimeError("No Kaggle page found in Chrome CDP targets")
    return kaggle_pages[0]["webSocketDebuggerUrl"]

def send_cdp(ws, method, params=None, msg_id=1):
    req = {"id": msg_id, "method": method}
    if params:
        req["params"] = params
    ws.send(json.dumps(req))
    while True:
        data = json.loads(ws.recv())
        if data.get("id") == msg_id:
            return data

JS_EXTRACT = r'''
(() => {
    const results = [];
    const seen = new Set();
    const links = Array.from(document.querySelectorAll('a[href*="/discussion/"]'));
    
    for (const a of links) {
        const href = a.href.split('#')[0];
        if (!href.match(/\/discussion\/\d+$/)) continue;
        if (seen.has(href)) continue;
        
        let row = a;
        while (row && row.parentElement && !row.getAttribute('role')?.includes('row') && row.tagName !== 'LI' && row !== document.body) {
            row = row.parentElement;
            if (row.innerText.includes('\n') && row.querySelectorAll('a[href*="/discussion/"]').length >= 1) {
                if (row.innerText.match(/\d+/) && row.innerText.length < 500) {
                    break;
                }
            }
        }
        
        seen.add(href);
        results.push({
            url: href,
            anchorText: a.innerText.trim(),
            rowText: row ? row.innerText.trim() : ''
        });
    }
    return results;
})()
'''

def parse_card(raw_card, page_num):
    url = raw_card["url"]
    topic_id = url.split("/")[-1]
    row_text = raw_card["rowText"]
    
    lines = [line.strip() for line in row_text.split("\n") if line.strip()]
    
    pinned = "push_pin" in lines
    lines = [l for l in lines if l != "push_pin"]
    
    title = lines[0] if lines else raw_card["anchorText"]
    
    # Author is usually in line 1: "Author Name · Last comment..."
    author = "Unknown"
    if len(lines) > 1 and "·" in lines[1]:
        author = lines[1].split("·")[0].strip()
    elif len(lines) > 1:
        author = lines[1].strip()
        
    # Votes between arrow_drop_up and arrow_drop_down
    votes = 0
    if "arrow_drop_up" in lines:
        idx = lines.index("arrow_drop_up")
        if idx + 1 < len(lines):
            try:
                votes = int(lines[idx + 1])
            except ValueError:
                pass
                
    # Comments: regex for "\d+ comment"
    comments = 0
    m_comments = re.search(r"(\d+)\s+comment", row_text)
    if m_comments:
        comments = int(m_comments.group(1))
        
    return {
        "id": topic_id,
        "title": title,
        "author": author,
        "votes": votes,
        "comments": comments,
        "pinned": pinned,
        "url": url,
        "page": page_num
    }

def main():
    print("[1/3] Connecting to Chrome CDP...")
    ws_url = get_kaggle_cdp_ws()
    ws = websocket.create_connection(ws_url, suppress_origin=True)
    
    all_topics = []
    seen_ids = set()
    
    print("[2/3] Scraping Discussion Pages 1 to 13...")
    for page in range(1, 14):
        url = f"https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion?sort=votes&page={page}"
        print(f"  - Navigating to Page {page}...")
        send_cdp(ws, "Page.navigate", {"url": url}, page * 10)
        time.sleep(2.5)  # Wait for React to render topic list
        
        res = send_cdp(ws, "Runtime.evaluate", {"expression": JS_EXTRACT, "returnByValue": True}, page * 10 + 1)
        raw_cards = res["result"]["result"].get("value", [])
        
        page_added = 0
        for card in raw_cards:
            parsed = parse_card(card, page)
            if parsed["id"] not in seen_ids:
                seen_ids.add(parsed["id"])
                all_topics.append(parsed)
                page_added += 1
                
        print(f"    Page {page}: Extracted {len(raw_cards)} cards, {page_added} unique topics added (Total so far: {len(all_topics)})")
        
    ws.close()
    
    print(f"\n[3/3] Saving {len(all_topics)} extracted topics to {OUTPUT_JSON}...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_topics, f, indent=2, ensure_ascii=False)
        
    print(f"SUCCESS: Saved {len(all_topics)} topics to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
