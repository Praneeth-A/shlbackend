import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

BASE_URL = "https://www.shl.com"
CATALOG_URL = BASE_URL + "/solutions/products/product-catalog/"

HEADERS = {"User-Agent": "Mozilla/5.0"}

def extract_test_rows(table_div):
    table = table_div.select_one("table")
    # print(f"\n\n\\\\\\\\\\\\\\{table}")
    if not table:
        return []

    rows = table.select("tr")[1:]  # Skip header row
    test_entries = []

    for row in rows:
        try:
            link_tag = row.select_one("td.custom__table-heading__title a")
            name = link_tag.get_text(strip=True)
            print(name)
            relative_url = link_tag["href"]
            full_url = urljoin(BASE_URL, relative_url)
            spans = row.select("span.catalogue__circle")
            remote_testing = "No"
            adaptive_irt = "No"

            if len(spans) > 0:
               remote_testing = "Yes" if "-yes" in spans[0].get("class", []) else "No"

            if len(spans) > 1:
                adaptive_irt = "Yes" if "-yes" in spans[1].get("class", []) else "No"

            tags = row.select("span.product-catalogue__key")
            tag_list = sorted(set(t.get_text(strip=True) for t in tags if t.get_text(strip=True)))

            test_entries.append({
                "name": name,
                "url": full_url,
                "remote_testing": remote_testing,
                "adaptive_irt": adaptive_irt,
                "assessment_types": tag_list,
            })

        except Exception as e:
            print(f"[!] Failed to parse a row: {e}")
            continue

    return test_entries


def extract_next_link(table_div):
    next_tag = table_div.find_next("ul", class_="pagination")
    if not next_tag:
        return None
    next_link = next_tag.select_one("li.pagination__item.-arrow.-next a")
    return urljoin(BASE_URL, next_link["href"]) if next_link and "href" in next_link.attrs else None


def scrape_table(start_url, table_index):
    all_entries = []
    url = start_url
    i=0
    while url:
        print(f"[+] Scraping {url}")
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        s=soup.select("div.col-12 ")
        table_wrappers = soup.select("div.col-12 div.custom__table-wrapper")
        # print(f"{len(table_wrappers)}")

        if i==0:
            table_div = table_wrappers[table_index]

            # len(table_wrappers) <= table_index
        else:
         table_div = table_wrappers[0]
        # print(table_div)
        entries = extract_test_rows(table_div)
        all_entries.extend(entries)

        next_url = extract_next_link(table_div)
        url = next_url
        time.sleep(1)
        i=i+1
    return all_entries


def extract_test_page_details(test_url):
    try:
        res = requests.get(test_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        def get_text(title):
            h4 = soup.find("h4", string=title)
            return h4.find_next("p").get_text(strip=True) if h4 else ""

        return {
            "description": get_text("Description"),
            "job_levels": get_text("Job levels"),
            "languages": get_text("Languages"),
            "assessment_length": get_text("Assessment length"),
        }
    except Exception as e:
        print(f"[!] Failed to fetch {test_url} due to: {e}")
        return {}

def scrape_all_tables():
    all_data = []

    for index, label in enumerate(["Table 1", "Table 2"]):
        print(f"\n[🏁] Starting scrape of {label}")
        entries = scrape_table(CATALOG_URL, table_index=index)
        print(entries )

        for entry in entries:
            details = extract_test_page_details(entry["url"])
            entry.update(details)
            time.sleep(1)
        all_data.extend(entries)

    return all_data


if __name__ == "__main__":
    data = scrape_all_tables()
    print(f"\n✅ Total tests scraped: {len(data)}")

    # Optional save
    import json
    with open("shl_final_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
