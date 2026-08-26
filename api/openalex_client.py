import json
import requests
import os


BASE_URL = "https://api.openalex.org/sources"

TARGET_JOURNALS = 1000
PER_PAGE = 100


def fetch_journals():

    journals = []
    cursor = "*"

    while len(journals) < TARGET_JOURNALS:

        remaining = TARGET_JOURNALS - len(journals)
        per_page = min(PER_PAGE, remaining)

        params = {
            "filter": "type:journal",
            "per-page": per_page,
            "cursor": cursor,
        }

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=30
        )

        if response.status_code != 200:
            print(
                "Request failed:",
                response.status_code,
                response.text
            )
            break

        data = response.json()

        results = data.get("results", [])

        if not results:
            break

        journals.extend(results)

        cursor = data["meta"]["next_cursor"]

        print(
            f"Downloaded {len(journals)}/{TARGET_JOURNALS} journals"
        )

    return journals[:TARGET_JOURNALS]


if __name__ == "__main__":

    journals = fetch_journals()

    os.makedirs("data", exist_ok=True)

    with open("data/journals.json", "w") as file:

        json.dump(
            journals,
            file,
            indent=4
        )

    print(
        f"\nSaved {len(journals)} journals to "
        f"data/journals.json"
    )