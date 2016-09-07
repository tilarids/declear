import requests
import json
from time import sleep

data = []

print("Fetching page #%s" % 1)

r = requests.get("http://declarations.com.ua/search?format=json").json()
data += r["results"]["object_list"]

for page in range(2, r["results"]["paginator"]["num_pages"] + 1):
    sleep(0.5)
    print("Fetching page #%s" % page)

    subr = requests.get(
        "http://declarations.com.ua/search?format=json&page=%s" % page).json()
    data += subr["results"]["object_list"]

print("Declarations exported %s" % len(data))
with open("feed.json", "w") as fp:
    json.dump(data, fp)
