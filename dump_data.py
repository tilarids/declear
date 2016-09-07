import json
import requests
import hashlib
import os
from time import sleep

TO_DOWNLOAD = {}

with open("feed.json", "r") as fp:
    data = json.load(fp)
    for decl in data:
      if "declaration" in decl and 'url' in decl['declaration']:
        url = decl['declaration']['url']
        if '.pdf' in url.lower():
          TO_DOWNLOAD[hashlib.sha224(url.encode('utf-8')).hexdigest() + '.pdf'] = url

for i, (key, val) in enumerate(TO_DOWNLOAD.iteritems()):
  if os.path.isfile('data/%s' % key):
    print "PDF #%s: %s already downloaded" % (i, val)
    continue
  print("Fetching PDF #%s: %s" % (i, val))
  r = requests.get(val, stream=True)
  with open('data/%s' % key, 'wb') as fd:
    for chunk in r.iter_content(1024 * 512):
        fd.write(chunk)
  sleep(0.5)
