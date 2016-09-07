#encoding: utf-8
import subprocess
import os
import json

MAX_PAGES_CHECK = 2

output = []
for f in list(os.walk('img_data'))[0][2][:50]:
  print "Checking file %s" % f
  for i in xrange(MAX_PAGES_CHECK):
    if f.endswith(".%s.png" % i):
      out = subprocess.check_output(["tesseract", "img_data/%s" %f, "stdout", "-l", "ukr"])
      if u'еклара' in out.decode('utf-8').lower():
        print "File found!"
        output.append(f)

with open("report.json", "w") as fp:
    json.dump(output, fp)

with open('report.html', 'w') as fp:
  fp.write('<body>')
  for img in output:
    fp.write('<img src="img_data/%s"><br/>' % img)
  fp.write('</body>')

