#encoding: utf-8
import subprocess
import os
import json

MAX_PAGES_CHECK = 2

output = {}
if os.path.isfile('report.json'):
  with open("report.json", "r") as fp:
    output = json.load(fp)

for f in os.listdir('data')[:5]:
  print "Checking file %s" % f
  if f in output:
    print "Skipping", f
    continue
  should_check_file = True
  for i in xrange(MAX_PAGES_CHECK):
    img_name = "%s.%s.png" % (f,i)
    if not os.path.isfile("img_data/%s" % img_name):
      print "There is no such image: img_data/%s" % img_name
      should_check_file = False
      break
  if not should_check_file:
    continue
  output[f] = None
  for i in xrange(MAX_PAGES_CHECK):
    img_name = "%s.%s.png" % (f,i)

    out = subprocess.check_output(["tesseract", "img_data/%s" % img_name, "stdout", "-l", "ukr"])
    with open("tesseract_data/%s.txt" % img_name, "w") as fp:
      fp.write(out)

    out = out.decode('utf-8').lower()
    if u'клара' in out or u'кпара' in out:
      output[f] = img_name
      break
  else:
    print "DECLARATION NOT FOUND: img_data/%s.0.png tesseract_data/%s.0.png.txt" % (f, f)

#        output.append(f)

with open("report.json", "w") as fp:
    json.dump(output, fp)

with open('report.html', 'w') as fp:
  fp.write('<body>')
  for key, val in output.iteritems():
    if val:
      fp.write('<img src="img_data/%s"><br/>' % val)
  fp.write('</body>')

