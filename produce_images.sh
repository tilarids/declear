set -e

# PDFIUM should contain full path to pdfium_test tool.
for f in `ls ./data/*.pdf`
do
  if [ -f './img_data/${f##*/}.0.png' ];
  then
    echo "File $f is already converted"
  else
    $PDFIUM --png $f
    mv ./data/*.png img_data/
  fi
done
