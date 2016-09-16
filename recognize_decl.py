# encoding: utf-8
import sys
import tempfile
import shutil
import os
import subprocess


import tensorflow as tf
import numpy as np

from extract_util import extract_fields
from model import CtcModel

BATCH_SIZE = 4

def recognize_extracted_image_with_digits(digits_model, session, bin_img):
    batch_seq_len = np.array([bin_img.shape[1]] * BATCH_SIZE)
    batch_inputs = np.array(BATCH_SIZE * [bin_img.T])

    pred = digits_model.run_predictions(session, batch_inputs, batch_seq_len)
    pred_dense = session.run(tf.sparse_to_dense(pred.indices, pred.shape, pred.values))
    out_str = ''
    for x in pred_dense[0]:
        if x < 10:
            out_str += str(x)
        elif x == 10:
            out_str += '.'
    return out_str

def recognize_second_page(img_path):
    record = extract_fields(img_path)
    num_classes = 12

    import cv2
    cv2.imwrite('/tmp/out.png',record['first_bin'])

    for k,v in record.iteritems():
        if isinstance(v, np.ndarray):
            if k.endswith("_bin"):
                max_time_steps = v.shape[1]
                num_features = v.shape[0]
                break

    digits_model = CtcModel(max_time_steps, num_features, num_classes)
    out_record = {}
    ####Run session
    with tf.Session(graph=digits_model.graph) as session:
        print('Initializing')
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        print("Restoring state...")
        saver.restore(session, './train/save')

        for k,v in record.iteritems():
            if isinstance(v, np.ndarray):
                if k.endswith("_bin"):
                    out_record[k] = recognize_extracted_image_with_digits(digits_model, session, v)
    return out_record

def main(file_name, skip_ocr=False):
    base_dir = tempfile.mkdtemp()
    pdf_name = os.path.join(base_dir, 'temp.pdf')
    shutil.copy(file_name, pdf_name)
    _ = subprocess.check_output(["pdfium_test", "--png", pdf_name])
    os.remove(pdf_name)
    if skip_ocr:
        print ("Skipping OCR step. Assuming the first page of the PDF is the one.")
        return recognize_second_page(os.path.join(base_dir,os.listdir(base_dir)[1]))
    for img_path in os.listdir(base_dir)[:2]:
        try:
            out = subprocess.check_output(["tesseract", os.path.join(base_dir,img_path), "stdout", "-l", "ukr"])
        except:
            continue

        ocr = out.decode('utf-8').lower()
        print ocr
        if ((u'клара' in ocr or u'кпара' in ocr) and (u'одаток' in ocr)):
            x = img_path
            second_page = x[:x[:-4].rfind('.')+1]+str(int(x[x[:-4].rfind('.')+1:-4]) + 1)+".png"
            return recognize_second_page(second_page)
    else:
        print ("Can't find the first page, is it a declaration? Assuming the first page of the PDF is the one.")
        return recognize_second_page(os.path.join(base_dir,os.listdir(base_dir)[1]))
    shutil.rmtree(base_dir)

if __name__ == '__main__':
    print(main(sys.argv[1], True))
