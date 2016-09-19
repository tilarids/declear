from Tkinter import *
import pg_toolkit as pgt
pgt.toolkit_config.set_pg_conn_string("dbname='decl'")

def get_unlabeled_images():
    return list(pgt.pg_query("""
        SELECT
            fei.data->>'ppm_path' as ppm_path
        FROM full_extract_imgs fei
        WHERE
            fei.data->>'label' is null
      """)['ppm_path'])

CURRENT_INDEX = 0
UNLABELED_IMAGES = get_unlabeled_images()

def onevent(event):
    global CURRENT_INDEX
    if event.keysym not in ['space', 'a', 's', 'd', 'f']:
        return
    img_record = pgt.pg_query_by_id('full_extract_imgs', UNLABELED_IMAGES[CURRENT_INDEX])
    img_record['label'] = event.keysym
    pgt.pg_update_record('full_extract_imgs', UNLABELED_IMAGES[CURRENT_INDEX], img_record)
    CURRENT_INDEX += 1
    redraw_image()

master = Tk()
label = Label(master)
label.pack()
master.bind('<KeyPress>', onevent)

def redraw_image():
    ppm_path = UNLABELED_IMAGES[CURRENT_INDEX]
    pi = PhotoImage(file=ppm_path).zoom(5,5)
    label.image = pi
    label.config(image=pi)

redraw_image()

mainloop()
