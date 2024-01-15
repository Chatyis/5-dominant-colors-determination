from tkinter import filedialog

from ttkbootstrap import *

from config import Config

# Components TODO
# Image
# input
# button
# radio btn
# label

tk = tk.Tk()

cluster_amt_entry: Entry
maximum_file_size_entry: Entry
rag_threshold_entry: Entry
file_path_entry: Entry
custom_parameters_yes_radiobutton: Radiobutton
custom_parameters_no_radiobutton: Radiobutton
rag_yes_radiobutton: Radiobutton
rag_no_radiobutton: Radiobutton
calculate_btn: Button


def add_file():
    config = Config()
    file_path = filedialog.askopenfilename()
    config.set_image_path(file_path)
    # file_path_entry.config(state="active")
    file_path_entry.config(state="active")
    file_path_entry.insert(0, file_path)
    file_path_entry.config(state="disabled")
    enable_edit_and_submit()
    print(f"Selected File: {file_path}")


def enable_edit_and_submit():
    global calculate_btn
    global custom_parameters_yes_radiobutton
    global custom_parameters_no_radiobutton

    set_entries_state(True,
                      list({calculate_btn, custom_parameters_yes_radiobutton, custom_parameters_no_radiobutton}))


def calculate():
    print('calculating')


def custom_parameters_toggled(bv: BooleanVar):
    global cluster_amt_entry
    global maximum_file_size_entry
    global rag_yes_radiobutton
    global rag_no_radiobutton
    global rag_threshold_entry

    set_entries_state(bv.get(), list(
        {cluster_amt_entry, maximum_file_size_entry, rag_yes_radiobutton, rag_no_radiobutton}))
    if not bv.get():
        set_entries_state(False, list({rag_threshold_entry}))


def rag_enabled_toggled(bv: BooleanVar):
    global rag_threshold_entry
    set_entries_state(bv.get(), list({rag_threshold_entry}))


def set_entries_state(enable, entries: list[Entry, Radiobutton]):
    for entry in entries:
        entry.config(state="active" if enable else "disabled")


# TODO Singleton z tymi polami wszystkimi i potem go dopiero odpalic na koncu koncu skryptu!!!
# TODO globals -> pass by reference https://realpython.com/python-pass-by-reference/
# TODO Pisac komponentowo jak w Angularze??? Wtedy pola public sa publicami i elo, jest jakiś obiekt Display() i z tego jazda
def init_display():
    config = Config()
    tk.title("Dominant colors determination")
    # root.iconbitmap('path/to/file')

    global cluster_amt_entry
    global maximum_file_size_entry
    global rag_threshold_entry
    global file_path_entry
    global custom_parameters_yes_radiobutton
    global custom_parameters_no_radiobutton
    global rag_yes_radiobutton
    global rag_no_radiobutton
    global calculate_btn

    is_custom_parameters_enabled = BooleanVar()
    is_custom_parameters_enabled.trace("w", lambda name, index, mode,
                                                   bv=is_custom_parameters_enabled: custom_parameters_toggled(bv))

    is_rag_enabled = BooleanVar()
    is_rag_enabled.trace("w", lambda name, index, mode, bv=is_rag_enabled: rag_enabled_toggled(bv))

    loaded_image = Image.open("data/input_images/full_moon.jpg")
    loaded_image = loaded_image.resize((550, 320))  # TODO Resize properly method ready in main.py
    _loaded_image = ImageTk.PhotoImage(loaded_image)
    image_box = Label(image=_loaded_image)
    image_box.grid(column=0, row=0, rowspan=13)

    file_path_entry = Entry(tk)
    file_path_entry.insert(0, "c:/path/to/file")
    file_path_entry.config(state="readonly")
    file_path_entry.grid(column=1, row=0, columnspan=2, padx=10)

    add_file_btn = Button(tk, text="Add image...", bootstyle=SUCCESS, command=add_file)
    add_file_btn.grid(column=1, row=1, columnspan=2)

    Label(tk, text="Custom parameters").grid(column=1, row=2, columnspan=2)

    custom_parameters_yes_radiobutton = Radiobutton(tk, text="Yes", variable=is_custom_parameters_enabled, value=True,
                                                    state="disabled")
    custom_parameters_yes_radiobutton.grid(column=1, row=3)
    custom_parameters_no_radiobutton = Radiobutton(tk, text="No", variable=is_custom_parameters_enabled, value=False,
                                                   state="disabled")
    custom_parameters_no_radiobutton.grid(column=2, row=3)

    Label(tk, text="Cluster amount").grid(column=1, row=4, columnspan=2)

    cluster_amt_entry = Entry(tk, state="disabled")
    cluster_amt_entry.grid(column=1, row=5, columnspan=2)

    Label(tk, text="Maximum file size").grid(column=1, row=6, columnspan=2)

    maximum_file_size_entry = Entry(tk, state="disabled")
    maximum_file_size_entry.grid(column=1, row=7, columnspan=2)

    Label(tk, text="Apply RAG").grid(column=1, row=8, columnspan=2)

    rag_yes_radiobutton = Radiobutton(tk, text="Yes", variable=is_rag_enabled, value=True, state="disabled")
    rag_yes_radiobutton.grid(column=1, row=9)
    rag_no_radiobutton = Radiobutton(tk, text="No", variable=is_rag_enabled, value=False, state="disabled")
    rag_no_radiobutton.grid(column=2, row=9)

    Label(tk, text="RAG Threshold").grid(column=1, row=10, columnspan=2)

    rag_threshold_entry = Entry(tk, state="disabled")
    rag_threshold_entry.grid(column=1, row=11, columnspan=2)

    calculate_btn = Button(tk, text="Calculate", bootstyle=SUCCESS, command=calculate, state="disabled")
    calculate_btn.grid(column=1, row=12, columnspan=2)

    tk.geometry('800x550')
    tk.mainloop()


init_display()

# def app_display_init(): # use this to wrap app, some wzorzec projektowy needed


# app_display_init()
