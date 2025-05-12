from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog

import argparse
from PIL.ImageTk import PhotoImage
from PIL import Image
from src.image_processor import ImageProcessor
from src.stored_data_manager import DbError, StoredDataManager
import numpy as np

mark_rad = 5

class Application:
    def __init__(self):
        # exec args parsing
        parser = argparse.ArgumentParser()
        parser.add_argument("-b", "--database_dir", help="Path to the cache directory", default="./database")
        parser.add_argument("-s", "--scandir", help="Path to the directory with raw images")

        args = parser.parse_args()
        self.d_man = StoredDataManager(args.database_dir)

        if args.scandir is not None:
            self.d_man.set_img_dir(args.scandir)

        # data
        self.image = None
        self.processor = None
        self.corners = None
        self.edges = None
        self.lines = None
        self.labels = None
        self.canvas_size = (960, 960)
        self.img_scale = 1
        self.selected_corner = -1

        # GUI init
        self.tkRoot = Tk()
        self.tkRoot.protocol("WM_DELETE_WINDOW", self._on_closing)
        for i in range(1, 10):
            self.tkRoot.bind(f"<KeyPress-{i}>", self._schema_keybind)
        self.tkRoot.bind("<space>", lambda _: self._finalize_image())
            
        self.tkRoot.geometry()
        self.left_frame = Frame(self.tkRoot)
        self.left_frame.pack(side=LEFT)
        self.right_frame = Frame(self.tkRoot)
        self.right_frame.pack(side=RIGHT, expand=NO, fill=Y)
        self.canvas = Canvas(self.left_frame, width=self.canvas_size[0], height=self.canvas_size[1], bg='white')
        self.canvas.pack(expand=YES, fill=BOTH)

        # menus
        self.menu = Menu(self.tkRoot)
        self.load_menu = Menu(self.menu)
        self.load_menu.add_command(label="Select data menu", command=self._select_data_menu)
        self.load_menu.add_command(label="Reload schemas", command=self._load_schemas)
        self.load_menu.add_command(label="Save data", command=lambda: self.d_man.save_mapping_data())
        self.menu.add_cascade(label="File", menu=self.load_menu)

        self.tkRoot.config(menu=self.menu)
        self.tkRoot.bind("<KeyPress-Up>", lambda e: self._on_arrow_key_pressed([0, -1]))
        self.tkRoot.bind("<KeyPress-Down>", lambda e: self._on_arrow_key_pressed([0, 1]))
        self.tkRoot.bind("<KeyPress-Left>", lambda e: self._on_arrow_key_pressed([-1, 0]))
        self.tkRoot.bind("<KeyPress-Right>", lambda e: self._on_arrow_key_pressed([1, 0]))    
        self.tkRoot.bind("<Shift-KeyPress-Up>", lambda e: self._on_arrow_key_pressed([0, -10]))
        self.tkRoot.bind("<Shift-KeyPress-Down>", lambda e: self._on_arrow_key_pressed([0, 10]))
        self.tkRoot.bind("<Shift-KeyPress-Left>", lambda e: self._on_arrow_key_pressed([-10, 0]))
        self.tkRoot.bind("<Shift-KeyPress-Right>", lambda e: self._on_arrow_key_pressed([10, 0]))
        self.tkRoot.bind("<Control-Shift-KeyPress-Up>", lambda e: self._on_arrow_key_pressed([0, -100]))
        self.tkRoot.bind("<Control-Shift-KeyPress-Down>", lambda e: self._on_arrow_key_pressed([0, 100]))
        self.tkRoot.bind("<Control-Shift-KeyPress-Left>", lambda e: self._on_arrow_key_pressed([-100, 0]))
        self.tkRoot.bind("<Control-Shift-KeyPress-Right>", lambda e: self._on_arrow_key_pressed([100, 0]))


        # controls
        self.letter_schema_cb = ttk.Combobox(self.right_frame, takefocus=False)
        self.letter_schema_cb.grid(row=0, column=0, columnspan=2, padx=0)
        self.letter_schema_cb.bind("<<ComboboxSelected>>", lambda e: self._draw_label_text())    
        
        self.v_thickness_sb = ttk.Spinbox(self.right_frame, from_=0, to=100, increment=0.5, command=self._draw_bounding_rect, takefocus=False)
        self.h_thickness_sb = ttk.Spinbox(self.right_frame, from_=0, to=100, increment=0.5, command=self._draw_bounding_rect, takefocus=False)
        self.h_thickness_sb.set(3)
        self.v_thickness_sb.set(3)
        self.ht_label = Label(self.right_frame, text='H thickness')
        self.vt_label = Label(self.right_frame, text='V thickness')
        self.vt_label.grid(row=1, column=0)
        self.v_thickness_sb.grid(row=1, column=1)        
        self.ht_label.grid(row=2, column=0)
        self.h_thickness_sb.grid(row=2, column=1)
        self.rb_frame = Frame(self.right_frame)
        self.rb_frame.grid(row=3, column=0, columnspan=2)
        self.rot_var = IntVar()
        self.rot_radios = [
            Radiobutton(self.rb_frame, text=txt, variable=self.rot_var, value=v)
            for v, txt in zip([0, 90, 180, 270], ['Up', 'Left', 'Down', 'Right'])
        ]
        self.rot_radios[0].grid(row=0, column=1)
        self.rot_radios[1].grid(row=1, column=0)
        self.rot_radios[2].grid(row=2, column=1)
        self.rot_radios[3].grid(row=1, column=2)
        

        self.next_img_button = Button(self.right_frame, text="Next image", command=self._finalize_image)
        self.next_img_button.grid()

        # init canvas components
        self.image_container = self.canvas.create_image(0, 0, anchor=NW)
        self.canvas.tag_bind(self.image_container, "<Button-1>", self._on_image_click)

        self.tag_texts = [
            self.canvas.create_text(-100, -100, state='hidden', anchor=SE)
            for _ in range(50)
        ]

        self.grid_lines = [
            self.canvas.create_line(0, 0, 0, 0, fill='lime', width=1)
            for _ in range(49 + 24)
        ]

        self.edge_lines = [
            self.canvas.create_line(0, 0, 0, 0, fill='red', width=1)
            for _ in range(4)
        ]
        
        self.corner_marks = [
            self.canvas.create_oval(-100, -100, -100, -100, outline='green', activewidth=5, width=3)
            for _ in range(4)
        ]
        for i, m in enumerate(self.corner_marks):
            self.canvas.tag_bind(m, "<B1-Motion>", self._create_corner_adjust_handler(i))
            self.canvas.tag_bind(m, "<Button-1>", self._create_select_corner_handler(i))

        # init functions
        self._load_schemas()

    def _load_schemas(self):
        try:
            self.d_man.load_schemas()
        except DbError as e:
            messagebox.showerror("Error", str(e))
            return

        self.letter_schema_cb['values'] = list(self.d_man.schemas.keys())

    def _select_data_menu(self):
        img_dir = filedialog.askdirectory(mustexist=True)
        self.d_man.set_img_dir(img_dir)
        pass
    
    def _finalize_image(self):
        if self.image is not None:
            settings = self._get_settings()
            if settings is None:
                return
        
            self.d_man.assign_image_settings(settings)
        self._get_next_image()

    def _get_next_image(self):
        im_path, p_data = self.d_man.get_next_img()
        if im_path is None:
            self.image = None
            self.processor = None
            messagebox.showinfo("Scan over", "No unprocessed images left in scanned directory")
            return
        
        self.tkRoot.title(im_path)
        self.rot_radios[0].invoke()
        self.image = Image.open(im_path).convert('RGB')
        self.processor = ImageProcessor(self.image, 30)
        self.img_scale = min(self.canvas_size[0] / self.image.width, self.canvas_size[1] / self.image.height)
        try:
            self.processor.find_corners()
            self.corners = self.processor.get_corners().reshape((4, 2)) * self.img_scale
        except ValueError:
            self.corners = np.zeros((4, 2))
        self._canvas_draw_image(self.image)
        self._draw_bounding_rect()

    def _draw_bounding_rect(self):
        if self.image is None:
            return 
        
        nw = self.corners[0]
        ne = self.corners[1]
        sw = self.corners[2]
        se = self.corners[3]
        
        shape = (25, 50) if self.image.width < self.image.height else (50, 25)
        
        wc = np.linspace(nw, sw, shape[1], endpoint=False, axis=0)[1:]
        ec = np.linspace(ne, se, shape[1], endpoint=False, axis=0)[1:]
        nc = np.linspace(nw, ne, shape[0], endpoint=False, axis=0)[1:]
        sc = np.linspace(sw, se, shape[0], endpoint=False, axis=0)[1:]
        
        v = np.column_stack([wc, ec]).reshape((shape[1] - 1, 4))
        h = np.column_stack([nc, sc]).reshape((shape[0] - 1, 4))
        
        # draw inner lines
        for i, p in enumerate(v):
            self.canvas.coords(self.grid_lines[i], list(p))
            self.canvas.itemconfig(self.grid_lines[i], width=self.h_thickness_sb.get())
        for i, p in enumerate(h, shape[1] - 1):
            self.canvas.coords(self.grid_lines[i], list(p))
            self.canvas.itemconfig(self.grid_lines[i], width=self.v_thickness_sb.get())

    # draw outer edges
        self.canvas.coords(self.edge_lines[0], [*nw, *ne])
        self.canvas.coords(self.edge_lines[1], [*nw, *sw])
        self.canvas.coords(self.edge_lines[2], [*se, *ne])
        self.canvas.coords(self.edge_lines[3], [*se, *sw])
        
        # draw marks
        for i, p in enumerate(self.corners):
            self.canvas.coords(self.corner_marks[i], [*(p - mark_rad), *(p + mark_rad)])
            
        self._draw_label_text()

    def _draw_label_text(self):
        ci = self.letter_schema_cb.get()
        if ci == '':
            return
        
        schema = self.d_man.schemas[ci]
        tags = schema['tags']
        if len(tags) == 25:
            offset = np.array((2, -5))
            c1 = self.corners[0]
            c2 = self.corners[1]
            angle = -90
            anchor = SE
        else:
            offset = np.array((-2, 2))
            c1 = self.corners[0]
            c2 = self.corners[2]
            angle = 0
            anchor = NE
            
        spacing = np.linspace(c1, c2, len(tags), endpoint=False, axis=0)
               
        for m, t, s in zip(self.tag_texts, tags, spacing):
            self.canvas.itemconfig(m, text=t, state='normal', angle=angle, anchor=anchor)
            self.canvas.coords(m, list(s + offset))
        
        for m in self.tag_texts[len(tags):]:
            self.canvas.itemconfig(m, state='hidden')

    def _create_corner_adjust_handler(self, index):
        def handler(event):
            self._adjust_corner(index, [event.x, event.y])
        return handler
    
    def _get_settings(self):
        schema = self.letter_schema_cb.get()
        if schema == '':
            messagebox.showwarning("Cannot proceed", "You have not selected a schema")
            return None
        
        return {
            "schema_key": schema,
            "corners": np.round(self.corners / self.img_scale).astype(int).tolist(),
            "h_width": float(self.h_thickness_sb.get()) / self.img_scale,
            "v_width": float(self.v_thickness_sb.get()) / self.img_scale,
            "rotation": self.rot_var.get()
        }
    
    def _create_select_corner_handler(self, index):
        def handler(_):
            self.selected_corner = index
            self.tkRoot.focus_set()
        return handler

    def _adjust_corner(self, corner, new_pos):
        if self.corners is None:
            return
        self.tkRoot.focus_set()
        self.corners[corner] = np.array(new_pos)
        self.selected_corner = corner
        self._draw_bounding_rect()

    def _on_arrow_key_pressed(self, delta):
        if self.selected_corner < 0:
            return
        
        corner = self.corners[self.selected_corner]
        corner += np.array(delta, dtype=float) * self.img_scale
        self._adjust_corner(self.selected_corner, corner)
            

    def _canvas_draw_image(self, image: Image):
        p_image = PhotoImage(image.resize((int(image.width * self.img_scale), int(image.height * self.img_scale)), 2))

        self.canvas.itemconfig(self.image_container, image=p_image)
        self.canvas.imgref = p_image
        
    def _schema_keybind(self, event):
        try:
            key = int(event.keysym)
        except any:
            return
        if key - 1 > len(self.letter_schema_cb.keys()):
            return
        self.letter_schema_cb.current(key - 1)
        self._draw_label_text()

    def _on_image_click(self, event):
        self.selected_corner = (0 if event.x < self.image.width * self.img_scale / 2 else 1) + (0 if event.y < self.image.height * self.img_scale / 2 else 2)
        self._adjust_corner(self.selected_corner, [event.x, event.y])
    
    def _on_closing(self):
        self.d_man.save_mapping_data()
        self.tkRoot.destroy()

    def start(self):
        self.tkRoot.mainloop()


if __name__ == "__main__":
    app = Application()
    app.start()
