import json
import os

import numpy as np


class DbError(Exception):

    def __init__(self, *args):
        super().__init__(*args)


class StoredDataManager:
    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.tag_map = json.load(open(os.path.join(self.db_dir, 'tag_map.json'), "r", encoding="utf-8"))
        self.schemas = {}
        self.mapping_data = {}
        self.scan_dir = None
        self.scan_dir_data = None
        self.img_dir_it = None
        self.scan_dir_entry: os.DirEntry | None = None

        self.load_mapping_data()

    def load_mapping_data(self):
        cache_path = os.path.join(self.db_dir, 'map_cache.json')
        if not os.path.exists(cache_path):
            return
        with open(cache_path, "r", encoding="utf-8") as f:
            self.mapping_data = json.load(f)
    
                
    def save_mapping_data(self):
        with open(os.path.join(self.db_dir, 'map_cache.json'), "w", encoding="utf-8") as f:
            json.dump(self.mapping_data, f, ensure_ascii=False)

    def load_schemas(self):
        try:
            schemas = json.load(open(os.path.join(self.db_dir, "letter_schemas.json"), encoding="utf-8"))
        except Exception as e:
            raise DbError("Bad schemas JSON", e)

        for k, schema in schemas.items():
            if "tag_prefix" in schema:
                pref = schema["tag_prefix"]
                schema["tags"] = [pref + t for t in schema["tags"]]

            if any(map(lambda x: x not in self.tag_map, schema["tags"])) or (len(schema["tags"]) != 25 and len(schema["tags"]) != 50):
                raise DbError("Bad schema format")

        self.schemas = schemas

    def set_img_dir(self, img_dir):
        img_dir = os.path.abspath(img_dir)
        self.scan_dir = img_dir
        if img_dir not in self.mapping_data:
            self.mapping_data[img_dir] = {}
        self.scan_dir_data = self.mapping_data[img_dir]

        self.img_dir_it = os.scandir(img_dir)

    def get_next_img(self) -> (np.ndarray | None, dict | None):
        if self.img_dir_it is None:
            return None, None
        name = None
        while name is None or name in self.scan_dir_data:
            try:
                self.scan_dir_entry = next(self.img_dir_it)
                name = self.scan_dir_entry.name
            except StopIteration:
                self.scan_dir_entry = None
                return None, None

        return self.scan_dir_entry.path, self.scan_dir_data.get(self.scan_dir_entry.name, None)

    def assign_image_settings(self, settings):
        self.scan_dir_data[self.scan_dir_entry.name] = settings
