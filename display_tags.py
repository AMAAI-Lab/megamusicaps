import argparse
import tkinter as tk
from tkinter import ttk
from pygame import mixer
import json

class MusicPlayerApp:
    def __init__(self, root, caption_file_path, media_root_directory):
        self.root = root
        self.root.title("Music Player")

        self.json_file_path = caption_file_path

        self.media_root_directory = media_root_directory

        self.file_paths = []
        self.current_index = 0

        self.create_widgets()

    def create_widgets(self):
        self.label_caption = tk.Label(self.root, text="Caption:")
        self.label_caption.pack(pady=10)

        self.button_previous = ttk.Button(self.root, text="Previous", command=self.play_previous)
        self.button_previous.pack(side=tk.LEFT, padx=10)

        self.button_play = ttk.Button(self.root, text="Play", command=self.play_current)
        self.button_play.pack(side=tk.LEFT, padx=10)

        self.button_next = ttk.Button(self.root, text="Next", command=self.play_next)
        self.button_next.pack(side=tk.LEFT, padx=10)

        self.load_file_paths()

    def load_file_paths(self):
        with open(self.json_file_path, 'r') as json_file:
            for line in json_file:
                item = json.loads(line)
                media_path = self.media_root_directory + item["location"]
                self.file_paths.append(media_path)

        if self.file_paths:
            self.play_current()

    def play_current(self):
        if self.file_paths:
            mixer.init()
            mixer.music.load(self.file_paths[self.current_index])
            mixer.music.play()

            # Display the caption
            self.display_caption()

    def play_previous(self):
        if self.file_paths:
            self.current_index = (self.current_index - 1) % len(self.file_paths)
            self.play_current()

    def play_next(self):
        if self.file_paths:
            self.current_index = (self.current_index + 1) % len(self.file_paths)
            self.play_current()

    def display_caption(self):
        with open(self.json_file_path, 'r') as json_file:
            for i, line in enumerate(json_file):
                if i == self.current_index:
                    item = json.loads(line)
                    # caption = item["caption"]
                    caption = ""
                    for key in item.keys():
                        if not key in ["caption", "location"]:
                            if key == "beats":
                                caption = caption + "bpm : " + str(item[key]["bpm"]) + "\n"
                            elif key == "gender":
                                if item["voice"] != "instrumental":
                                    caption = caption + str(key) + " : " + str(item[key]) + "\n"
                                else:
                                    continue
                            elif key == "chords":
                                caption = caption + str(key) + " "
                                for i in range(len(item[key])):
                                    caption = caption + str(item[key][i].strip("\n"))
                                    if i != len(item[key]) - 1:
                                        caption = caption + ", "
                                    if i%4 == 0 and i >= 4:
                                        caption = caption + "\n"
                            else:
                                caption = caption + str(key) + " : " + str(item[key]) + "\n"
                    self.label_caption.config(text=caption)
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Json with generated captions")
    parser.add_argument("caption_file", help="Path to the file with path and captions")
    parser.add_argument("media_root_directory", default="", help="Path to root folder of media files")
    args = parser.parse_args()

    root = tk.Tk()
    app = MusicPlayerApp(root, args.caption_file, args.media_root_directory)
    root.mainloop()
