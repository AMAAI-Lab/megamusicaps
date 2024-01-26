import argparse
import tkinter as tk
from tkinter import ttk
from pygame import mixer
import json

class MusicPlayerApp:
    def __init__(self, root, caption_file_path):
        self.root = root
        self.root.title("Music Player")

        self.json_file_path = caption_file_path

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
                self.file_paths.append(item["location"])

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
                    caption = item["caption"]
                    self.label_caption.config(text=caption)
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Json with generated captions")
    parser.add_argument("caption_file", help="Path to the file with path and captions")
    args = parser.parse_args()

    root = tk.Tk()
    app = MusicPlayerApp(root, args.caption_file)
    root.mainloop()
