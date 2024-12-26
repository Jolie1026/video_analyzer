import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from typing import Dict, Any
from ..utils.i18n import get_text, set_language
from ..main import load_config, process_video, save_results, initialize_system

class VideoAnalyzerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Analyzer")
        
        # Set window size and center it
        window_width = 540
        window_height = 648
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)
        self.root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
        
        # Variables
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.language = tk.StringVar(value="English")
        self.processing = False
        
        # Load default config
        try:
            self.config = load_config()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {str(e)}")
            self.config = {}

        self.create_widgets()
        self.setup_layout()

    def create_widgets(self):
        # Language selection
        lang_frame = ttk.LabelFrame(self.root, text=get_text("language_selection"))
        self.lang_combo = ttk.Combobox(
            lang_frame, 
            textvariable=self.language,
            values=["English", "简体中文", "한국어"],
            state="readonly"
        )
        self.lang_combo.bind("<<ComboboxSelected>>", self.change_language)
        self.lang_combo.pack(padx=5, pady=5)

        # File selection
        file_frame = ttk.LabelFrame(self.root, text=get_text("file_selection"))
        
        ttk.Label(file_frame, text=get_text("video_file")).pack(anchor="w", padx=5, pady=2)
        file_select_frame = ttk.Frame(file_frame)
        self.video_entry = ttk.Entry(
            file_select_frame, 
            textvariable=self.video_path,
            width=50
        )
        self.video_entry.pack(side="left", padx=5)
        ttk.Button(
            file_select_frame,
            text=get_text("browse"),
            command=self.browse_video
        ).pack(side="left", padx=5)
        file_select_frame.pack(fill="x", padx=5, pady=2)

        ttk.Label(file_frame, text=get_text("output_file")).pack(anchor="w", padx=5, pady=2)
        output_select_frame = ttk.Frame(file_frame)
        self.output_entry = ttk.Entry(
            output_select_frame,
            textvariable=self.output_path,
            width=50
        )
        self.output_entry.pack(side="left", padx=5)
        ttk.Button(
            output_select_frame,
            text=get_text("browse"),
            command=self.browse_output
        ).pack(side="left", padx=5)
        output_select_frame.pack(fill="x", padx=5, pady=2)

        # Control buttons
        button_frame = ttk.Frame(file_frame)
        ttk.Button(
            button_frame,
            text=get_text("init_system"),
            command=self.init_system
        ).pack(side="left", padx=5)
        
        self.process_btn = ttk.Button(
            button_frame,
            text=get_text("process_video"),
            command=self.process_video
        )
        self.process_btn.pack(side="left", padx=5)
        button_frame.pack(pady=10)

        # Progress
        self.progress = ttk.Progressbar(
            file_frame,
            mode='determinate',
            maximum=100
        )
        self.progress.pack(fill="x", padx=5, pady=5)

        # Results
        self.results_text = tk.Text(
            file_frame,
            height=20,
            width=70,
            wrap=tk.WORD
        )
        self.results_text.pack(padx=5, pady=5, fill="both", expand=True)

        # Pack frames
        lang_frame.pack(fill="x", padx=10, pady=5)
        file_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def setup_layout(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def change_language(self, event=None):
        lang = self.language.get()
        if lang == "English":
            set_language("en")
        elif lang == "简体中文":
            set_language("zh_CN")
        elif lang == "한국어":
            set_language("ko")
        
        # Update all UI text
        self.update_ui_text()
        
        # Disable language selection after choosing
        self.lang_combo.state(['disabled'])
        messagebox.showinfo(get_text("success"), get_text("language_set"))

    def update_ui_text(self):
        """Update all UI text elements after language change"""
        # Update window title
        self.root.title(get_text("video_analyzer"))
        
        # Update frame labels
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.LabelFrame):
                if "language" in widget.cget("text").lower():
                    widget.configure(text=get_text("language_selection"))
                elif "file" in widget.cget("text").lower():
                    widget.configure(text=get_text("file_selection"))
                    
                # Update labels and buttons inside frames
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Label):
                        if "video" in child.cget("text").lower():
                            child.configure(text=get_text("video_file"))
                        elif "output" in child.cget("text").lower():
                            child.configure(text=get_text("output_file"))
                    elif isinstance(child, ttk.Frame):
                        for subchild in child.winfo_children():
                            if isinstance(subchild, ttk.Button):
                                if "browse" in subchild.cget("text").lower():
                                    subchild.configure(text=get_text("browse"))
                                elif "init" in subchild.cget("text").lower():
                                    subchild.configure(text=get_text("init_system"))
                                elif "process" in subchild.cget("text").lower():
                                    subchild.configure(text=get_text("process_video"))

    def browse_video(self):
        filename = filedialog.askopenfilename(
            title=get_text("select_video"),
            filetypes=(
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            )
        )
        if filename:
            self.video_path.set(filename)
            # Auto-set output path
            if not self.output_path.get():
                video_name = Path(filename).stem
                output_dir = self.config.get('output', {}).get('output_dir', 'output')
                self.output_path.set(
                    os.path.join(output_dir, f"{video_name}_analysis.md")
                )

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title=get_text("select_output"),
            defaultextension=".md",
            filetypes=(
                ("Markdown files", "*.md"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            )
        )
        if filename:
            self.output_path.set(filename)

    def init_system(self):
        if messagebox.askyesno(
            get_text("init_system"),
            get_text("init_confirm")
        ):
            try:
                if initialize_system():
                    messagebox.showinfo(
                        get_text("success"),
                        get_text("init_success")
                    )
                else:
                    messagebox.showerror(
                        get_text("error"),
                        get_text("init_failed")
                    )
            except Exception as e:
                messagebox.showerror(
                    get_text("error"),
                    f"{get_text('init_error')}: {str(e)}"
                )

    def process_video(self):
        if not self.video_path.get():
            messagebox.showerror(
                get_text("error"),
                get_text("select_video_prompt")
            )
            return

        if not self.output_path.get():
            messagebox.showerror(
                get_text("error"),
                get_text("select_output_prompt")
            )
            return

        self.processing = True
        self.process_btn.state(['disabled'])
        self.progress['value'] = 0
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, get_text("processing") + "\n")
        
        try:
            # Process video
            self.progress['value'] = 30
            self.root.update_idletasks()
            results = process_video(self.video_path.get(), self.config)
            
            self.progress['value'] = 70
            self.root.update_idletasks()
            # Save results
            save_results(results, self.output_path.get())
            
            self.progress['value'] = 100
            self.root.update_idletasks()
            
            # Show results in text widget
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results['summary'])
            
            messagebox.showinfo(
                get_text("success"),
                f"{get_text('process_complete')} {self.output_path.get()}"
            )
            
        except Exception as e:
            messagebox.showerror(
                get_text("error"),
                f"{get_text('process_error')}: {str(e)}"
            )
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"{get_text('error')}: {str(e)}")
        
        finally:
            self.processing = False
            self.process_btn.state(['!disabled'])
            self.progress.stop()

def launch_gui():
    root = tk.Tk()
    app = VideoAnalyzerGUI(root)
    root.mainloop()

if __name__ == '__main__':
    launch_gui()
