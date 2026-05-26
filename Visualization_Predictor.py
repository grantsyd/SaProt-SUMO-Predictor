# -*- coding: utf-8 -*-
"""
Visualization_Predictor.py

Simplified visualization predictor for SUMOylation site prediction.

Functions:
1. Import FASTA file.
2. Import corresponding .npy feature file.
3. Load trained_model/Model1.pkl.
4. Output prediction probability and positive/negative sample judgment.
5. Save prediction results.
"""

import tkinter as tk
from tkinter import filedialog, messagebox

from predictor_core import read_fasta, SUMOPredictor, save_results


class PredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SUMOylation Site Predictor")
        self.root.geometry("850x620")

        self.model_path = "trained_model/Model1.pkl"
        self.predictor = SUMOPredictor(self.model_path)

        self.records = []
        self.results = []

        title_label = tk.Label(
            root,
            text="SUMOylation Site Predictor",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        text_frame = tk.Frame(root)
        text_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.text = tk.Text(
            text_frame,
            width=95,
            height=28,
            font=("Consolas", 10),
            wrap=tk.WORD
        )
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame, command=self.text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.config(yscrollcommand=scrollbar.set)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.btn_predict = tk.Button(
            button_frame,
            text="Import and Predict",
            width=18,
            height=2,
            command=self.import_and_predict
        )
        self.btn_predict.grid(row=0, column=0, padx=15)

        self.btn_save = tk.Button(
            button_frame,
            text="Save Results",
            width=18,
            height=2,
            command=self.save_all
        )
        self.btn_save.grid(row=0, column=1, padx=15)

        self.btn_quit = tk.Button(
            button_frame,
            text="Exit",
            width=18,
            height=2,
            command=root.quit
        )
        self.btn_quit.grid(row=0, column=2, padx=15)

        self.show_message(
            "Please click 'Import and Predict'.\n\n"
            "Steps:\n"
            "1. Select a FASTA file.\n"
            "2. Select the corresponding .npy feature file.\n"
            "3. The predictor will output prediction probabilities and labels.\n"
        )

    def show_message(self, message):
        self.text.insert(tk.END, message + "\n")
        self.text.see(tk.END)

    def import_and_predict(self):
        fasta_path = filedialog.askopenfilename(
            title="Select FASTA file",
            filetypes=[("FASTA files", "*.fasta *.fa *.txt"), ("All files", "*.*")]
        )

        if not fasta_path:
            return

        feature_path = filedialog.askopenfilename(
            title="Select feature file",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )

        if not feature_path:
            return

        try:
            self.records = read_fasta(fasta_path)
            self.results = self.predictor.predict_by_features(self.records, feature_path)

            self.text.delete("1.0", tk.END)
            self.show_message("Prediction Results:\n")

            for item in self.results:
                self.show_message(
                    f"ID: {item['id']}\n"
                    f"Sequence: {item['sequence']}\n"
                    f"Probability: {item['probability']:.6f}\n"
                    f"Prediction: {item['prediction']}\n"
                    f"{'-' * 60}"
                )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_all(self):
        if not self.results:
            messagebox.showwarning("Warning", "No prediction results to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save prediction results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not save_path:
            return

        try:
            save_results(self.results, save_path)
            messagebox.showinfo("Success", f"Results saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    app = PredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
