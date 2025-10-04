# Create the main application file - main.py
main_py_content = '''#!/usr/bin/env python3
"""
Minimal Agentic LLM Builder - Python Desktop MVP
A lightweight desktop application for local LLM interaction and evaluation.
"""

import sys
import os
import yaml
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPlainTextEdit, QTextEdit, QPushButton, QLabel, QFileDialog, 
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QTableWidget, 
    QTableWidgetItem, QProgressBar, QStatusBar, QSplitter, QTabWidget,
    QMessageBox, QLineEdit
)
from PySide6.QtCore import Qt, QThread, QSignal, QTimer
from PySide6.QtGui import QFont, QAction

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    Llama = None


class LlamaWorker(QThread):
    """Worker thread for LLM operations to prevent UI blocking."""
    
    finished = QSignal(str, float)  # response, latency
    error = QSignal(str)
    
    def __init__(self, llama_model, messages, max_tokens, temperature):
        super().__init__()
        self.llama_model = llama_model
        self.messages = messages
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def run(self):
        try:
            start_time = time.time()
            
            response = self.llama_model.create_chat_completion(
                messages=self.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            response_text = response['choices'][0]['message']['content']
            
            self.finished.emit(response_text, latency)
            
        except Exception as e:
            self.error.emit(str(e))


class EvalWorker(QThread):
    """Worker thread for running evaluations."""
    
    progress = QSignal(int, int)  # current, total
    result = QSignal(dict)  # evaluation result
    finished = QSignal(list)  # all results
    error = QSignal(str)
    
    def __init__(self, llama_model, eval_cases, max_tokens, temperature):
        super().__init__()
        self.llama_model = llama_model
        self.eval_cases = eval_cases
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.results = []
    
    def run(self):
        try:
            for i, case in enumerate(self.eval_cases):
                start_time = time.time()
                
                messages = [{"role": "user", "content": case.get("input", "")}]
                
                response = self.llama_model.create_chat_completion(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                latency = (time.time() - start_time) * 1000
                response_text = response['choices'][0]['message']['content'].strip()
                expected = case.get("expected", "").strip()
                
                # Simple pass/fail check - exact match or substring
                passed = expected.lower() in response_text.lower() if expected else True
                
                result = {
                    "input": case.get("input", ""),
                    "expected": expected,
                    "actual": response_text,
                    "passed": passed,
                    "latency_ms": round(latency, 2)
                }
                
                self.results.append(result)
                self.result.emit(result)
                self.progress.emit(i + 1, len(self.eval_cases))
            
            self.finished.emit(self.results)
            
        except Exception as e:
            self.error.emit(str(e))


class LLMBuilderApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.llama_model = None
        self.config = self.load_config()
        self.current_worker = None
        self.eval_worker = None
        
        self.init_ui()
        self.apply_config()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Minimal Agentic LLM Builder")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Model tab
        self.init_model_tab()
        
        # Prompt tab
        self.init_prompt_tab()
        
        # Evaluation tab
        self.init_eval_tab()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load a model to get started")
        
        # Menu bar
        self.init_menu()
    
    def init_menu(self):
        """Initialize menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_model_action = QAction('Load Model...', self)
        load_model_action.triggered.connect(self.load_model_dialog)
        file_menu.addAction(load_model_action)
        
        file_menu.addSeparator()
        
        save_prompt_action = QAction('Save Prompt...', self)
        save_prompt_action.triggered.connect(self.save_prompt)
        file_menu.addAction(save_prompt_action)
        
        load_prompt_action = QAction('Load Prompt...', self)
        load_prompt_action.triggered.connect(self.load_prompt)
        file_menu.addAction(load_prompt_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
    
    def init_model_tab(self):
        """Initialize model loading and configuration tab."""
        model_widget = QWidget()
        self.tab_widget.addTab(model_widget, "Model")
        
        layout = QVBoxLayout(model_widget)
        
        # Model loading group
        model_group = QGroupBox("Model Loading")
        model_layout = QFormLayout(model_group)
        
        self.model_path_edit = QLineEdit()
        model_path_button = QPushButton("Browse...")
        model_path_button.clicked.connect(self.browse_model_path)
        
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(model_path_button)
        
        model_layout.addRow("Model Path (.gguf):", model_path_layout)
        
        # Model controls
        model_controls = QHBoxLayout()
        self.load_button = QPushButton("Load Model")
        self.unload_button = QPushButton("Unload Model")
        self.load_button.clicked.connect(self.load_model)
        self.unload_button.clicked.connect(self.unload_model)
        self.unload_button.setEnabled(False)
        
        model_controls.addWidget(self.load_button)
        model_controls.addWidget(self.unload_button)
        model_controls.addStretch()
        
        model_layout.addRow(model_controls)
        
        layout.addWidget(model_group)
        
        # Model settings group
        settings_group = QGroupBox("Model Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.n_threads_spin = QSpinBox()
        self.n_threads_spin.setRange(1, 32)
        self.n_threads_spin.setValue(4)
        settings_layout.addRow("Threads:", self.n_threads_spin)
        
        self.n_ctx_spin = QSpinBox()
        self.n_ctx_spin.setRange(512, 8192)
        self.n_ctx_spin.setValue(2048)
        settings_layout.addRow("Context Length:", self.n_ctx_spin)
        
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(0.7)
        settings_layout.addRow("Temperature:", self.temperature_spin)
        
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 2048)
        self.max_tokens_spin.setValue(512)
        settings_layout.addRow("Max Tokens:", self.max_tokens_spin)
        
        layout.addWidget(settings_group)
        
        # Model status
        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.model_status_label)
        
        layout.addStretch()
    
    def init_prompt_tab(self):
        """Initialize prompt and response tab."""
        prompt_widget = QWidget()
        self.tab_widget.addTab(prompt_widget, "Prompt & Run")
        
        layout = QHBoxLayout(prompt_widget)
        
        # Left side - prompts
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # System prompt
        left_layout.addWidget(QLabel("System Prompt:"))
        self.system_prompt_edit = QPlainTextEdit()
        self.system_prompt_edit.setPlaceholderText("Enter system prompt here...")
        left_layout.addWidget(self.system_prompt_edit)
        
        # User prompt
        left_layout.addWidget(QLabel("User Prompt:"))
        self.user_prompt_edit = QPlainTextEdit()
        self.user_prompt_edit.setPlaceholderText("Enter user prompt here...")
        left_layout.addWidget(self.user_prompt_edit)
        
        # Run button
        self.run_button = QPushButton("Run Prompt")
        self.run_button.clicked.connect(self.run_prompt)
        self.run_button.setEnabled(False)
        left_layout.addWidget(self.run_button)
        
        layout.addWidget(left_widget)
        
        # Right side - response
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        right_layout.addWidget(QLabel("Response:"))
        self.response_edit = QTextEdit()
        self.response_edit.setReadOnly(True)
        right_layout.addWidget(self.response_edit)
        
        # Stats
        self.stats_label = QLabel("Ready")
        right_layout.addWidget(self.stats_label)
        
        layout.addWidget(right_widget)
    
    def init_eval_tab(self):
        """Initialize evaluation tab."""
        eval_widget = QWidget()
        self.tab_widget.addTab(eval_widget, "Evaluation")
        
        layout = QVBoxLayout(eval_widget)
        
        # Eval controls
        controls_layout = QHBoxLayout()
        
        self.eval_file_edit = QLineEdit()
        self.eval_file_edit.setPlaceholderText("eval.yaml")
        controls_layout.addWidget(QLabel("Eval File:"))
        controls_layout.addWidget(self.eval_file_edit)
        
        browse_eval_button = QPushButton("Browse...")
        browse_eval_button.clicked.connect(self.browse_eval_file)
        controls_layout.addWidget(browse_eval_button)
        
        self.run_eval_button = QPushButton("Run Evaluation")
        self.run_eval_button.clicked.connect(self.run_evaluation)
        self.run_eval_button.setEnabled(False)
        controls_layout.addWidget(self.run_eval_button)
        
        export_button = QPushButton("Export Results")
        export_button.clicked.connect(self.export_results)
        controls_layout.addWidget(export_button)
        
        controls_layout.addStretch()
        layout.addWidget(QWidget())  # Spacer
        layout.itemAt(-1).widget().setLayout(controls_layout)
        
        # Progress bar
        self.eval_progress = QProgressBar()
        layout.addWidget(self.eval_progress)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["Input", "Expected", "Actual", "Pass/Fail", "Latency (ms)"])
        layout.addWidget(self.results_table)
        
        self.eval_results = []
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml."""
        config_path = Path("config.yaml")
        default_config = {
            "last_model_path": "",
            "n_threads": 4,
            "n_ctx": 2048,
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return default_config
    
    def save_config(self):
        """Save current configuration to config.yaml."""
        config = {
            "last_model_path": self.model_path_edit.text(),
            "n_threads": self.n_threads_spin.value(),
            "n_ctx": self.n_ctx_spin.value(),
            "temperature": self.temperature_spin.value(),
            "max_tokens": self.max_tokens_spin.value()
        }
        
        try:
            with open("config.yaml", 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def apply_config(self):
        """Apply loaded configuration to UI elements."""
        self.model_path_edit.setText(self.config.get("last_model_path", ""))
        self.n_threads_spin.setValue(self.config.get("n_threads", 4))
        self.n_ctx_spin.setValue(self.config.get("n_ctx", 2048))
        self.temperature_spin.setValue(self.config.get("temperature", 0.7))
        self.max_tokens_spin.setValue(self.config.get("max_tokens", 512))
    
    def browse_model_path(self):
        """Browse for model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", "", "GGUF Files (*.gguf);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def browse_eval_file(self):
        """Browse for evaluation YAML file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Evaluation File", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            self.eval_file_edit.setText(file_path)
    
    def load_model_dialog(self):
        """Show file dialog and load model."""
        self.browse_model_path()
        if self.model_path_edit.text():
            self.load_model()
    
    def load_model(self):
        """Load the LLM model."""
        if not LLAMA_AVAILABLE:
            QMessageBox.critical(self, "Error", "llama-cpp-python is not installed!")
            return
        
        model_path = self.model_path_edit.text().strip()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "Warning", "Please select a valid model file!")
            return
        
        try:
            self.status_bar.showMessage("Loading model...")
            self.load_button.setEnabled(False)
            
            # Create Llama instance
            self.llama_model = Llama(
                model_path=model_path,
                n_ctx=self.n_ctx_spin.value(),
                n_threads=self.n_threads_spin.value(),
                verbose=False
            )
            
            self.model_status_label.setText(f"Model loaded: {Path(model_path).name}")
            self.model_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.unload_button.setEnabled(True)
            self.run_button.setEnabled(True)
            self.run_eval_button.setEnabled(True)
            self.status_bar.showMessage("Model loaded successfully")
            
            # Save config
            self.save_config()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\\n{str(e)}")
            self.status_bar.showMessage("Model loading failed")
        finally:
            self.load_button.setEnabled(True)
    
    def unload_model(self):
        """Unload the current model."""
        if self.llama_model:
            del self.llama_model
            self.llama_model = None
        
        self.model_status_label.setText("No model loaded")
        self.model_status_label.setStyleSheet("color: red; font-weight: bold;")
        self.unload_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.run_eval_button.setEnabled(False)
        self.status_bar.showMessage("Model unloaded")
    
    def run_prompt(self):
        """Run the current prompt through the model."""
        if not self.llama_model:
            QMessageBox.warning(self, "Warning", "Please load a model first!")
            return
        
        system_prompt = self.system_prompt_edit.toPlainText().strip()
        user_prompt = self.user_prompt_edit.toPlainText().strip()
        
        if not user_prompt:
            QMessageBox.warning(self, "Warning", "Please enter a user prompt!")
            return
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # Disable UI during processing
        self.run_button.setEnabled(False)
        self.status_bar.showMessage("Generating response...")
        
        # Start worker thread
        self.current_worker = LlamaWorker(
            self.llama_model,
            messages,
            self.max_tokens_spin.value(),
            self.temperature_spin.value()
        )
        self.current_worker.finished.connect(self.on_response_finished)
        self.current_worker.error.connect(self.on_response_error)
        self.current_worker.start()
    
    def on_response_finished(self, response: str, latency: float):
        """Handle successful response generation."""
        self.response_edit.setPlainText(response)
        self.stats_label.setText(f"Tokens: ~{len(response.split())} | Latency: {latency:.1f}ms")
        self.status_bar.showMessage("Response generated")
        self.run_button.setEnabled(True)
    
    def on_response_error(self, error: str):
        """Handle response generation error."""
        QMessageBox.critical(self, "Error", f"Failed to generate response:\\n{error}")
        self.status_bar.showMessage("Response generation failed")
        self.run_button.setEnabled(True)
    
    def run_evaluation(self):
        """Run evaluation on the loaded eval file."""
        if not self.llama_model:
            QMessageBox.warning(self, "Warning", "Please load a model first!")
            return
        
        eval_file = self.eval_file_edit.text().strip() or "eval.yaml"
        
        if not Path(eval_file).exists():
            QMessageBox.warning(self, "Warning", f"Evaluation file not found: {eval_file}")
            return
        
        try:
            with open(eval_file, 'r') as f:
                eval_data = yaml.safe_load(f)
            
            if not isinstance(eval_data, list):
                QMessageBox.warning(self, "Warning", "Evaluation file should contain a list of test cases!")
                return
            
            # Clear previous results
            self.eval_results.clear()
            self.results_table.setRowCount(0)
            self.eval_progress.setValue(0)
            self.eval_progress.setMaximum(len(eval_data))
            
            # Disable UI during evaluation
            self.run_eval_button.setEnabled(False)
            self.status_bar.showMessage("Running evaluation...")
            
            # Start evaluation worker
            self.eval_worker = EvalWorker(
                self.llama_model,
                eval_data,
                self.max_tokens_spin.value(),
                self.temperature_spin.value()
            )
            self.eval_worker.progress.connect(self.on_eval_progress)
            self.eval_worker.result.connect(self.on_eval_result)
            self.eval_worker.finished.connect(self.on_eval_finished)
            self.eval_worker.error.connect(self.on_eval_error)
            self.eval_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load evaluation file:\\n{str(e)}")
    
    def on_eval_progress(self, current: int, total: int):
        """Update evaluation progress."""
        self.eval_progress.setValue(current)
        self.status_bar.showMessage(f"Running evaluation... {current}/{total}")
    
    def on_eval_result(self, result: dict):
        """Add evaluation result to table."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        self.results_table.setItem(row, 0, QTableWidgetItem(result["input"][:100] + "..." if len(result["input"]) > 100 else result["input"]))
        self.results_table.setItem(row, 1, QTableWidgetItem(result["expected"][:100] + "..." if len(result["expected"]) > 100 else result["expected"]))
        self.results_table.setItem(row, 2, QTableWidgetItem(result["actual"][:100] + "..." if len(result["actual"]) > 100 else result["actual"]))
        self.results_table.setItem(row, 3, QTableWidgetItem("PASS" if result["passed"] else "FAIL"))
        self.results_table.setItem(row, 4, QTableWidgetItem(str(result["latency_ms"])))
        
        # Color code pass/fail
        pass_item = self.results_table.item(row, 3)
        if result["passed"]:
            pass_item.setBackground(Qt.green)
        else:
            pass_item.setBackground(Qt.red)
        
        self.eval_results.append(result)
    
    def on_eval_finished(self, results: List[dict]):
        """Handle evaluation completion."""
        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        avg_latency = sum(r["latency_ms"] for r in results) / total if total > 0 else 0
        
        self.status_bar.showMessage(f"Evaluation complete: {passed}/{total} passed, avg latency: {avg_latency:.1f}ms")
        self.run_eval_button.setEnabled(True)
        
        # Auto-resize columns
        self.results_table.resizeColumnsToContents()
    
    def on_eval_error(self, error: str):
        """Handle evaluation error."""
        QMessageBox.critical(self, "Error", f"Evaluation failed:\\n{error}")
        self.status_bar.showMessage("Evaluation failed")
        self.run_eval_button.setEnabled(True)
    
    def export_results(self):
        """Export evaluation results to CSV."""
        if not self.eval_results:
            QMessageBox.warning(self, "Warning", "No results to export!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "eval_results.csv", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=["input", "expected", "actual", "passed", "latency_ms"])
                    writer.writeheader()
                    writer.writerows(self.eval_results)
                
                QMessageBox.information(self, "Success", f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export results:\\n{str(e)}")
    
    def save_prompt(self):
        """Save the current prompt to a markdown file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Prompt", "prompt.md", "Markdown Files (*.md);;All Files (*)"
        )
        
        if file_path:
            try:
                content = f"# System Prompt\\n\\n{self.system_prompt_edit.toPlainText()}\\n\\n# User Prompt\\n\\n{self.user_prompt_edit.toPlainText()}"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                QMessageBox.information(self, "Success", f"Prompt saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save prompt:\\n{str(e)}")
    
    def load_prompt(self):
        """Load a prompt from a markdown file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Prompt", "", "Markdown Files (*.md);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple parsing - split by headers
                parts = content.split("# User Prompt")
                if len(parts) == 2:
                    system_part = parts[0].replace("# System Prompt", "").strip()
                    user_part = parts[1].strip()
                    self.system_prompt_edit.setPlainText(system_part)
                    self.user_prompt_edit.setPlainText(user_part)
                else:
                    # Fallback - put everything in user prompt
                    self.user_prompt_edit.setPlainText(content)
                
                QMessageBox.information(self, "Success", "Prompt loaded successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load prompt:\\n{str(e)}")
    
    def closeEvent(self, event):
        """Handle application close event."""
        self.save_config()
        if self.llama_model:
            del self.llama_model
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Minimal Agentic LLM Builder")
    app.setOrganizationName("LLM Tools")
    
    if not LLAMA_AVAILABLE:
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Missing Dependencies")
        msg.setText("llama-cpp-python is not installed!")
        msg.setInformativeText("Please install it using: pip install llama-cpp-python")
        msg.exec()
    
    window = LLMBuilderApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
'''

# Save the main.py file
with open("main.py", "w", encoding="utf-8") as f:
    f.write(main_py_content)

print("Created main.py - Main application file with PySide6 GUI")