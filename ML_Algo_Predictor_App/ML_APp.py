import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor  # Added XGBoost import
import time

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Best Machine Learning Algorithm Prediction ")
        self.root.state('zoomed')  # Full screen

        # Load images
        self.bg_image = ImageTk.PhotoImage(Image.open("background2.jpg").resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight())))
        self.logo_image = ImageTk.PhotoImage(Image.open("logo1.png").resize((200, 200)))

        self.current_user = None
        self.feature_indices = None
        self.target_indices = None
        self.data = None
        self.best_model = None
        self.best_model_type = None

        self.frames = {}
        for F in (WelcomePage, LoginPage, SignupPage, ServicesPage, UploadPage, InputPage, ResultPage):
            page_name = F.__name__
            frame = F(parent=self.root, controller=self)
            frame.place(relwidth=1, relheight=1)
            self.frames[page_name] = frame

        self.show_frame("WelcomePage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def authenticate(self, username, password):
        conn = sqlite3.connect("ml_app.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()
        return user

    def register_user(self, username, password):
        conn = sqlite3.connect("ml_app.db")
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
        c.execute("INSERT INTO users VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()

    def analyze_classification(self, X, y):
        start_time = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Support Vector Machine": SVC(),
            "Kernel SVM": SVC(kernel='rbf'),
            "Naive Bayes": GaussianNB(),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "Random Forest Classifier": RandomForestClassifier(),
            "XGBoost Classifier": XGBClassifier()  # Added XGBoost Classifier
        }

        results = []
        best_algorithm = None
        best_accuracy = 0
        best_model = None
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred)
            results.append({
                "Algorithm": name,
                "Accuracy": accuracy,
                "Confusion Matrix": confusion
            })
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_algorithm = name
                best_model = model

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Classification analysis completed in {elapsed_time:.2f} seconds.")

        self.best_model = best_model
        self.best_model_type = "classification"
        return results, best_algorithm

    def analyze_regression(self, X, y):
        start_time = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        models = {
            "Linear Regression": LinearRegression(),
            "Polynomial Linear Regression": LinearRegression(),  # Placeholder for simplicity
            "Support Vector Regression": SVR(),
            "Decision Tree Regression": DecisionTreeRegressor(),
            "Random Forest Regression": RandomForestRegressor(),
            "XGBoost Regressor": XGBRegressor()  # Added XGBoost Regressor
        }

        results = []
        best_algorithm = None
        best_r_squared = float('-inf')
        best_model = None
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r_squared = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            results.append({
                "Algorithm": name,
                "R-Squared Score": r_squared,
                "Mean Squared Error": mse,
                "Root Mean Squared Error": np.sqrt(mse)
            })
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_algorithm = name
                best_model = model

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Regression analysis completed in {elapsed_time:.2f} seconds.")

        self.best_model = best_model
        self.best_model_type = "regression"
        return results, best_algorithm

# Classes for WelcomePage, LoginPage, SignupPage, ServicesPage, UploadPage, InputPage, and ResultPage remain unchanged.


# Classes for WelcomePage, LoginPage, SignupPage, ServicesPage, UploadPage, InputPage, and ResultPage remain unchanged.
# Ensure to include these classes in your final code as in the provided code above.

class WelcomePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Set background image
        bg_label = tk.Label(self, image=controller.bg_image)
        bg_label.place(relwidth=1, relheight=1)

        # Add widgets on top of the background
        logo_label = tk.Label(self, image=controller.logo_image, bg="#2b2b2b")
        logo_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

        company_name_label = tk.Label(self, text="ML Algo Predictor", font=("Helvetica", 24), bg="#2b2b2b", fg="white")
        company_name_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        developer_name_label = tk.Label(self, text="Developed by Tejaswini", font=("Helvetica", 16), bg="#2b2b2b", fg="white")
        developer_name_label.place(relx=0.5, rely=0.55, anchor=tk.CENTER)

        description_label = tk.Label(self, text="We provide various Machine Learning services including classification and regression algorithm predictions.",
                                     wraplength=800, bg="#2b2b2b", fg="white")
        description_label.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

        button_frame = tk.Frame(self, bg="#2b2b2b")
        button_frame.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

        login_button = tk.Button(button_frame, text="Login", command=lambda: controller.show_frame("LoginPage"), bg="#4caf50", fg="white", width=15, height=2)
        login_button.pack(side="left", padx=20)

        signup_button = tk.Button(button_frame, text="Signup", command=lambda: controller.show_frame("SignupPage"), bg="#2196f3", fg="white", width=15, height=2)
        signup_button.pack(side="right", padx=20)


class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#2b2b2b")
        self.controller = controller

        tk.Label(self, text="Login", font=("Helvetica", 24), bg="#2b2b2b", fg="white").pack(pady=10)

        tk.Label(self, text="Username", bg="#2b2b2b", fg="white").pack(pady=5)
        self.username_entry = tk.Entry(self)
        self.username_entry.pack(pady=5)

        tk.Label(self, text="Password", bg="#2b2b2b", fg="white").pack(pady=5)
        self.password_entry = tk.Entry(self, show="*")
        self.password_entry.pack(pady=5)

        tk.Button(self, text="Login", command=self.login_user, bg="#4caf50", fg="white").pack(pady=10)
        tk.Button(self, text="Back", command=lambda: controller.show_frame("WelcomePage"), bg="#2196f3", fg="white").pack(pady=5)

    def login_user(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if self.controller.authenticate(username, password):
            self.controller.current_user = username
            messagebox.showinfo("Success", "Login successful")
            self.controller.show_frame("ServicesPage")
        else:
            messagebox.showerror("Error", "Invalid username or password")

class SignupPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#2b2b2b")
        self.controller = controller

        tk.Label(self, text="Signup", font=("Helvetica", 24), bg="#2b2b2b", fg="white").pack(pady=10)

        tk.Label(self, text="Username", bg="#2b2b2b", fg="white").pack(pady=5)
        self.username_entry = tk.Entry(self)
        self.username_entry.pack(pady=5)

        tk.Label(self, text="Password", bg="#2b2b2b", fg="white").pack(pady=5)
        self.password_entry = tk.Entry(self, show="*")
        self.password_entry.pack(pady=5)

        tk.Button(self, text="Signup", command=self.signup_user, bg="#4caf50", fg="white").pack(pady=10)
        tk.Button(self, text="Back", command=lambda: controller.show_frame("WelcomePage"), bg="#2196f3", fg="white").pack(pady=5)

    def signup_user(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if username and password:
            self.controller.register_user(username, password)
            messagebox.showinfo("Success", "Signup successful")
            self.controller.show_frame("LoginPage")
        else:
            messagebox.showerror("Error", "Please enter both username and password")

class ServicesPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#2b2b2b")
        self.controller = controller

        tk.Label(self, text="Services", font=("Helvetica", 24), bg="#2b2b2b", fg="white").pack(pady=10)

        tk.Button(self, text="Upload Data", command=lambda: controller.show_frame("UploadPage"), bg="#4caf50", fg="white", width=20, height=2).pack(pady=20)
        tk.Button(self, text="Logout", command=self.logout_user, bg="#2196f3", fg="white", width=20, height=2).pack(pady=20)

    def logout_user(self):
        self.controller.current_user = None
        self.controller.show_frame("WelcomePage")

class UploadPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#2b2b2b")
        self.controller = controller

        tk.Label(self, text="Upload Data", font=("Helvetica", 24), bg="#2b2b2b", fg="white").pack(pady=10)

        tk.Button(self, text="Choose File", command=self.upload_file, bg="#4caf50", fg="white", width=20, height=2).pack(pady=20)
        tk.Button(self, text="Back", command=lambda: controller.show_frame("ServicesPage"), bg="#2196f3", fg="white", width=20, height=2).pack(pady=20)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.controller.data = pd.read_csv(file_path)
                messagebox.showinfo("Success", "File uploaded successfully")
                self.controller.show_frame("InputPage")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to upload file: {e}")

class InputPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#2b2b2b")
        self.controller = controller

        tk.Label(self, text="Input Indices", font=("Helvetica", 24), bg="#2b2b2b", fg="white").pack(pady=10)

        tk.Label(self, text="Analysis Type (classification/regression):", bg="#2b2b2b", fg="white").pack(pady=5)
        self.analysis_type_entry = tk.Entry(self)
        self.analysis_type_entry.pack(pady=5)

        tk.Label(self, text="Feature Indices (e.g., 0:9):", bg="#2b2b2b", fg="white").pack(pady=5)
        self.feature_indices_entry = tk.Entry(self)
        self.feature_indices_entry.pack(pady=5)

        tk.Label(self, text="Number of Target Variables:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.num_targets_entry = tk.Entry(self)
        self.num_targets_entry.pack(pady=5)

        tk.Label(self, text="Target Variable Indices (e.g., 9):", bg="#2b2b2b", fg="white").pack(pady=5)
        self.target_indices_entry = tk.Entry(self)
        self.target_indices_entry.pack(pady=5)

        tk.Button(self, text="Submit", command=self.submit_indices, bg="#4caf50", fg="white").pack(pady=10)
        tk.Button(self, text="Back", command=lambda: controller.show_frame("UploadPage"), bg="#2196f3", fg="white").pack(pady=5)

    def submit_indices(self):
        analysis_type = self.analysis_type_entry.get().lower()
        feature_indices = self.feature_indices_entry.get()
        num_targets = self.num_targets_entry.get()
        target_indices = self.target_indices_entry.get()

        if not analysis_type in ["classification", "regression"]:
            messagebox.showerror("Error", "Please enter a valid analysis type (classification or regression).")
            return

        try:
            feature_indices = list(map(int, feature_indices.split(":")))
            target_indices = list(map(int, target_indices.split(",")))
            num_targets = int(num_targets)
        except ValueError:
            messagebox.showerror("Error", "Invalid indices format. Please enter valid integers.")
            return

        self.controller.feature_indices = feature_indices
        self.controller.target_indices = target_indices

        data = self.controller.data
        if data is not None:
            X = data.iloc[:, feature_indices[0]:feature_indices[1]].values
            y = data.iloc[:, target_indices].values

            if analysis_type == "classification":
                if y.dtype == 'O':  # If target is categorical
                    y = LabelEncoder().fit_transform(y)
                results, best_algorithm = self.controller.analyze_classification(X, y)
            else:
                results, best_algorithm = self.controller.analyze_regression(X, y)

            self.controller.frames["ResultPage"].display_results(results, best_algorithm)
            self.controller.show_frame("ResultPage")
        else:
            messagebox.showerror("Error", "No data available to process.")

class ResultPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#2b2b2b")
        self.controller = controller

        tk.Label(self, text="Results", font=("Helvetica", 24), bg="#2b2b2b", fg="white").pack(pady=10)
        self.results_text = tk.Text(self, height=20, width=80, wrap="word")
        self.results_text.pack(pady=20)

        tk.Button(self, text="Predict", command=self.predict_with_best_model, bg="#4caf50", fg="white", width=20, height=2).pack(pady=10)
        tk.Button(self, text="Back to Services", command=lambda: controller.show_frame("ServicesPage"), bg="#2196f3", fg="white", width=20, height=2).pack(pady=20)

    def display_results(self, results, best_algorithm):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Best Algorithm: {best_algorithm}\n\n")
        for result in results:
            self.results_text.insert(tk.END, f"Algorithm: {result['Algorithm']}\n")
            self.results_text.insert(tk.END, f"Metrics:\n")
            for key, value in result.items():
                if key != "Algorithm":
                    self.results_text.insert(tk.END, f"  {key}: {value}\n")
            self.results_text.insert(tk.END, "\n")

    def predict_with_best_model(self):
        if self.controller.best_model is None:
            messagebox.showerror("Error", "No model available for prediction.")
            return

        input_data = simpledialog.askstring("Input", "Enter input data (comma-separated):")
        if input_data:
            try:
                input_data = np.array(list(map(float, input_data.split(",")))).reshape(1, -1)
                if self.controller.best_model_type == "classification":
                    prediction = self.controller.best_model.predict(input_data)
                else:
                    prediction = self.controller.best_model.predict(input_data)
                messagebox.showinfo("Prediction", f"Predicted Value: {prediction[0]}")
            except ValueError:
                messagebox.showerror("Error", "Invalid input format. Please enter valid numbers.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()
