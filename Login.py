import sqlite3
import hashlib
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QMessageBox, QComboBox
from PySide6.QtGui import QIcon

def init_database():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            profession TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

class LoginDialog(QDialog):
    def __init__(self, parent=None, colors=None):
        super().__init__(parent)
        self.colors = colors or {}
        self.username = None  # Store logged-in username
        self.profession = None  # Store logged-in profession
        self.setWindowTitle("Connexion")
        self.setWindowIcon(QIcon('icons/Logo.ico'))
        self.setFixedSize(350, 250)
        self.setStyleSheet(f"""
            background: {self.colors.get('panel_bg', '#1A2333')};
            color: {self.colors.get('text_primary', '#E0E0E0')};
            border-radius: 8px;
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        self.setLayout(layout)
        
        # Title
        title_label = QLabel("Connexion à IRM")
        title_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: {self.colors.get('accent', '#00CED1')};
            padding-bottom: 10px;
        """)
        layout.addWidget(title_label)
        
        # Username
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Nom d'utilisateur")
        self.username_input.setStyleSheet(f"""
            background: {self.colors.get('secondary_dark', '#1E1E2E')};
            color: {self.colors.get('text_primary', '#E0E0E0')};
            border: 1px solid {self.colors.get('accent_dark', '#5A6FB5')};
            border-radius: 4px;
            padding: 8px;
            font-size: 13px;
        """)
        layout.addWidget(self.username_input)
        
        # Password
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Mot de passe")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet(f"""
            background: {self.colors.get('secondary_dark', '#1E1E2E')};
            color: {self.colors.get('text_primary', '#E0E0E0')};
            border: 1px solid {self.colors.get('accent_dark', '#5A6FB5')};
            border-radius: 4px;
            padding: 8px;
            font-size: 13px;
        """)
        layout.addWidget(self.password_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        login_button = QPushButton("Se connecter")
        login_button.setStyleSheet(f"""
            QPushButton {{
                background: {self.colors.get('accent', '#00CED1')};
                color: white;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {self.colors.get('accent_light', '#A4B8FF')};
            }}
        """)
        login_button.clicked.connect(self.handle_login)
        button_layout.addWidget(login_button)
        
        register_button = QPushButton("S'inscrire")
        register_button.setStyleSheet(f"""
            QPushButton {{
                background: {self.colors.get('secondary_dark', '#1E1E2E')};
                color: {self.colors.get('text_primary', '#E0E0E0')};
                border: 1px solid {self.colors.get('accent', '#00CED1')};
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {self.colors.get('accent_dark', '#5A6FB5')};
            }}
        """)
        register_button.clicked.connect(self.show_register_dialog)
        button_layout.addWidget(register_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
    
    def handle_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()
        
        if not username or not password:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer un nom d'utilisateur et un mot de passe.")
            return
        
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash, profession FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] == hash_password(password):
            self.username = username
            self.profession = result[1]
            self.accept()
        else:
            QMessageBox.warning(self, "Erreur", "Nom d'utilisateur ou mot de passe incorrect.")
            
    def show_register_dialog(self):
        register_dialog = RegisterDialog(self, self.colors)
        if register_dialog.exec():
            QMessageBox.information(self, "Succès", "Inscription réussie ! Veuillez vous connecter.")

class RegisterDialog(QDialog):
    def __init__(self, parent=None, colors=None):
        super().__init__(parent)
        self.colors = colors or {}
        self.setWindowTitle("Inscription")
        self.setWindowIcon(QIcon('icons/Logo.ico'))
        self.setFixedSize(350, 350)
        self.setStyleSheet(f"""
            background: {self.colors.get('panel_bg', '#1A2333')};
            color: {self.colors.get('text_primary', '#E0E0E0')};
            border-radius: 8px;
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        self.setLayout(layout)
        
        # Title
        title_label = QLabel("Inscription à IRM")
        title_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: {self.colors.get('accent', '#00CED1')};
            padding-bottom: 10px;
        """)
        layout.addWidget(title_label)
        
        # Username
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Nom d'utilisateur")
        self.username_input.setStyleSheet(f"""
            background: {self.colors.get('secondary_dark', '#1E1E2E')};
            color: {self.colors.get('text_primary', '#E0E0E0')};
            border: 1px solid {self.colors.get('accent_dark', '#5A6FB5')};
            border-radius: 4px;
            padding: 8px;
            font-size: 13px;
        """)
        layout.addWidget(self.username_input)
        
        # Password
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Mot de passe")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet(f"""
            background: {self.colors.get('secondary_dark', '#1E1E2E')};
            color: {self.colors.get('text_primary', '#E0E0E0')};
            border: 1px solid {self.colors.get('accent_dark', '#5A6FB5')};
            border-radius: 4px;
            padding: 8px;
            font-size: 13px;
        """)
        layout.addWidget(self.password_input)
        
        # Confirm Password
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setPlaceholderText("Confirmer le mot de passe")
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        self.confirm_password_input.setStyleSheet(f"""
            background: {self.colors.get('secondary_dark', '#1E1E2E')};
            color: {self.colors.get('text_primary', '#E0E0E0')};
            border: 1px solid {self.colors.get('accent_dark', '#5A6FB5')};
            border-radius: 4px;
            padding: 8px;
            font-size: 13px;
        """)
        layout.addWidget(self.confirm_password_input)
        
        # Profession
        profession_label = QLabel("Profession:")
        profession_label.setStyleSheet(f"color: {self.colors.get('text_secondary', '#B0C4DE')}; font-size: 13px;")
        layout.addWidget(profession_label)
        
        self.profession_combo = QComboBox()
        self.profession_combo.addItems([
            "Utilisateur Standard",
            "Chercheur",
            "Ingénieur du son",
            "Développeur"
        ])
        self.profession_combo.setStyleSheet(f"""
            QComboBox {{
                background: {self.colors.get('secondary_dark', '#1E1E2E')};
                color: {self.colors.get('text_primary', '#E0E0E0')};
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 25px;
            }}
            QComboBox QAbstractItemView {{
                background: {self.colors.get('secondary_dark', '#1E1E2E')};
                color: {self.colors.get('text_primary', '#E0E0E0')};
                selection-background-color: {self.colors.get('accent', '#00CED1')};
            }}
        """)
        layout.addWidget(self.profession_combo)
        
        # Buttons
        button_layout = QHBoxLayout()
        register_button = QPushButton("S'inscrire")
        register_button.setStyleSheet(f"""
            QPushButton {{
                background: {self.colors.get('accent', '#00CED1')};
                color: white;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {self.colors.get('accent_light', '#A4B8FF')};
            }}
        """)
        register_button.clicked.connect(self.handle_register)
        button_layout.addWidget(register_button)
        
        cancel_button = QPushButton("Annuler")
        cancel_button.setStyleSheet(f"""
            QPushButton {{
                background: {self.colors.get('secondary_dark', '#1E1E2E')};
                color: {self.colors.get('text_primary', '#E0E0E0')};
                border: 1px solid {self.colors.get('accent', '#00CED1')};
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {self.colors.get('accent_dark', '#5A6FB5')};
            }}
        """)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
    
    def handle_register(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()
        confirm_password = self.confirm_password_input.text().strip()
        profession = self.profession_combo.currentText()
        
        if not username or not password:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer un nom d'utilisateur et un mot de passe.")
            return
        if password != confirm_password:
            QMessageBox.warning(self, "Erreur", "Les mots de passe ne correspondent pas.")
            return
        
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                QMessageBox.warning(self, "Erreur", "Ce nom d'utilisateur existe déjà.")
                return
            password_hash = hash_password(password)
            cursor.execute("INSERT INTO users (username, password_hash, profession) VALUES (?, ?, ?)",
                         (username, password_hash, profession))
            conn.commit()
            self.accept()
        except sqlite3.Error as e:
            QMessageBox.warning(self, "Erreur", f"Erreur de base de données : {str(e)}")
        finally:
            conn.close()