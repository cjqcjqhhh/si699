from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
app.secret_key = 'your-secret-key'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# 模拟一个简单的用户数据库
users = {
    'user1': {'password': 'password1'},
    'user2': {'password': 'password2'}
}

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id) if user_id in users else None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']
        if user_id in users and users[user_id]['password'] == password:
            user = User(user_id)
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('用户名或密码错误', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/search', methods=['POST'])
def search():
    search_text = request.form['search_text']
    result = {
        "Project A": {
            "url": "https://github.com/user/project-a",
            "readme": "This is a sample project A. It does some interesting things and is an example for the MAGI search engine."
        },
        "Project B": {
            "url": "https://github.com/user/project-b",
            "readme": "This is a sample project B. It is another example for the MAGI search engine, demonstrating different projects."
        },
        "Project C": {
            "url": "https://github.com/user/project-c",
            "readme": "This is a sample project C."
        },
        "Project D": {
            "url": "https://github.com/user/project-d",
            "readme": "This is a sample project D."
        },
        "Project E": {
            "url": "https://github.com/user/project-e",
            "readme": "This is a sample project E."
        }
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
