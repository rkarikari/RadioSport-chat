import bcrypt
password = "@dm1n!@#"
hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
print(hashed.decode('utf-8'))  # Output: e.g., $2b$12$...