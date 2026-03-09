import mysql.connector
from mysql.connector import Error

def get_connection():
    """
    Create and return a MySQL database connection.
    Update host, user, password, and database as per your XAMPP setup.
    """
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",          # default XAMPP password is empty
            database="peptides_db"
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def close_connection(connection):
    """Close the database connection."""
    if connection is not None and connection.is_connected():
        connection.close()
