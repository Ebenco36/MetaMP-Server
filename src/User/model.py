import datetime
from database.db import db
from passlib.hash import bcrypt_sha256 as sha256
from flask_bcrypt import generate_password_hash, check_password_hash


class UserModel(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(80), nullable=False)
    username = db.Column(db.String(64), index=True, unique=True)
    phone = db.Column(db.String(15), nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    institute = db.Column(db.String(255), nullable=True)
    location = db.Column(db.String(255), nullable=True)
    password = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(8), default=False, nullable=False)
    is_admin = db.Column(db.Boolean(), default=False, nullable=False)
    has_taken_tour = db.Column(db.Boolean(), default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=True)

    @classmethod
    def find_by_email(cls, email):
        return cls.query.filter_by(email=email).first()

    @classmethod
    def find_all_omit_record_with_this_email(cls, email):
        return cls.query.filter_by(cls.email != email).all()

    @staticmethod
    def generate_hash(password):
        return sha256.hash(password)

    @staticmethod
    def verify_hash(password, hashed_password):
        return sha256.verify(password, hashed_password)
    
    def __init__(self, **kwargs):
        """
        The function takes in a dictionary of keyword arguments and assigns the values to the class
        attributes
        """
        self.username = kwargs.get("username")
        self.name = kwargs.get("name")
        self.phone = kwargs.get("phone")
        self.email = kwargs.get("email")
        self.password = kwargs.get("password")

    def __repr__(self):
        """
        The __repr__ function is used to return a string representation of the object
        :return: The username of the user.
        """
        return "<User {}>".format(self.username)

    def hash_password(self, password):
        """
        It takes the password that the user has entered, hashes it, and then stores the hashed password in
        the database
        """
        self.password = generate_password_hash(password if password else self.password).decode("utf8")

    def check_password(self, password):
        """
        It takes a plaintext password, hashes it, and compares it to the hashed password in the database
        
        :param password: The password to be hashed
        :return: The password is being returned.
        """
        return check_password_hash(self.password, password)
