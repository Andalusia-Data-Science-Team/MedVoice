import sqlite3
import logging
import os
from datetime import datetime
from ..core.config import Config

logger = logging.getLogger(__name__)

class DatabaseService:
    """Service for database operations"""
    
    DB_PATH = Config.DATABASE_PATH
    
    @classmethod
    def initialize_db(cls):
        """Create database tables if they don't exist"""
        try:
            conn = sqlite3.connect(cls.DB_PATH)
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                hashed_password TEXT NOT NULL,
                insertion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create audio_results table (add user_id)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                filename TEXT NOT NULL,
                language TEXT NOT NULL,
                model TEXT NOT NULL,
                is_conversation BOOLEAN NOT NULL,
                raw_text TEXT,
                arabic_text TEXT,
                translation_text TEXT,
                json_data TEXT,
                reasoning TEXT,
                preprocessing_time REAL,
                voice_processing_time REAL,
                llm_processing_time REAL,
                doctor_name TEXT,
                feedback TEXT,
                insertion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            conn.commit()
            logger.info(f"Database initialized at {cls.DB_PATH}")
            return True
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()

    @classmethod
    def register_user(cls, username: str, hashed_password: str) -> int:
        """Register a new user and return their ID."""
        try:
            conn = sqlite3.connect(cls.DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
                (username, hashed_password)
            )
            conn.commit()
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            user_id = cursor.fetchone()[0]
            logger.info(f"User registered: {username}")
            return user_id
        except sqlite3.IntegrityError:
            logger.error(f"User registration failed: Username already exists")
            raise Exception("Username already exists")
        except sqlite3.Error as e:
            logger.error(f"User registration failed: {str(e)}")
            raise Exception(f"User registration failed: {str(e)}")
        finally:
            if conn:
                conn.close()

    @classmethod
    def update_user_password(cls, username, new_pass):
        """Update a user's password if the user exists."""
        try:
            conn = sqlite3.connect(cls.DB_PATH)
            cursor = conn.cursor()

            # Check if the user exists
            cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if not user:
                logger.warning(f"User not found: {username}")
                return False

            # Update the password
            cursor.execute(
                "UPDATE users SET hashed_password = ? WHERE username = ?",
                (new_pass, username)
            )
            conn.commit()
            logger.info(f"Password updated for user: {username}")
            return True

        except sqlite3.Error as e:
            logger.error(f"User password update failed: {str(e)}")
            raise Exception(f"User password update failed: {str(e)}")

        finally:
            if conn:
                conn.close()


    @classmethod
    def verify_user(cls, username: str) -> dict:
        """Verify user exists and return user data."""
        try:
            conn = sqlite3.connect(cls.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, hashed_password FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if user:
                return {"id": user[0], "username": user[1], "hashed_password": user[2]}
            logger.warning(f"User not found: {username}")
            return None
        except sqlite3.Error as e:
            logger.error(f"User verification failed: {str(e)}")
            raise Exception(f"User verification failed: {str(e)}")
        finally:
            if conn:
                conn.close()

    @classmethod
    def save_audio_result(cls, 
                          user_id: int,
                          filename: str, 
                          language: str, 
                          model: str, 
                          is_conversation: bool, 
                          raw_text: str, 
                          arabic_text: str, 
                          translation_text: str, 
                          json_data: str, 
                          reasoning: str, 
                          preprocessing_time: float, 
                          voice_processing_time: float, 
                          llm_processing_time: float,
                          doctor_name: str = None,
                          feedback: str = None):
        """Save audio processing results to database"""
        try:
            conn = sqlite3.connect(cls.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO audio_results 
            (user_id, filename, language, model, is_conversation, raw_text, arabic_text, translation_text, 
            json_data, reasoning, preprocessing_time, voice_processing_time, llm_processing_time,
            doctor_name, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, filename, language, model, is_conversation, raw_text, arabic_text,
                translation_text, json_data, reasoning, preprocessing_time, voice_processing_time,
                llm_processing_time, doctor_name, feedback
            ))
            
            conn.commit()
            result_id = cursor.lastrowid
            logger.info(f"Saved audio result with ID: {result_id}")
            return result_id
        except Exception as e:
            logger.error(f"Error saving audio result: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()
    
    @classmethod
    def get_audio_results(cls, limit=100):
        """Get recent audio processing results"""
        try:
            conn = sqlite3.connect(cls.DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM audio_results
            ORDER BY insertion_date DESC
            LIMIT ?
            ''', (limit,))
            
            results = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Retrieved {len(results)} audio results")
            return results
        except Exception as e:
            logger.error(f"Error retrieving audio results: {str(e)}")
            return []
        finally:
            if conn:
                conn.close()

    @classmethod
    def update_feedback(cls, result_id: int, feedback: str) -> bool:
        """Update feedback for an existing audio result record."""
        try:
            conn = sqlite3.connect(cls.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE audio_results SET feedback = ? WHERE id = ?",
                (feedback, result_id)
            )
            conn.commit()
            
            if cursor.rowcount > 0:
                logger.info(f"Updated feedback for result ID: {result_id}")
                return True
            else:
                logger.warning(f"No record found with ID: {result_id}")
                return False
        except Exception as e:
            logger.error(f"Error updating feedback in database: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()