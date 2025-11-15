import os
import uuid
import shutil

class FileService:
    """Service for file handling."""
    
    @staticmethod
    def save_file(file, upload_folder):
        """Save uploaded file with a unique filename."""
        # Check if file is a string or a file object
        if isinstance(file, str):
            # If it's a string, use it as the filename directly
            source_path = file
            filename = os.path.basename(source_path)
            unique_filename = FileService._generate_unique_filename(filename)
            file_path = os.path.join(upload_folder, unique_filename)
            try:
                # Copy the file instead of saving it
                shutil.copy(source_path, file_path)
                return file_path
            except Exception as e:
                raise Exception(f"Failed to copy file: {str(e)}")
        else:
            # Original logic for file objects
            unique_filename = FileService._generate_unique_filename(file.filename)
            file_path = os.path.join(upload_folder, unique_filename)
            try:
                file.save(file_path)
                return file_path
            except Exception as e:
                raise Exception(f"Failed to save file: {str(e)}")
    def save_file(file, upload_folder):
        """Save uploaded file to the specified folder."""
        # Check if file is a string (path) or a file object
        if isinstance(file, str):
            # It's already a path, just return it or handle differently
            return file
        else:
            # It's a file object, process as before
            unique_filename = FileService._generate_unique_filename(file.filename)
            file_path = os.path.join(upload_folder, unique_filename)
            file.save(file_path)
            return file_path
    
    @staticmethod
    def cleanup_file(file_path):
        """Remove a file from the filesystem."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            raise Exception(f"Failed to clean up file: {str(e)}")
    
    @staticmethod
    def _generate_unique_filename(filename):
        """Generate a unique filename."""
        unique_id = uuid.uuid4().hex
        _, ext = os.path.splitext(filename)
        return f"{unique_id}{ext}"