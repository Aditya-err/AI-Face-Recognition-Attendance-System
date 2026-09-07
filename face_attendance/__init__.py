"""AI Face Recognition Attendance System.

A local web application that:

    registers people  ->  captures their face  ->  stores face embeddings
    ->  live camera recognition  ->  confident match  ->  attendance record
    ->  dashboard & records UI

Face pipeline: OpenCV YuNet (detection) + OpenCV SFace (128-d embeddings).
"""

__version__ = "1.0.0"
