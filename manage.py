#!/usr/bin/env python
"""
manage.py - Django's command-line utility for administrative tasks.
"""

import os
import sys


def main():
    """Entry point for Django administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_detector.settings')

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        message = (
            "Couldn't import Django. Make sure it is installed and the virtual environment is activated.\n"
            "To install Django, run: pip install django"
        )
        raise ImportError(message) from exc

    # Pass the command-line arguments to Django's command execution utility
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()