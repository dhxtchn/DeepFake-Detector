from django.core.management.base import BaseCommand
from detector.model_setup import setup_model

class Command(BaseCommand):
    help = 'Downloads and sets up the deepfake detection model'

    def handle(self, *args, **kwargs):
        self.stdout.write('Setting up deepfake detection model...')
        try:
            version = setup_model()
            self.stdout.write(self.style.SUCCESS(
                f'Successfully set up model version {version.version_id}'
            ))
        except Exception as e:
            self.stdout.write(self.style.ERROR(
                f'Error setting up model: {str(e)}'
            )) 