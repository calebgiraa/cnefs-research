import os
import time
import requests
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SERVER_URL = 'http://10.153.48.151:5000/upload'

SOURCE_IMAGE_DIR = os.path.join(os.getcwd(), 'new_images_to_send')

SENT_IMAGE_DIR = os.path.join(os.getcwd(), 'sent_images')

CHECK_INTERVAL_SECONDS = 10

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}


if not os.path.exists(SOURCE_IMAGE_DIR):
    os.makedirs(SENT_IMAGE_DIR)
    logging.info(f"Created source image directory: {SOURCE_IMAGE_DIR}")

if not os.path.exists(SENT_IMAGE_DIR):
    os.makedirs(SENT_IMAGE_DIR)
    logging.info(f"Created sent image directory: {SENT_IMAGE_DIR}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def send_image(filepath, server_url):
    filename = os.path.basename(filepath)
    if not allowed_file(filename):
        logging.warning(f"Skipping {filename}: Not an allowed file type.")
        return False
    
    try:
        with open(filepath, 'rb') as f:
            files = {'image': (filename, f, 'application/octet-stream')}
            response = requests.post(server_url, files=files)

        if response.status_code == 200:
            logging.info(f"Succesfully sent {filename}. Server response: {response.json()}")
            return True
        else:
            logging.info(f"Failed to send {filename}. Status code: {response.status_code}, Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection error when sending {filename}: {e}. Is the server running and reachable at {server_url}?")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occured while sending {filename}: {e}")
        return False

def monitor_and_send():
    logging.info(f"Monitoring directory: {SOURCE_IMAGE_DIR} for new images...")
    logging.info(f"Will send to: {SERVER_URL}")

    while True:
        try:
            found_new_images = False
            for filename in os.listdir(SOURCE_IMAGE_DIR):
                filepath = os.path.join(SOURCE_IMAGE_DIR, filename)

                if os.path.isfile(filepath) and allowed_file(filename):
                    logging.info(f"Found new image: {filename}")
                    found_new_images = True
                    if send_image(filepath, SERVER_URL):
                        shutil.move(filepath, os.path.join(SENT_IMAGE_DIR, filename))
                        logging.info(f"Moved {filename} to {SENT_IMAGE_DIR}")
                    else:
                        logging.warning(f"Could not send {filename}. Will retry later.")
                else:
                    logging.debug(f"Skipping non-image file or directory: {filename}")
            
            if not found_new_images:
                logging.info(f"No new images found. Checking again in {CHECK_INTERVAL_SECONDS} seconds.")
        except FileNotFoundError:
            logging.error(f"Source directory not found: {SOURCE_IMAGE_DIR}. Please ensure it exists.")
        except Exception as e:
            logging.error(f"An error occured in the monitoring loop: {e}")
        
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == '__main__':
    if '0.0.0.0' in SERVER_URL:
        logging.error("Please update 'SERVER_URL' in the script with your processing machine's actual IP address or host.")
    else:
        monitor_and_send()