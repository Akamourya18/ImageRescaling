# Project-I (CEC23)
## Topic: Efficient Image Rescaling

Leveraging several machine learning and image processing techniques for efficient image upscaling.

This flask web application demonstrates the results of various studied algorithms on users' input images.

### Usage Instructions
- Download the 4 trained model files from this link https://www.dropbox.com/sh/4w0njlvtf1lqtey/AACfZ9xbMBEpWu0vCn22Mntma?dl=0 and paste those to models folder before proceeding further.
- Make sure all the dependencies in `requirements.txt` are installed. To install the requirements, run the following command:
  ```
  $ pip install -r requirements.txt
  ```

- Run the following command:
  ```
  $ gunicorn main:app
  ```
  Now the flask webapp is running at `http://localhost:8000` 
