# Deploying License Plate Recognition App to Render

This document provides step-by-step instructions for deploying the License Plate Recognition application to Render.com.

## Prerequisites

- A GitHub account
- A Render.com account (sign up at https://render.com)

## Deployment Steps

1. Push your code to GitHub if you haven't already.

2. Log in to your Render account.

3. From the Render dashboard, click on "New" and select "Blueprint" (if you want to use the `render.yaml` configuration).

4. Connect your GitHub repository.

5. Render will automatically detect the `render.yaml` file and set up your web service.

6. Review the settings and click "Apply" to start the deployment process.

7. Wait for the build and deployment to complete (this may take several minutes).

8. Once deployed, your application will be available at the URL provided by Render.

## Manual Deployment (Alternative to Blueprint)

If you prefer to set up the service manually:

1. From the Render dashboard, click on "New" and select "Web Service".

2. Connect your GitHub repository.

3. Configure the service with the following settings:
   - Name: license-plate-recognition
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --config gunicorn.conf.py wsgi:app`

4. Add the following environment variables:
   - PYTHON_VERSION: 3.11.0

5. Configure a disk:
   - Name: data
   - Mount Path: /opt/render/project/src/static
   - Size: 1 GB

6. Click "Create Web Service" to start the deployment process.

## Important Notes

- Tesseract OCR will need to be installed on the server. Render will automatically install it if specified in the `apt.txt` file.
- The application uses a placeholder model by default. For proper functionality, you may need to provide actual model files.
- The free tier of Render has certain limitations:
  - Services on the free plan will spin down after 15 minutes of inactivity
  - File storage is ephemeral and will be cleared when the service is restarted
  - Limited compute resources

## Troubleshooting

If you encounter any issues during deployment:

1. Check the build logs in the Render dashboard
2. Make sure all dependencies are correctly specified in requirements.txt
3. Verify that your app.py is properly configured to work with gunicorn 