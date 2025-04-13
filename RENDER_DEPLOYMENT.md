# Deploying License Plate Recognition on Render.com

This guide provides step-by-step instructions for deploying your License Plate Recognition application on Render.com's free tier.

## Step 1: Create a Render Account

1. Go to [Render.com](https://render.com/) and sign up for a free account
2. Verify your email and log in to your account

## Step 2: Connect Your GitHub Repository

1. From the Render dashboard, click on the "New +" button in the top right
2. Select "Web Service"
3. Connect your GitHub account if you haven't already
4. Find and select your License Plate Recognition repository

## Step 3: Configure the Web Service

1. Complete the configuration form:
   - **Name**: license-plate-recognition (or your preferred name)
   - **Region**: Choose the region closest to your users
   - **Branch**: main (or your default branch)
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app_simplified:app`

2. Under "Advanced" settings:
   - Set the Python version to 3.9 or higher
   - Add any environment variables if needed

3. Choose the **Free** plan

4. Click "Create Web Service"

## Step 4: Monitor the Deployment

1. Render will now build and deploy your application
2. You can monitor the build and deployment process in the logs
3. Wait for the "Your service is live" message, which provides your URL (e.g., https://license-plate-recognition.onrender.com)

## Step 5: Test Your Application

1. Once deployed, visit your application URL
2. Upload an image to test the license plate recognition
3. Check if the app correctly detects and reads license plates

## Step 6: Understanding Render's Free Tier Limitations

Render's free tier has some limitations you should be aware of:

1. **Sleep after inactivity**: Free services will "sleep" after 15 minutes of inactivity, causing a slight delay on the first request when it wakes up
2. **Limited compute resources**: 512 MB RAM and 0.1 CPU
3. **Build minutes**: 500 free build minutes per month
4. **Bandwidth**: 100 GB/month

## Troubleshooting

If you encounter issues with your deployment:

1. **Application errors**: Check the service logs in the Render dashboard
2. **Missing dependencies**: Update your requirements.txt file and redeploy
3. **Timeout errors**: Your app might need more time to process images than allowed by the default request timeout
4. **Storage issues**: Render's free tier has ephemeral disk storage, meaning files will be lost when the service restarts. Consider using an external storage service like AWS S3 for persistent file storage.

## Enhancements for Production

For a more robust deployment:

1. **Add a persistent storage solution**: Use AWS S3, Google Cloud Storage, or similar service to store uploaded images and processed results
2. **Implement rate limiting**: Prevent abuse of your service by implementing rate limiting
3. **Add user authentication**: Protect your service with user authentication
4. **Upgrade to a paid plan**: For more resources and better performance

## Alternative to Render: Railway.app

If Render doesn't meet your needs, [Railway.app](https://railway.app/) is another excellent option with a generous free tier that includes:

1. 5 projects
2. 512 MB RAM
3. 1 GB disk space
4. Shared CPU
5. $5 of free usage per month

The deployment process is similar to Render, but Railway has a slightly different interface and configuration approach. 