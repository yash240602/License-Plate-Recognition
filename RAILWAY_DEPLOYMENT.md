# Deploying License Plate Recognition on Railway.app

This guide provides step-by-step instructions for deploying your License Plate Recognition application on Railway.app, which offers a generous free tier that includes $5 worth of usage per month.

## Step 1: Create a Railway Account

1. Go to [Railway.app](https://railway.app/) and sign up for a free account
2. Verify your email and log in to your account

## Step 2: Connect Your GitHub Repository

1. From the Railway dashboard, click on the "New Project" button
2. Select "Deploy from GitHub repo"
3. Connect your GitHub account if you haven't already
4. Find and select your License Plate Recognition repository

## Step 3: Configure the Project

1. Once your repository is connected, Railway will automatically detect that it's a Python application
2. Railway will use the existing `Procfile` in your repository (which contains `web: gunicorn app_simplified:app`)
3. Click "Deploy" to start the initial deployment

## Step 4: Configure Environment Variables (Optional)

If you need to set any environment variables:

1. Go to your project's dashboard
2. Click on the "Variables" tab
3. Add any necessary environment variables (not required for this basic deployment)

## Step 5: Set Up the Domain

1. Go to the "Settings" tab of your project
2. Under "Domains", click "Generate Domain"
3. Railway will generate a domain like `license-plate-recognition-production.up.railway.app`
4. This domain is now your public URL for accessing the application

## Step 6: Monitor the Deployment

1. Go to the "Deployments" tab to view the deployment status and logs
2. If the deployment is successful, you'll see a green checkmark
3. Click on the deployment to view detailed logs
4. Wait for the service to be fully deployed and running

## Step 7: Test Your Application

1. Once deployed, visit your application's domain
2. Upload an image to test the license plate recognition
3. Check if the app correctly detects and reads license plates

## Step 8: Understanding Railway's Free Tier

Railway offers a free tier with these resources:

1. $5 of free usage credits per month
2. 512 MB RAM and shared CPU
3. 1 GB of disk space
4. 5 projects per account

If you exceed these limits, your project will be paused until the next month or you can upgrade to a paid plan.

## Troubleshooting

If you encounter issues with your deployment:

1. **Build errors**: Check the build logs in the "Deployments" tab for specific error messages
2. **Application errors**: Check the runtime logs in the "Deployments" tab
3. **Missing dependencies**: Make sure all dependencies are correctly listed in your `requirements.txt` file
4. **Memory issues**: If your app crashes due to memory limits, consider optimizing your code or upgrading to a paid plan

## Advanced Configuration

For more advanced configurations:

1. **Custom build command**: You can specify a custom build command in a `railway.json` file:
   ```json
   {
     "build": {
       "builder": "NIXPACKS",
       "buildCommand": "pip install -r requirements.txt"
     },
     "deploy": {
       "startCommand": "gunicorn app_simplified:app",
       "restartPolicyType": "ON_FAILURE",
       "restartPolicyMaxRetries": 3
     }
   }
   ```

2. **Resource allocation**: You can adjust memory and CPU allocation in the project settings (paid plans only)

3. **Database integration**: Railway offers PostgreSQL, MySQL, and MongoDB as services you can attach to your app (each using your monthly credits)

## Tips for Optimizing Your Railway Deployment

1. **Minimize resource usage**: Keep your app's memory usage low to stay within the free tier limits
2. **Avoid unnecessary dependencies**: Only include required dependencies in your `requirements.txt`
3. **Use efficient image processing**: Consider resizing images before processing to reduce memory usage
4. **Implement caching**: Cache results to avoid redundant processing 