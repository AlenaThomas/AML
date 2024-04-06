FROM python:3.9-slim

# setting working directory
WORKDIR /app

# setting up python environment
COPY requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copying file contents
COPY . .

# making port 5000 available for the app
EXPOSE 5000

# running the app on container launch
CMD ["python", "app.py"]