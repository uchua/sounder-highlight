FROM python:3.10-slim

# add source files
ADD src/ /opt/app/src/
ADD main.py /opt/app/
ADD requirements.txt .

# install/download dependencies
RUN pip install -r requirements.txt
RUN python -c 'import nltk; nltk.download("punkt")'

# configure container start
WORKDIR /opt/app
EXPOSE 8000
ENTRYPOINT ["uvicorn", "main:app", "--host=0.0.0.0" , "--port=8000"]
