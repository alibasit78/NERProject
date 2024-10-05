FROM python:3.11

# RUN apt update -y &&  \
#     apt-get update &&  \
#     pip install --upgrade pip
WORKDIR /NERproject_container

COPY ./requirements.txt /NERproject_container/requirements.txt

COPY ./setup.py /NERproject_container/setup.py

RUN pip install -r /NERproject_container/requirements.txt

COPY . /NERproject_container/

# EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]