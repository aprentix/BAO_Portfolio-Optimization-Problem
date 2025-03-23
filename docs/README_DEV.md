# Virtual Environment Management

## Prerequisites

`pip install virtualenv`

## Create Virtual Environment

`python3 -m venv env`

## Activate the Virtual Environment
`source env/bin/activate`

## Save and Load Dependencies with `requirements.txt`

### Save
`pip freeze > requirements.txt`

### Load
`pip install -r requirements.txt`

## Deactivate the Virtual Environment

`deactivate`

## Delete a Virtual Environment

`rm -rf env`