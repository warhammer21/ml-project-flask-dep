FROM  continuumio/anaconda3:4.4.0
MAINTAINER UNP, https://unp.education
#WORKDIR /opt/rf_flask_project

COPY ./rf_flask_project /usr/local/python/
#ADD ./rf_flask_project /opt/rf_flask_project/

EXPOSE 5000
WORKDIR /usr/local/python/
RUN pip install -r requirements.txt
CMD python flask_api.py
