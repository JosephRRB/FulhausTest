FROM python:3.10
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
ADD ./app app/
ADD ./core core/
ADD ./trained_model trained_model/
ADD ./validation_data validation_data/
ADD main_app.py ./main_app.py
CMD streamlit run main_app.py --server.port $PORT
