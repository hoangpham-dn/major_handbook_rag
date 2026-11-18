FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8501

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mh-rag"]

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
