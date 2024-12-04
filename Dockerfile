FROM apache/spark

# Запуск от имени суперпользователя
USER root

# Обновление списка пакетов и установка pip
RUN apt-get update && apt-get install -y python3-pip

# Установка необходимых Python библиотек
RUN pip3 install requests

RUN pip install geohash2

# Копирование всех файлов в контейнер
COPY . /app

# Установка рабочей директории
WORKDIR /app

ENV PATH="/opt/spark/bin:${PATH}"

# Запуск Spark с вашим Python скриптом
CMD ["spark-submit", "etl_job.py"]