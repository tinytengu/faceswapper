# Используем базовый образ alpine с установленным python
FROM python:3.12-alpine

# Устанавливаем необходимые системные зависимости
RUN apk add --no-cache build-base libffi-dev

# Устанавливаем рабочую директорию для приложения
WORKDIR /app

# Копируем все файлы приложения в рабочую директорию
COPY . /app

# Создаем папку /app/data/
RUN mkdir -p /app/data

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем зависимости из py-lmdb
RUN pip install --no-cache-dir -r ./py-lmdb/requirements.txt

# Устанавливаем зависимости из CodeFormer
RUN pip install --no-cache-dir -r ./CodeFormer/requirements.txt

# Указываем команду для запуска приложения
CMD ["python", "run.py"]
