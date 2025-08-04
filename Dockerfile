# Temel Python imajı
FROM python:3.10-slim

# UTF-8 terminal desteği
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8

# Sistem bağımlılıklarını kur
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
 && apt-get clean

# Çalışma dizini
WORKDIR /app

# Gereken dosyaları kopyala
COPY . /app

# Bağımlılıkları yükle
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Flask portu
EXPOSE 5000

# Uygulamayı başlat
CMD ["python", "app.py"]
