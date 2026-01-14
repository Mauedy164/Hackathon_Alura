# Usamos una imagen base ligera de Python 3.10
FROM python:3.10-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos el archivo de requerimientos primero (para aprovechar la caché de Docker)
COPY requirements.txt .

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto de los archivos (API.py y el modelo .pkl)
COPY . .

# Comando para arrancar la aplicación
# Render asigna un puerto dinámico en la variable de entorno $PORT
CMD sh -c "uvicorn API:app --host 0.0.0.0 --port ${PORT:-8000}"