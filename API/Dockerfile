# Usamos una imagen ligera de Python
FROM python:3.9-slim

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos los archivos de tu proyecto al contenedor
COPY . .

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Comando para iniciar la API
# Nota: Cloud Run espera que escuches en el puerto 8080 por defecto
CMD ["uvicorn", "API_individual:app", "--host", "0.0.0.0", "--port", "8080"]