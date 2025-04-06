# Usar una imagen base con Python
FROM python:3.9

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivos del proyecto
COPY . /app

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 8000 para FastAPI
EXPOSE 8000

# Comando de ejecución de la API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
