#Instalar las siguientes librerias desde terminal, ya sea por entorno virtual, que es lo recomandado o de manera global, pero puede dar errores con otros proyectos

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

class DatosCliente(BaseModel):
    customer_tenure: float
    account_charges_monthly: float
    cliente_nuevo: int
    contrato_mensual: int
    alto_costo: int


try: 
    modelo = joblib.load("telecomx_churn_model.pkl")
    print("Modelo cargado")
except:
    modelo = None
    print("El modelo no se ha encontrado")

app = FastAPI()

@app.post("/baja-cliente")
def prediccion_cliente(datos:DatosCliente):
    
    if modelo is None:
        return {"Error" : "El modelo no está diponible"}

    datos_para_modelo = [[
        datos.customer_tenure,
        datos.account_charges_monthly,
        datos.cliente_nuevo,
        datos.contrato_mensual,
        datos.alto_costo
    ]]
    
    resultado_cancelar = modelo.predict(datos_para_modelo)[0] #Aqui me arroja una lista, por eso extraigo el primer valor de la lista (1 o 0, es decir, "va a cancelar" o "no va a cancelar")

    resultado_probabilidad = modelo.predict_proba(datos_para_modelo)[0] #Aqui me dará una lista, la cual está formada por más listas con la probabilidad de que sea 0 y de que sea 1
    probabilidad_churn = resultado_probabilidad[1] #Aquí elijo el dato que corresponde a la probabilidad de que sea 1


    mensaje_prevision = ""
    if resultado_cancelar == 1:
        mensaje_prevision = "Va a cancelar"
    else:
        mensaje_prevision = "No va a cancelar"

    return {
        "prevision": mensaje_prevision,
        "probabilidad_de_churn": float(round(probabilidad_churn,2))
    }

@app.post("/prediccion-masiva")
def prediccion_masiva(lista_clientes:list[DatosCliente]):
    
    if modelo is None:
        return {"Error" : "El modelo no está diponible"}

    datos_para_modelo = []

    for cliente in lista_clientes:
        fila = [
            cliente.customer_tenure,
            cliente.account_charges_monthly,
            cliente.cliente_nuevo,
            cliente.contrato_mensual,
            cliente.alto_costo
        ]
        datos_para_modelo.append(fila)
    
    predicciones = modelo.predict(datos_para_modelo)
    probabilidades = modelo.predict_proba(datos_para_modelo)

    respuesta = []

    for cliente, resultado, probabilidad in zip(lista_clientes, predicciones, probabilidades):
        if resultado == 1 :
            mensaje_prevision = "Va a cancelar"
        else:
            mensaje_prevision = "No va a cancelar"

        respuesta.append({
            "datos_cliente": cliente,
            "estado": mensaje_prevision,
            "probabilidad_de_churn" : float(round(probabilidad[1], 2))
        })
    return respuesta
