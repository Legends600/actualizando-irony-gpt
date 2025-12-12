import openai
import os
import sys
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de API
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("Error: OPENAI_API_KEY no está configurada")
    sys.exit(1)

# Colores ANSI
class Colores:
    ROJO = "\x1b[1;31m"
    AMARILLO = "\x1b[1;33m"
    VERDE = "\x1b[1;32m"
    RESET = "\x1b[0;37m"

# Prompt inicial
INITIAL_PROMPT = """Eres un analizador de sentimientos. Analiza el sentimiento de los mensajes que recibas.
Responde SOLO con un número entre -1 y 1:
- (-1): Negatividad máxima
- 0: Neutral
- 1: Positivo máximo
Puedes usar valores decimales (ej: -0.5, 0.3, etc.)"""

# Inicializar historial de mensajes
messages = [
    {"role": "system", "content": INITIAL_PROMPT}
]

class AnalizadorDeSentimientos:
    """Analiza la polaridad del sentimiento y retorna una etiqueta formateada"""
    
    def analizar_sentimiento(self, polaridad):
        """
        Convierte una polaridad numérica a una etiqueta con color.
        
        Args:
            polaridad (float): Valor entre -1 y 1
            
        Returns:
            str: Etiqueta con color ANSI
        """
        # Definir rangos de sentimiento
        rangos = [
            ((-1, -0.6), "muy negativo", Colores.ROJO),
            ((-0.6, -0.3), "negativo", Colores.ROJO),
            ((-0.3, 0), "algo negativo", Colores.ROJO),
            ((0, 0), "neutral", Colores.AMARILLO),
            ((0, 0.3), "algo positivo", Colores.AMARILLO),
            ((0.3, 0.6), "positivo", Colores.VERDE),
            ((0.6, 0.9), "muy positivo", Colores.VERDE),
            ((0.9, 1), "extremadamente positivo", Colores.VERDE),
        ]
        
        for (min_val, max_val), etiqueta, color in rangos:
            if min_val < polaridad <= max_val:
                return f"{color}{etiqueta}{Colores.RESET}"
        
        # Por si acaso
        if polaridad == -1:
            return f"{Colores.ROJO}muy negativo{Colores.RESET}"
        
        return f"{Colores.ROJO}Error: polaridad inválida{Colores.RESET}"


analizador = AnalizadorDeSentimientos()


def obtener_sentimiento_de_openai(user_input):
    """
    Envía un mensaje a OpenAI y obtiene el análisis de sentimiento.
    
    Args:
        user_input (str): El mensaje del usuario
        
    Returns:
        float: Polaridad del sentimiento (-1 a 1) o None si hay error
    """
    messages.append({"role": "user", "content": user_input})
    
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=10,  # Solo necesitamos un número
            temperature=0
        )
        
        respuesta = completion.choices[0].message['content'].strip()
        
        # Agregar respuesta al historial
        messages.append({
            "role": "assistant",
            "content": respuesta
        })
        
        # Convertir a float con validación
        sentimiento = float(respuesta)
        
        # Validar que esté en rango
        if -1 <= sentimiento <= 1:
            return sentimiento
        else:
            print(f"{Colores.ROJO}Error: Valor fuera de rango [-1, 1]{Colores.RESET}")
            return None
            
    except ValueError:
        print(f"{Colores.ROJO}Error: Respuesta no numérica de OpenAI{Colores.RESET}")
        return None
    except openai.error.OpenAIError as e:
        print(f"{Colores.ROJO}Error de OpenAI: {str(e)}{Colores.RESET}")
        return None


# Loop principal
def main():
    """Loop principal de la aplicación"""
    print(f"{Colores.AMARILLO}=== Analizador de Sentimientos con GPT ==={Colores.RESET}\n")
    
    while True:
        try:
            user_input = input(f"{Colores.AMARILLO}Decime algo: {Colores.RESET}").strip()
            
            if not user_input:
                print(f"{Colores.ROJO}Por favor ingresa algo{Colores.RESET}")
                continue
            
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print(f"{Colores.AMARILLO}¡Hasta luego!{Colores.RESET}")
                break
            
            sentimiento_valor = obtener_sentimiento_de_openai(user_input)
            
            if sentimiento_valor is not None:
                sentimiento_texto = analizador.analizar_sentimiento(sentimiento_valor)
                print(f"Sentimiento: {sentimiento_texto} ({sentimiento_valor})\n")
        
        except KeyboardInterrupt:
            print(f"\n{Colores.AMARILLO}Programa interrumpido{Colores.RESET}")
            break
        except Exception as e:
            print(f"{Colores.ROJO}Error inesperado: {str(e)}{Colores.RESET}")


if __name__ == "__main__":
    main()    
