import cv2
import mediapipe as mp
import math

# Inicializa la detección de la mano
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inicializa la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Inicializa una variable para el estado previo de la mano (inicialmente desconocido)
prev_hand_position_y = None
prev_hand_position_x = None
prev_fingers_together = False

while cap.isOpened():
    # Lee un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        continue

    # Convierte el frame a color (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta la mano en el frame
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Extrae las coordenadas y de los puntos 9 (dedo índice) y 12 (dedo medio)
            h, w, _ = frame.shape
            y9, y12 = int(landmarks.landmark[9].y * h),int(landmarks.landmark[12].y * h)

            x9, x12 = int(landmarks.landmark[9].x * w),int(landmarks.landmark[12].x * w)

            # Dedos juntos

            x8, y8 = int(landmarks.landmark[8].x * w), int(landmarks.landmark[8].y * h)
            x4, y4 = int(landmarks.landmark[4].x * w), int(landmarks.landmark[4].y * h)

            # Calcula la distancia entre los puntos 9 y 12
            distance = math.sqrt((x8 - x4) ** 2 + (y8 - y4) ** 2)

            # Verifica si la distancia está por debajo del umbral
            fingers_together = distance < 35

            # Imprime el estado solo si ha cambiado
            if prev_fingers_together != fingers_together:
                if fingers_together:
                    print("Dedos juntos")
                else:
                    print("Dedos separados")
                prev_fingers_together = fingers_together

            # Compara la posición vertical de los puntos para determinar el estado
            if y9 > y12:
                hand_position_y = "Sube"
            else:
                hand_position_y = "Baja"

            if (x9+1) > (x12+1):
               hand_position_x = "Derecha"
            else:
               hand_position_x = "Izquierda"

            # Imprime el estado solo si ha cambiado
            if prev_hand_position_y != hand_position_y:
                print(hand_position_y)
                prev_hand_position_y = hand_position_y
            
            if prev_hand_position_x != hand_position_x:
                print(hand_position_x)
                prev_hand_position_x = hand_position_x

            # Dibuja círculos en los puntos 9 y 12
            cv2.circle(frame, (int(landmarks.landmark[9].x * w), y9), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(landmarks.landmark[12].x * w), y12), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(landmarks.landmark[8].x * w), y8), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(landmarks.landmark[4].x * w), y4), 5, (0, 255, 0), -1)

    # Muestra el frame con los puntos de la mano
    cv2.imshow("Hand Tracking", frame)

    # Sale del bucle si se presiona la tecla 'esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera los recursos
cap.release()
cv2.destroyAllWindows()
