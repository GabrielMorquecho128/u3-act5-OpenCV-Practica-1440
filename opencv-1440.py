import cv2
import numpy as np

# --- 1. Leer la imagen original ---
dino_rojo = cv2.imread('imagenes/dinorojo.jpg')

# AQUI AGREGAMOS TU NOMBRE Y NC A LA IMAGEN (Color Amarillo)
# Coordenadas (10, 30), Fuente, Tamaño 0.8, Color BGR (0, 255, 255), Grosor 2
cv2.putText(dino_rojo, 'Marco Morquecho 1440', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

cv2.imshow('1. Original _marco_morquecho_1440', dino_rojo) # Abre ventana con tu nombre

# --- 2. Convertir a escala de grises ---
gray_image = cv2.cvtColor(dino_rojo, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image_marco_morquecho_1440.jpg', gray_image)
cv2.imshow('2. Escala de grises _marco_morquecho_1440', gray_image)

# --- 3. Redimensionar la imagen a 300x200 ---
resized_image = cv2.resize(dino_rojo, (300, 200))
cv2.imwrite('resized_image_marco_morquecho_1440.jpg', resized_image)
cv2.imshow('3. Redimensionada _marco_morquecho_1440', resized_image)

# --- 4. Rotar la imagen 90 grados ---
(h, w) = dino_rojo.shape[:2]
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated_image = cv2.warpAffine(dino_rojo, matrix, (w, h))
cv2.imwrite('rotated_image_marco_morquecho_1440.jpg', rotated_image)
cv2.imshow('4. Rotada _marco_morquecho_1440', rotated_image)

# --- 5. Aplicar desenfoque gaussiano ---
blurred_image = cv2.GaussianBlur(dino_rojo, (15, 15), 0)
cv2.imwrite('blurred_image_marco_morquecho_1440.jpg', blurred_image)
cv2.imshow('5. Desenfoque _marco_morquecho_1440', blurred_image)

# --- 6. Detección de bordes (Algoritmo de Canny) ---
edges = cv2.Canny(gray_image, 100, 200)
cv2.imwrite('edges_marco_morquecho_1440.jpg', edges)
cv2.imshow('6. Bordes Canny _marco_morquecho_1440', edges)

# --- 7. Dibujar formas y añadir texto ---
image_shapes = dino_rojo.copy()
cv2.rectangle(image_shapes, (50, 50), (200, 200), (255, 0, 0), 3)
# (El tutorial pide agregar otro texto aquí, lo dejamos como extra)
cv2.putText(image_shapes, 'OpenCV', (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imwrite('output_image_marco_morquecho_1440.jpg', image_shapes)
cv2.imshow('7. Formas y texto _marco_morquecho_1440', image_shapes)

# --- 8. Umbralización binaria ---
_, thresh_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('threshold_image_marco_morquecho_1440.jpg', thresh_image)
cv2.imshow('8. Umbralizacion _marco_morquecho_1440', thresh_image)

# --- 9. Detección de contornos ---
image_contours = dino_rojo.copy()
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)
cv2.imwrite('contours_image_marco_morquecho_1440.jpg', image_contours)
cv2.imshow('9. Contornos _marco_morquecho_1440', image_contours)

# --- Esperar y cerrar ---
cv2.waitKey(0)
cv2.destroyAllWindows()