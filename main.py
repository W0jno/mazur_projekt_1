import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# ==========================================
# 1. PARAMETRY I USTAWIENIA
# ==========================================
# Strzałki kardynalne (8 klas)
CLASSES = ['↑', '↓', '←', '→', '↖', '↗', '↙', '↘']
NUM_CLASSES = len(CLASSES)
SAMPLES_PER_CLASS = 100
IMG_SIZE = 32

# UWAGA: Podaj ścieżkę do czcionki w systemie, która obsługuje te znaki (np. Arial).
# Windows: 'arial.ttf', Mac: '/Library/Fonts/Arial Unicode.ttf', Linux: '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
FONT_PATH = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf' 

# ==========================================
# 2. GENEROWANIE DANYCH (Augmentacja)
# ==========================================
def generate_base_image(char, font_path, size=32):
    """Generuje czysty obrazek ze znakiem."""
    img = Image.new('L', (size, size), color=255) # 'L' to skala szarości, tło białe (255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        print(f"BŁĄD: Nie znaleziono czcionki {font_path}. Podaj poprawną ścieżkę!")
        exit()
        
    # Wyśrodkowanie tekstu
    bbox = draw.textbbox((0,0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (size - w) / 2
    y = (size - h) / 2 - bbox[1]
    
    draw.text((x, y), char, fill=0, font=font) # Czarny znak (0)
    return img

def augment_image(img, is_noisy=False):
    """Dodaje losowe zniekształcenia (przesunięcie, obrót, szum)."""
    # Losowy obrót i przesunięcie
    angle = random.uniform(-15, 15)
    dx = random.randint(-3, 3)
    dy = random.randint(-3, 3)
    
    img_aug = img.rotate(angle, fillcolor=255, translate=(dx, dy))
    
    # Rozmycie
    if random.random() > 0.5:
        img_aug = img_aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
    
    # Zmiana na tablicę numpy
    arr = np.array(img_aug)
    
    # Dodanie szumu (jeśli wymagane lub wylosowane)
    if is_noisy or random.random() > 0.7:
        noise = np.random.normal(0, 15, arr.shape)
        arr = arr + noise
        arr = np.clip(arr, 0, 255) # Utrzymanie wartości w zakresie 0-255
        
    return arr.astype(np.uint8)

print("Generowanie zbioru danych...")
X_data = []
y_data = []

# Dodatkowe listy do testu końcowego (20 czystych, 20 zaszumionych)
clean_samples = []
noisy_samples = []
clean_labels = []
noisy_labels = []

for label, char in enumerate(CLASSES):
    base_img = generate_base_image(char, FONT_PATH)
    
    for i in range(SAMPLES_PER_CLASS):
        # 80% to lekko zmodyfikowane obrazki, 20% to mocno zaszumione
        is_noisy = (i % 5 == 0) 
        aug_arr = augment_image(base_img, is_noisy)
        
        X_data.append(aug_arr)
        y_data.append(label)
        
        # Zbieramy próbki do końcowego raportu (max 20)
        if len(clean_samples) < 20 and not is_noisy:
            clean_samples.append(aug_arr)
            clean_labels.append(label)
        elif len(noisy_samples) < 20 and is_noisy:
            noisy_samples.append(aug_arr)
            noisy_labels.append(label)

X_data = np.array(X_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # Dodajemy kanał koloru (1)
y_data = np.array(y_data)

# Zapisanie próbek obrazu wejściowego do PDF (wymóg z zadania)
fig, axes = plt.subplots(4, 5, figsize=(10, 8))
fig.suptitle("Przykładowe wygenerowane obrazy wejściowe")
for i, ax in enumerate(axes.flat):
    ax.imshow(X_data[i].reshape(32, 32), cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
plt.savefig('obraz_wejsciowy.pdf')
plt.close()

# Normalizacja danych (0-1) dla sieci neuronowej
X_data = X_data / 255.0
y_data_cat = to_categorical(y_data, NUM_CLASSES)

# Podział 50% nauka, 50% testowanie (losowo)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_cat, test_size=0.5, random_state=42)

print(f"Wygenerowano {len(X_data)} obrazków. Trening: {len(X_train)}, Test: {len(X_test)}")

# ==========================================
# 3. ARCHITEKTURA SIECI LeNet-5
# ==========================================
model = Sequential([
    Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    AveragePooling2D(pool_size=(2, 2)),
    Conv2D(16, kernel_size=(5, 5), activation='relu'),
    AveragePooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 4. TRENING I KRZYWE UCZENIA
# ==========================================
print("Rozpoczynam uczenie modelu...")
# Zgodnie z poleceniem walidacji z osobnego zbioru nie robimy, używamy testowego do pokazania krzywej
history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Zapis krzywej uczenia do PDF
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Dokładność (Trening)')
plt.plot(history.history['val_accuracy'], label='Dokładność (Test)')
plt.title('Krzywe uczenia - Dokładność')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.grid()
plt.savefig('krzywe_uczenia.pdf')
plt.close()

# ==========================================
# 5. TESTOWANIE WYNIKÓW (Czyste vs Zaszumione)
# ==========================================
def evaluate_samples(samples, true_labels, title):
    samples_normalized = np.array(samples).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    predictions = model.predict(samples_normalized, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    correct = sum(1 for p, t in zip(predicted_classes, true_labels) if p == t)
    print(f"\n--- {title} ---")
    print(f"Poprawnie sklasyfikowano: {correct} / {len(samples)}")
    
    # Wyświetlenie kilku pierwszych wyników
    for i in range(min(5, len(samples))):
        print(f"Prawdziwa: {CLASSES[true_labels[i]]} | Sieć zgadła: {CLASSES[predicted_classes[i]]}")

evaluate_samples(clean_samples, clean_labels, "WYNIKI DLA 20 CZYSTYCH OBRAZÓW")
evaluate_samples(noisy_samples, noisy_labels, "WYNIKI DLA 20 ZASZUMIONYCH OBRAZÓW")

print("\nGotowe! Wygenerowano pliki: 'obraz_wejsciowy.pdf' oraz 'krzywe_uczenia.pdf'.")