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
CLASSES = ['G', 'D', 'L', 'P', 'LG', 'PG', 'LD', 'PD'] 
CHARS = ['↑', '↓', '←', '→', '↖', '↗', '↙', '↘'] 
NUM_CLASSES = len(CLASSES)
SAMPLES_PER_CLASS = 100
IMG_SIZE = 32

FONT_PATH = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf' # Ścieżka do czcionki

# ==========================================
# 2. GENEROWANIE DANYCH
# ==========================================
def generate_base_image(char, font_path, size=32):
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        print(f"BŁĄD: Nie znaleziono czcionki {font_path}.")
        exit()
        
    bbox = draw.textbbox((0,0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (size - w) / 2
    y = (size - h) / 2 - bbox[1]
    
    draw.text((x, y), char, fill=0, font=font)
    return img

def augment_image(img, force_noise=False):
    angle = random.uniform(-15, 15)
    dx = random.randint(-3, 3)
    dy = random.randint(-3, 3)
    img_aug = img.rotate(angle, fillcolor=255, translate=(dx, dy))
    
    if random.random() > 0.5:
        img_aug = img_aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
    
    arr = np.array(img_aug)
    
    if force_noise or random.random() > 0.7:
        noise = np.random.normal(0, 25, arr.shape) 
        arr = arr + noise
        arr = np.clip(arr, 0, 255)
        
    return arr.astype(np.uint8)

print("Generowanie zbioru danych...")
X_data, y_data = [], []
clean_samples_eval, clean_labels_eval = [], []
noisy_samples_eval, noisy_labels_eval = [], []

for label, char in enumerate(CHARS):
    base_img = generate_base_image(char, FONT_PATH)
    
    for i in range(SAMPLES_PER_CLASS):
        is_noisy = (i % 5 == 0) 
        aug_arr = augment_image(base_img, force_noise=is_noisy)
        
        X_data.append(aug_arr)
        y_data.append(label)
        
        if not is_noisy and len(clean_samples_eval) < 20:
            clean_samples_eval.append(aug_arr)
            clean_labels_eval.append(label)
        elif is_noisy and len(noisy_samples_eval) < 20:
            noisy_samples_eval.append(aug_arr)
            noisy_labels_eval.append(label)

X_data = np.array(X_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_data = np.array(y_data)

# --- WYMÓG 1: Obraz wejściowy w formacie graficznym jako jedna strona PDF ---
fig, axes = plt.subplots(4, 5, figsize=(8, 10))
fig.suptitle("Obraz wejściowy - przykladowe wygenerowane znaki", fontsize=14)
for i, ax in enumerate(axes.flat):
    if i < len(X_data):
        ax.imshow(X_data[i].reshape(32, 32), cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
plt.savefig('obraz_wejsciowy.pdf')
plt.close()
print("Zapisano: obraz_wejsciowy.pdf")

# --- WYMÓG 2: Pliki do uczenia/testowania ---
np.save('dane_wejsciowe_X.npy', X_data)
np.save('etykiety_y.npy', y_data)
print("Zapisano pliki bazy danych: dane_wejsciowe_X.npy, etykiety_y.npy")

# Normalizacja i podział danych
X_data_norm = X_data / 255.0
y_data_cat = to_categorical(y_data, NUM_CLASSES)
X_train, X_test, y_train, y_test = train_test_split(X_data_norm, y_data_cat, test_size=0.5, random_state=42)

# ==========================================
# 3. BUDOWA I TRENING SIECI
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

print("Rozpoczynam uczenie modelu...")
history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# --- WYMÓG 3: Krzywe uczenia (PDF) ---
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Dokladnosc (Trening)')
plt.plot(history.history['val_accuracy'], label='Dokladnosc (Test)')
plt.title('Krzywe uczenia')
plt.xlabel('Epoka')
plt.ylabel('Dokladnosc')
plt.legend()
plt.grid()
plt.savefig('krzywe_uczenia.pdf')
plt.close()
print("Zapisano: krzywe_uczenia.pdf")

# ==========================================
# 4. WYNIKI KLASYFIKACJI NA PDF (Czyste vs Szum)
# ==========================================
def generate_results_pdf(samples, true_labels, filename, title):
    samples_norm = np.array(samples).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    preds = np.argmax(model.predict(samples_norm, verbose=0), axis=1)
    
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            ax.imshow(samples[i].reshape(32, 32), cmap='gray', vmin=0, vmax=255)
            true_c = CLASSES[true_labels[i]]
            pred_c = CLASSES[preds[i]]
            color = 'green' if true_c == pred_c else 'red'
            ax.set_title(f"Prawda: {true_c}\nSiec: {pred_c}", color=color, fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wyniki klasyfikacji: {filename}")

generate_results_pdf(clean_samples_eval, clean_labels_eval, 'wyniki_klasyfikacji_czyste.pdf', 'Wyniki klasyfikacji - 20 przypadkow bazowych')
generate_results_pdf(noisy_samples_eval, noisy_labels_eval, 'wyniki_klasyfikacji_szum.pdf', 'Wyniki klasyfikacji - 20 przypadkow zaszumionych')

print("\nKoniec procesu uczenia i zapisu PDF!")

# ==========================================
# 5. TESTOWE WCZYTANIE I WYŚWIETLENIE Z PLIKÓW .NPY
# ==========================================
print("\nWczytywanie zapisanych danych z plików .npy w celu weryfikacji...")
loaded_X = np.load('dane_wejsciowe_X.npy')
loaded_y = np.load('etykiety_y.npy')

print(f"Pomyślnie wczytano tablice o kształtach: X={loaded_X.shape}, y={loaded_y.shape}")

# Otwieramy okienko i wyświetlamy 5 losowych obrazków z wczytanego pliku
plt.figure(figsize=(12, 3))
plt.suptitle("Weryfikacja: Wczytano obrazki bezposrednio z pliku .npy", fontsize=12)

for i in range(5):
    # Losujemy indeks obrazka od 0 do 799
    idx = random.randint(0, len(loaded_X) - 1)
    
    ax = plt.subplot(1, 5, i + 1)
    ax.imshow(loaded_X[idx].reshape(32, 32), cmap='gray', vmin=0, vmax=255)
    ax.set_title(f"Klasa wczytana: {CLASSES[loaded_y[idx]]}", fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('weryfikacja_npy.pdf')
plt.close()
print("Zapisano testowy odczyt z pliku .npy jako: weryfikacja_npy.pdf")