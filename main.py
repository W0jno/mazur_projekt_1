import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

# =====================================================================
# 1. GENERACJA I AUGMENTACJA DANYCH
# =====================================================================

def draw_base_emoticon(emoticon_type):
    """
    Rysuje bazowy symbol 32x32 w skali szarości (0-255).
    Zachowuje spójny, monochromatyczny wariant (jasne linie na ciemnym tle).
    """
    img = Image.new('L', (32, 32), color=0)
    draw = ImageDraw.Draw(img)
    
    if emoticon_type == 0:    # ☺ Uśmiech
        draw.ellipse([3, 3, 28, 28], outline=255, width=2)
        draw.rectangle([10, 10, 11, 12], fill=255)
        draw.rectangle([20, 10, 21, 12], fill=255)
        draw.arc([10, 14, 21, 23], start=20, end=160, fill=255, width=2)
        
    elif emoticon_type == 1:  # ☹ Smutek
        draw.ellipse([3, 3, 28, 28], outline=255, width=2)
        draw.rectangle([10, 11, 11, 13], fill=255)
        draw.rectangle([20, 11, 21, 13], fill=255)
        draw.arc([10, 17, 21, 25], start=200, end=340, fill=255, width=2)
        
    elif emoticon_type == 2:  # ⚇ Zdziwienie / Neutralny
        draw.ellipse([3, 3, 28, 28], outline=255, width=2)
        draw.rectangle([10, 10, 11, 12], fill=255)
        draw.rectangle([20, 10, 21, 12], fill=255)
        draw.ellipse([13, 18, 18, 23], outline=255, width=2)
        
    elif emoticon_type == 3:  # ☼ Słońce
        draw.ellipse([9, 9, 22, 22], outline=255, width=2)
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            x1, y1 = 15.5 + 8 * math.cos(rad), 15.5 + 8 * math.sin(rad)
            x2, y2 = 15.5 + 13 * math.cos(rad), 15.5 + 13 * math.sin(rad)
            draw.line([x1, y1, x2, y2], fill=255, width=2)
            
    elif emoticon_type == 4:  # ☽ Księżyc rosnący
        draw.ellipse([5, 3, 26, 24], fill=255)
        draw.ellipse([2, 3, 21, 24], fill=0)
        
    elif emoticon_type == 5:  # ☾ Księżyc malejący
        draw.ellipse([5, 3, 26, 24], fill=255)
        draw.ellipse([10, 3, 29, 24], fill=0)
        
    elif emoticon_type == 6:  # ☯ Yin-Yang
        draw.ellipse([4, 4, 27, 27], outline=255, width=2)
        draw.chord([4, 4, 27, 27], start=90, end=270, fill=255)
        draw.ellipse([10, 4, 21, 15.5], fill=255)
        draw.ellipse([10, 15.5, 21, 27], fill=0)
        draw.ellipse([14.5, 8.5, 16.5, 10.5], fill=0)
        draw.ellipse([14.5, 20.5, 16.5, 22.5], fill=255)
        
    elif emoticon_type == 7:  # ☮ Pacyfka
        draw.ellipse([4, 4, 27, 27], outline=255, width=2)
        draw.line([15.5, 4, 15.5, 27], fill=255, width=2)
        draw.line([15.5, 15.5, 6, 24], fill=255, width=2)
        draw.line([15.5, 15.5, 25, 24], fill=255, width=2)
        
    return img

def augment_image(img):
    """
    Dodaje zakłócenia, przesunięcia, rozmycia oraz pochylenia/obroty.
    """
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    
    dx = random.randint(-3, 3)
    dy = random.randint(-3, 3)
    arr = np.array(img)
    arr = np.roll(arr, shift=(dy, dx), axis=(0, 1))
    
    if dy > 0: arr[:dy, :] = 0
    elif dy < 0: arr[dy:, :] = 0
    if dx > 0: arr[:, :dx] = 0
    elif dx < 0: arr[:, dx:] = 0
    img = Image.fromarray(arr)
    
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
        
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(loc=0, scale=15, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    
    return arr

def generate_dataset():
    """
    Generuje zbiór danych dla 8 klas, po 100 próbek na klasę.
    """
    X, y = [], []
    for class_idx in range(8):
        base_img = draw_base_emoticon(class_idx)
        for _ in range(100):
            aug_arr = augment_image(base_img)
            X.append(aug_arr)
            y.append(class_idx)
            
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int64)

# =====================================================================
# 2. ZAPIS ZBIORÓW DANYCH DO PLIKÓW PDF (WYMÓG SPECYFIKACJI)
# =====================================================================

def save_dataset_to_pdf(X, y, filename, title_text, class_names):
    """
    Zapisuje podsumowanie oraz zawartość zbioru danych (macierze obrazów) 
    do wielostronicowego pliku PDF.
    """
    print(f"Generowanie pliku danych: {filename}...")
    with PdfPages(filename) as pdf:
        # Strona tytułowa i statystyki
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.text(0.5, 0.9, title_text, fontsize=16, fontweight='bold', ha='center')
        ax.text(0.5, 0.7, f"Całkowita liczba próbek: {len(X)}", fontsize=12, ha='center')
        ax.text(0.5, 0.6, "Format: 32x32 piksele, skala szarości (0-255)", fontsize=12, ha='center')
        
        # Podsumowanie liczby próbek na klasę
        unique, counts = np.unique(y, return_counts=True)
        stats_text = "Liczba próbek w klasach:\n" + "\n".join(
            [f" - Klasa {cls} ({class_names[cls]}): {cnt} szt." for cls, cnt in zip(unique, counts)]
        )
        ax.text(0.1, 0.3, stats_text, fontsize=10, va='top', fontname='DejaVu Sans')
        pdf.savefig(fig)
        plt.close()

        # Zapis podglądu próbek w siatkach (np. 4x4 próbki na stronę)
        samples_per_page = 16
        num_pages = math.ceil(len(X) / samples_per_page)
        
        for page in range(num_pages):
            fig, axes = plt.subplots(4, 4, figsize=(10, 10))
            fig.suptitle(f"{title_text} - Strona {page+1}/{num_pages}", fontsize=12)
            
            start_idx = page * samples_per_page
            for i, ax in enumerate(axes.flat):
                cur_idx = start_idx + i
                if cur_idx < len(X):
                    ax.imshow(X[cur_idx], cmap='gray', vmin=0, vmax=255)
                    ax.set_title(f"ID: {cur_idx} | Klasa: {y[cur_idx]}", fontsize=9)
                ax.axis('off')
                
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            
    print(f"Zapisano pomyślnie: {filename}")

# =====================================================================
# 3. ARCHITEKTURA LeNet-5
# =====================================================================

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

# =====================================================================
# 4. GŁÓWNA LOGIKA PROGRAMU
# =====================================================================

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Generacja i podział zbioru danych
    X_all, y_all = generate_dataset()
    num_classes = len(np.unique(y_all)) # 8 klas
    class_names = ['☺', '☹', '⚇', '☼', '☽', '☾', '☯', '☮']
    
    X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []
    for cls in range(num_classes):
        cls_mask = (y_all == cls)
        X_cls = X_all[cls_mask]
        y_cls = y_all[cls_mask]
        
        indices = np.random.permutation(len(X_cls))
        train_idx, test_idx = indices[:50], indices[50:]
        
        X_train_list.append(X_cls[train_idx])
        y_train_list.append(y_cls[train_idx])
        X_test_list.append(X_cls[test_idx])
        y_test_list.append(y_cls[test_idx])
        
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # --- REALIZACJA WYMOGÓW PDF ---
    
    # 1. Obraz wejściowy jako jedna strona PDF
    with PdfPages("obrazy_wejsciowe.pdf") as pdf:
        fig, axes = plt.subplots(4, 8, figsize=(12, 6))
        fig.suptitle("Weryfikacja bazy - losowe próbki wejściowe (0-255)", fontsize=14)
        sample_indices = np.random.choice(len(X_all), size=32, replace=False)
        for ax, idx in zip(axes.flat, sample_indices):
            ax.imshow(X_all[idx], cmap='gray', vmin=0, vmax=255)
            ax.set_title(f"{class_names[y_all[idx]]} ({y_all[idx]})", fontname='DejaVu Sans')
            ax.axis('off')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    print("Zapisano plik: obrazy_wejsciowe.pdf")

    # 2. Pliki do uczenia i testowania w formacie PDF
    save_dataset_to_pdf(X_train, y_train, "dane_do_uczenia.pdf", "Zbiór Uczący (Train Set)", class_names)
    save_dataset_to_pdf(X_test, y_test, "dane_do_testowania.pdf", "Zbiór Testowy (Test Set)", class_names)

    # Przygotowanie tensorów
    t_X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    t_y_train = torch.tensor(y_train, dtype=torch.int64)
    t_X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    t_y_test = torch.tensor(y_test, dtype=torch.int64)

    train_loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=16, shuffle=True)
    
    model = LeNet5(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # Trening
    epochs = 40
    train_losses, train_accs = [], []
    
    print(f"\nRozpoczęcie trenowania modelu LeNet-5 na {num_classes} klasach...")
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoka [{epoch+1}/{epochs}] | Strata: {epoch_loss:.4f} | Dokładność: {epoch_acc*100:.1f}%")

    # 3. Krzywe uczenia jako PDF
    with PdfPages("krzywe_uczenia.pdf") as pdf:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_xlabel('Epoka')
        ax1.set_ylabel('Strata (Loss)', color='tab:red')
        ax1.plot(range(1, epochs+1), train_losses, color='tab:red', label='Strata')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        ax2 = ax1.twinx()  
        ax2.set_ylabel('Dokładność (Accuracy)', color='tab:blue')
        ax2.plot(range(1, epochs+1), train_accs, color='tab:blue', label='Dokładność')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        plt.title('Krzywe uczenia LeNet-5 (8 klas)')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close()
    print("Zapisano plik: krzywe_uczenia.pdf")

    # Ewaluacja
    model.eval()
    with torch.no_grad():
        test_outputs = model(t_X_test)
        _, test_preds = test_outputs.max(1)
        test_acc = test_preds.eq(t_y_test).sum().item() / t_y_test.size(0)
        cm = confusion_matrix(y_test, test_preds.numpy())

    print(f"\n--- WYNIKI NA ZBIORZE TESTOWYM ---")
    print(f"Dokładność: {test_acc*100:.2f}%")
    print("Macierz pomyłek:")
    print(cm)

    # 4. Wyniki klasyfikacji (losowe obrazki z bazy i z szumem)
    indices_eval = np.random.choice(len(X_all), size=20, replace=False)
    samples_normal = X_all[indices_eval]
    labels_true = y_all[indices_eval]
    
    samples_noisy = []
    for img_arr in samples_normal:
        noise = np.random.normal(0, 45, img_arr.shape)
        noisy_arr = np.clip(img_arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        samples_noisy.append(noisy_arr)
    samples_noisy = np.array(samples_noisy)

    t_normal = torch.tensor(samples_normal, dtype=torch.float32).unsqueeze(1)
    t_noisy = torch.tensor(samples_noisy, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        preds_normal = model(t_normal).max(1)[1].numpy()
        preds_noisy = model(t_noisy).max(1)[1].numpy()

    with PdfPages("wyniki_klasyfikacji.pdf") as pdf:
        fig, axes = plt.subplots(4, 5, figsize=(10, 8))
        fig.suptitle("Klasyfikacja - 20 losowych przypadków z bazy", fontsize=14)
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples_normal[i], cmap='gray', vmin=0, vmax=255)
            color = 'green' if preds_normal[i] == labels_true[i] else 'red'
            ax.set_title(f"P:{preds_normal[i]} R:{labels_true[i]}", color=color)
            ax.axis('off')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        fig, axes = plt.subplots(4, 5, figsize=(10, 8))
        fig.suptitle("Klasyfikacja - 20 przypadków z dodanym szumem", fontsize=14)
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples_noisy[i], cmap='gray', vmin=0, vmax=255)
            color = 'green' if preds_noisy[i] == labels_true[i] else 'red'
            ax.set_title(f"P:{preds_noisy[i]} R:{labels_true[i]}", color=color)
            ax.axis('off')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print("Zapisano plik: wyniki_klasyfikacji.pdf")
    
    # Podsumowanie wygenerowanych artefaktów
    print("\nProjekt kompletny. Spis wygenerowanych plików PDF:")
    print(" 1. obrazy_wejsciowe.pdf    - Jedna strona z przeglądem klas")
    print(" 2. dane_do_uczenia.pdf     - Zbiór treningowy (wizualizacja próbek i metadane)")
    print(" 3. dane_do_testowania.pdf  - Zbiór testowy (wizualizacja próbek i metadane)")
    print(" 4. krzywe_uczenia.pdf      - Wykresy loss i accuracy z procesu treningu")
    print(" 5. wyniki_klasyfikacji.pdf - Predykcje dla 20 obrazów czystych i 20 zaszumionych")

if __name__ == "__main__":
    main()