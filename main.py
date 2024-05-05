import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import pandas as pd

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ÖDEV 1")

        # Bilgilerim ve Ana Sayfa
        self.label = QLabel("Dersin Adı: Dijital Görüntü İşleme\nÖğrenci Numarası: 211229006\nAdı: Muhammed Arif ACAY")
        self.label.setAlignment(Qt.AlignCenter)

        # Küçük Ödev Menüsü
        menubar = self.menuBar()
        menu = menubar.addMenu("Ödevler")

        # Ödev 1
        action_odev1 = QAction("Ödev 1: Kontrast Güçlendirme", self)
        action_odev1.triggered.connect(self.odev1)
        menu.addAction(action_odev1)

        # Ödev 2: Çizgi Tespiti
        action_odev2 = QAction("Ödev 2: Çizgi Tespiti", self)
        action_odev2.triggered.connect(self.odev2)
        menu.addAction(action_odev2)

        # Ödev 2: Göz Tespiti
        action_odev3 = QAction("Ödev 2: Göz Tespiti", self)
        action_odev3.triggered.connect(self.odev3)
        menu.addAction(action_odev3)

        # Ödev 3: Deblurring
        action_odev4 = QAction("Ödev 3: Deblurring", self)
        action_odev4.triggered.connect(self.deblur_image)
        menu.addAction(action_odev4)

        # Ödev 4: Yeşil Alan Analizi
        action_odev5 = QAction("Ödev 4: Yeşil Alan Analizi", self)
        action_odev5.triggered.connect(self.detect_and_analyze_green_areas)
        menu.addAction(action_odev5)

        # Ödev1 Detayları
        self.odev1_label = QLabel("Ödev 1: Kontrast Güçlendirme")
        self.odev1_button = QPushButton("Görüntü Yükle ve Kontrastı Güçlendir")
        self.odev1_button.clicked.connect(self.apply_contrast_enhancement)
        self.odev1_result_label = QLabel()

        # Ödev2 Detayları
        self.odev2_label = QLabel("Ödev 2: Çizgi Tespiti")
        self.odev2_button = QPushButton("Görüntü Yükle ve Çizgileri Tespit Et")
        self.odev2_button.clicked.connect(self.detect_lines)
        self.odev2_result_label = QLabel()

        # Ödev3 Detayları
        self.odev3_label = QLabel("Ödev 2: Göz Tespiti")
        self.odev3_button = QPushButton("Görüntü Yükle ve Gözleri Tespit Et")
        self.odev3_button.clicked.connect(self.detect_eyes)
        self.odev3_result_label = QLabel()

        # Ödev4 Detayları
        self.odev4_label = QLabel("Ödev 3: Deblurring")
        self.odev4_button = QPushButton("Görüntü Yükle ve Deblur Yap")
        self.odev4_button.clicked.connect(self.deblur_image)
        self.odev4_result_label = QLabel()

        self.odev5_label = QLabel("Ödev 4: Yeşil Alan Analizi")
        self.odev5_button = QPushButton("Görüntü Yükle ve Yeşil Alanları Analiz Et")
        self.odev5_button.clicked.connect(self.detect_and_analyze_green_areas)
        self.odev5_result_label = QLabel()


        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.odev1_label)
        layout.addWidget(self.odev1_button)
        layout.addWidget(self.odev1_result_label)
        layout.addWidget(self.odev2_label)
        layout.addWidget(self.odev2_button)
        layout.addWidget(self.odev2_result_label)
        layout.addWidget(self.odev3_label)
        layout.addWidget(self.odev3_button)
        layout.addWidget(self.odev3_result_label)
        layout.addWidget(self.odev4_label)
        layout.addWidget(self.odev4_button)
        layout.addWidget(self.odev4_result_label)
        layout.addWidget(self.odev5_label)
        layout.addWidget(self.odev5_button)
        layout.addWidget(self.odev5_result_label)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def odev1(self):
        self.odev1_label.setText("Ödev 1: Kontrast Güçlendirme")
        self.odev1_button.setText("Görüntü Yükle ve Kontrastı Güçlendir")
        self.odev1_button.clicked.disconnect()
        self.odev1_button.clicked.connect(self.apply_contrast_enhancement)
        self.odev1_result_label.clear()

    def odev2(self):
        self.odev2_label.setText("Ödev 2: Çizgi Tespiti")
        self.odev2_button.setText("Görüntü Yükle ve Çizgileri Tespit Et")
        self.odev2_button.clicked.disconnect()
        self.odev2_button.clicked.connect(self.detect_lines)
        self.odev2_result_label.clear()

    def odev3(self):
        self.odev3_label.setText("Ödev 2: Göz Tespiti")
        self.odev3_button.setText("Görüntü Yükle ve Gözleri Tespit Et")
        self.odev3_button.clicked.disconnect()
        self.odev3_button.clicked.connect(self.detect_eyes)
        self.odev3_result_label.clear()

    def apply_contrast_enhancement(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            image = cv2.imread(filename)
            if image is not None:
                enhanced_image = self.apply_sigmoid(image, curve_type='standard')
                cv2.imwrite("enhanced_image.png", enhanced_image)
                pixmap = QPixmap("enhanced_image.png")
                self.odev1_result_label.setPixmap(pixmap)
                self.odev1_result_label.setAlignment(Qt.AlignCenter)
            else:
                print("Görüntü yüklenemedi, dosya bozuk olabilir veya yanlış format.")

    def apply_sigmoid(self, image, curve_type='standard'):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x = np.arange(0, 256, 1)
        if curve_type == 'standard':
            sigmoid = 1 / (1 + np.exp(-0.05 * (x - 128)))
        elif curve_type == 'shifted':
            sigmoid = 1 / (1 + np.exp(-0.1 * (x - 100)))
        elif curve_type == 'inclined':
            sigmoid = 1 / (1 + np.exp(-0.1 * (x - 150)))
        else:
            sigmoid = np.tanh(0.1 * (x - 128))  # Örnek bir eğimli tanh fonksiyonu

        sigmoid = (sigmoid - np.min(sigmoid)) / (np.max(sigmoid) - np.min(sigmoid))
        img_contrast_enhanced = cv2.LUT(img, (sigmoid * 255).astype('uint8'))
        return img_contrast_enhanced

    def detect_lines(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite("lines_detected.png", image)
            pixmap = QPixmap("lines_detected.png")
            self.odev2_result_label.setPixmap(pixmap)
            self.odev2_result_label.setAlignment(Qt.AlignCenter)

    def detect_eyes(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            eyes = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
            if eyes is not None:
                eyes = np.uint16(np.around(eyes))
                for i in eyes[0, :]:
                    center = (i[0], i[1])
                    radius = i[2]
                    cv2.circle(image, center, radius, (0, 255, 0), 2)
            cv2.imwrite("eyes_detected.png", image)
            pixmap = QPixmap("eyes_detected.png")
            self.odev3_result_label.setPixmap(pixmap)
            self.odev3_result_label.setAlignment(Qt.AlignCenter)

    def deblur_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Basit bir sharpening kernel'i
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            deblurred_image = cv2.filter2D(gray, -1, kernel)
            cv2.imwrite("deblurred_image.png", deblurred_image)
            pixmap = QPixmap("deblurred_image.png")
            self.odev4_result_label.setPixmap(pixmap)
            self.odev4_result_label.setAlignment(Qt.AlignCenter)

    def detect_and_analyze_green_areas(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            image = cv2.imread(filename)
            green_mask = self.filter_green_regions(image)
            contours = self.find_contours(green_mask)
            properties = self.calculate_properties(contours, image)
            self.save_to_excel(properties)
            self.odev5_result_label.setText("Analiz tamamlandı ve Excel'e kaydedildi.")

    def filter_green_regions(self, image):
        # Renk aralığını ayarla
        lower_green = np.array([34, 100, 34])
        upper_green = np.array([80, 255, 80])
        mask = cv2.inRange(image, lower_green, upper_green)
        return mask

    def find_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def calculate_properties(self, contours, image):
        properties = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            diagonal = np.sqrt(w**2 + h**2)
            area = cv2.contourArea(contour)
            properties.append([center, w, h, diagonal, area])
        return properties

    def save_to_excel(self, properties):
        df = pd.DataFrame(properties, columns=['Center', 'Width', 'Height', 'Diagonal', 'Area'])
        df.to_excel('output.xlsx', index=False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
