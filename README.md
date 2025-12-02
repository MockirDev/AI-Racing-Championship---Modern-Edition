# 🏎️ AI Racing Championship - Modern Edition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyGame](https://img.shields.io/badge/PyGame-2.0%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**AI Racing Championship**, Derin Pekiştirmeli Öğrenme (Deep Reinforcement Learning) kullanan otonom araçların yarıştığı, gelişmiş bir 2D yarış simülasyonudur. Proje, kendi kendine öğrenen yapay zeka ajanları, kapsamlı bir pist editörü ve modern bir kullanıcı arayüzü sunar.

## 🌟 Özellikler

### 🧠 Gelişmiş Yapay Zeka (AI)
*   **Deep Q-Network (DQN):** Ajanlar, çevrelerini algılayarak (ray-casting sensörleri) en iyi sürüş stratejilerini öğrenir.
*   **Multi-Agent Sistemi:** Birden fazla araç aynı anda yarışır ve deneyim paylaşımı (Experience Sharing) yapabilir.
*   **Adaptif Zorluk:** AI, performansına göre zorluk seviyesini dinamik olarak ayarlar.
*   **Davranışsal Modlar:** Agresif, Defansif, Dengeli ve Adaptif sürüş karakterleri.

### 🏎️ Fizik ve Mekanik
*   **Gerçekçi Araç Fiziği:** İvme, sürtünme, dönüş yarıçapı ve drift mekanikleri.
*   **Performans Presetleri:** Yarış sırasında değiştirilebilir modlar (Hız, Dengeli, Yol Tutuşu).
*   **Boost Bölgeleri:** Pist üzerindeki özel bölgelerde hız artışı.
*   **Görsel Efektler:** Araç arkası izleri (trails), dinamik renkler ve modern HUD.

### 🛠️ Pist Editörü
*   **Spline Tabanlı Yumuşatma:** Keskin köşeleri otomatik olarak yumuşatarak yarışa uygun hale getirir.
*   **Şablonlar:** Hazır pist şablonlarını (Oval, S-Curve vb.) kullanma imkanı.
*   **Akıllı Checkpointler:** Otomatik checkpoint yerleşimi.
*   **Kaydet & Yükle:** Tasarladığınız pistleri JSON formatında kaydedin ve paylaşın.

### 📊 Telemetri ve Analiz
*   **Canlı İstatistikler:** FPS, AI öğrenme verileri (Reward, Loss), tur zamanları.
*   **Veri Dışa Aktarma:** Yarış verilerini analiz için CSV/JSON olarak kaydetme.

## 🚀 Kurulum

Projenin çalışması için Python ve aşağıdaki kütüphanelere ihtiyacınız vardır:

```bash
pip install pygame torch numpy
```

## 🎮 Kullanım

Oyunu başlatmak için `main.py` dosyasını çalıştırın:

```bash
python main.py
```

### Ana Menü
*   **Başlat:** Simülasyonu/Yarışı başlatır.
*   **Harita Editörü:** Kendi pistlerinizi tasarlayın.
*   **Ayarlar:** Tur sayısı, araç sayısı, AI zorluğu vb. ayarları değiştirin.
*   **Çıkış:** Oyundan çıkar.

### Yarış Kontrolleri

| Tuş | Fonksiyon |
| --- | --- |
| **WASD / Ok Tuşları** | Kamerayı hareket ettir (Pan) |
| **Mouse Tekerleği** | Yakınlaştır / Uzaklaştır (Zoom) |
| **Mouse Orta Tuş** | Kamerayı sürükle |
| **R** | Kamerayı Sıfırla (Piste odakla) |
| **S** | Sensörleri Göster/Gizle (Ray-casting çizgileri) |
| **P** | Performans Bilgilerini Göster (FPS, Zoom vb.) |
| **I** | AI İstatistiklerini Göster (Reward, Epsilon vb.) |
| **H** | HUD'u Göster/Gizle |
| **M** | Mini haritayı aç/kapat |
| **ESC** | Duraklat / Menü |

### Performans Modları (Canlı Değiştirilebilir)
*   `1`: **Balanced** (Dengeli)
*   `2`: **Speed** (Hız Odaklı - Daha yüksek son hız, daha az yol tutuş)
*   `3`: **Handling** (Yol Tutuş Odaklı - Daha iyi dönüş, daha düşük hız)
*   `T`: Modlar arasında geçiş yap

### AI Model Yönetimi
*   `F5`: Tüm AI modellerini kaydet (`ai_models/` klasörüne).
*   `F9`: Kaydedilmiş AI modellerini yükle.

## 📂 Proje Yapısı

*   `main.py`: Oyun döngüsü, UI ve sahne yönetimi.
*   `ai.py`: DQN modeli, Replay Memory ve Ajan mantığı.
*   `car.py`: Araç fiziği, sensörler ve ödül sistemi.
*   `editor.py`: Pist tasarım aracı.
*   `telemetry.py`: Veri kayıt ve analiz sistemi.
*   `tracks/`: Pist verilerinin (JSON) saklandığı klasör.

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
