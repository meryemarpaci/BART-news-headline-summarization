# Haber Başlıklarından Otomatik Özetleme Sistemi

### Colab: https://colab.research.google.com/drive/15NgtTG7lxyimnut_7dQA_yNhe9Tpal3X?usp=sharing
### Hugging Face: https://huggingface.co/meryemarpaci/BART-news-headline-summarization/

## Proje Sonuçları

**Model:** facebook/bart-base  
**Platform:** Google Colab (NVIDIA A100-SXM4-40GB)  
**Eğitim Süresi:** 7.6 dakika (454 saniye)  
**Epoch:** 5 (tam eğitim)  
**Batch Size:** 80 (optimize edilmiş)

## Final ROUGE Skorları

- **ROUGE-1**: 0.3353 (33.53%) 
- **ROUGE-2**: 0.1268 (12.68%) 
- **ROUGE-L**: 0.2362 (23.62%) 

## 📊 Eğitim Metrikleri

- **Final Training Loss:** 3.4359
- **Final Evaluation Loss:** 4.8902
- **Training Samples per Second:** 63.21
- **Evaluation Samples per Second:** 10.75
- **Veri Seti:** 5,742 eğitim, 66 doğrulama, 57 test örneği

## Test Sonuçları

### Örnek 1: Uluslararası Hukuk (ROUGE-L: 26.87%)
```
Orijinal: Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June. Israel and the United States opposed the move, which could open the door to war crimes investigations against Israelis.

Üretilen: Palestinians officially become 123rd member of the International Criminal Court. Israel and the United States oppose Palestinian efforts to join the court. Palestinian Authority formally becomes a State Party to the Rome Statute.
```

### Örnek 2: Hayvan Kurtarma (ROUGE-L: 34.78%)
```
Orijinal: Theia, a bully breed mix, was apparently hit by a car, whacked with a hammer and buried in a field. "She's a true miracle dog and she deserves a good life," says Sara Mellado, who is looking for a home for Theia.

Üretilen: A dog in Washington State has used up at least three of her own after being hit by a car, apparently whacked on the head with a hammer in a misguided mercy killing and then buried in a field -- only to survive. Theia, a friendly white-and-black bully...
```

### Örnek 3: Dış Politika (ROUGE-L: 15.62%)
```
Orijinal: Mohammad Javad Zarif has spent more time with John Kerry than any other foreign minister. He once participated in a takeover of the Iranian Consulate in San Francisco. The Iranian foreign minister tweets in English.

Üretilen: Iranian Foreign Minister Mohammad Javad Zarif received a hero's welcome as he arrived in Iran. He is "polished" and, unusually for one burdened with such weighty issues, "jovial."
```

### Örnek 4: Sağlık (ROUGE-L: 21.43%)
```
Orijinal: 17 Americans were exposed to the Ebola virus while in Sierra Leone in March. Another person was diagnosed with the disease and taken to hospital in Maryland. National Institutes of Health says the patient is in fair condition after weeks of treatment.

Üretilen: Five Americans who were monitored for three weeks at Omaha, Nebraska, hospital after being exposed to Ebola in West Africa have been released, Nebraska Medicine spokesman said in an email Wednesday. One of the five had a heart-related issue on Saturday...
```

### Örnek 5: Üniversite Olayları (ROUGE-L: 24.69%)
```
Orijinal: Student is no longer on Duke University campus and will face disciplinary review. School officials identified student during investigation and the person admitted to hanging the noose, Duke says. The noose, made of rope, was discovered on campus about 2...

Üretilen: A Duke student has admitted to hanging a noose made of rope from a tree near a student union, university officials said. Students and faculty members marched Wednesday afternoon chanting "We are not afraid. We stand together". The student was identified...
```

## 📊 Performans Analizi

**Güçlü Yönler:**
- ✅ Hayvan kurtarma konularında yüksek başarı (34.78% ROUGE-L)
- ✅ Ana olayları doğru özetleme
- ✅ Önemli varlık isimlerini koruma
- ✅ Karakter limitlerine uyum (maksimum 256 karakter)
- ✅ Test metinlerinde 512 karakter sınırı

**Ortalama ROUGE-L:** ~24.68% (5 örnek üzerinden)

## 🎯 İyileştirmeler

**Yapılan Güncellemeler:**
- ✅ Epoch sayısı 3'ten 5'e artırıldı
- ✅ Loss değerleri CSV dosyasına kaydediliyor
- ✅ Özetler maksimum 256 karakter ile sınırlandırıldı
- ✅ Test metinleri maksimum 512 karakter ile sınırlandırıldı
- ✅ Colab uyumlu optimizasyonlar yapıldı
- ✅ Detaylı logging ve görselleştirme eklendi
- ✅ Batch size 80'e optimize edildi

## 🚀 Model Parametreleri

- **Model:** facebook/bart-base
- **Batch Size:** 80
- **Learning Rate:** 3e-5
- **Training Loss:** 3.4359
- **Evaluation Loss:** 4.8902
- **Veri Seti:** 5,742 eğitim, 66 doğrulama, 57 test örneği
- **Özet Maksimum Karakter:** 256
- **Test Metni Maksimum Karakter:** 512

## 📁 Çıktı Dosyaları

- **Model:** `./saved_model/`
- **Eğitim sonuçları:** `./results/train_results.json`
- **Değerlendirme sonuçları:** `./results/eval_results.json`
- **Loss kayıtları:** `./logs/training_losses_YYYYMMDD_HHMMSS.csv`
- **Eğitim özeti:** `./results/training_summary.txt`
- **Örnek özetler:** `./results/sample_summaries.json`
- **Okunabilir özetler:** `./results/sample_summaries_readable.txt`
- **Eğitim grafikleri:** `./results/training_plots.png`

## 🎉 Sonuç

Model başarıyla eğitildi ve test edildi. Karakter limitleri uygulandı, loss değerleri kaydedildi ve detaylı metrikler toplandı. Sistem artık Colab'da çalışmaya hazır!
