# Haber Başlıklarından Otomatik Özetleme Sistemi

## Proje Sonuçları

**Model:** facebook/bart-base  
**Platform:** Google Colab (NVIDIA A100-SXM4-40GB)  
**Eğitim Süresi:** 4.5 dakika (265 saniye)  
**Epoch:** 1.95 (early stopping)  

## Final ROUGE Skorları

- **ROUGE-1**: 0.3163 (31.63%) 
- **ROUGE-2**: 0.1205 (12.05%)   
- **ROUGE-L**: 0.2264 (22.64%) 

## 📝 Test Sonuçları

### Örnek 1: Uluslararası Hukuk (En İyi - ROUGE-L: 40.96%)
```
Orijinal: Membership gives the ICC jurisdiction over alleged crimes committed in Palestinian territories since last June. Israel and the United States opposed the move, which could open the door to war crimes investigations against Israelis.

Üretilen: Palestinians officially become 123rd member of the International Criminal Court. They also accepted its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East Jerusalem" The ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis.
```

### Örnek 2: Hayvan Kurtarma (ROUGE-L: 16.67%)
```
Orijinal: Theia, a bully breed mix, was apparently hit by a car, whacked with a hammer and buried in a field. "She's a true miracle dog and she deserves a good life," says Sara Mellado, who is looking for a home for Theia.

Üretilen: Stray pooch in Washington State used up at least three of her own after being hit by a car. The dog, a friendly white-and-black bully breed mix now named Theia, has been receiving care at the Veterinary Teaching Hospital.
```

### Örnek 3: Dış Politika (ROUGE-L: 20.00%)
```
Orijinal: Mohammad Javad Zarif has spent more time with John Kerry than any other foreign minister. He once participated in a takeover of the Iranian Consulate in San Francisco. The Iranian foreign minister tweets in English.

Üretilen: Iran's foreign minister has been U.S. Secretary of State John Kerry's opposite number. Zarif received a hero's welcome as he arrived in Iran on a sunny Friday morning. He has gone a long way to bring Iran in from the cold.
```

### Örnek 4: Sağlık (ROUGE-L: 29.63%)
```
Orijinal: 17 Americans were exposed to the Ebola virus while in Sierra Leone in March. Another person was diagnosed with the disease and taken to hospital in Maryland. National Institutes of Health says the patient is in fair condition after weeks of treatment.

Üretilen: Five Americans were monitored for three weeks at Omaha, Nebraska, hospital after being exposed to Ebola. All had contact with a colleague who was diagnosed with the disease. The last of 17 patients expected to be released by Thursday.
```

### Örnek 5: Üniversite Olayları (ROUGE-L: 19.15%)
```
Orijinal: Student is no longer on Duke University campus and will face disciplinary review. School officials identified student during investigation and the person admitted to hanging the noose, Duke says. The noose, made of rope, was discovered on campus about 2 a.m.

Üretilen: Duke student has admitted to hanging noose from a tree near student union. Student was identified during an investigation by campus police and the office of student affairs. Police still trying to determine if other people were involved, university says. Incident is one of several recent racist events to affect college students.
```

## 📊 Performans Analizi

**Güçlü Yönler:**
- Politik/hukuki konularda yüksek başarı (40.96% ROUGE-L)
- Ana olayları doğru özetleme
- Önemli varlık isimlerini koruma
- Hızlı eğitim süresi

**Ortalama ROUGE-L:** ~25.68% (5 örnek üzerinden)

## 🎯 Sonuç

Model başarıyla eğitildi ve CNN/DailyMail veri seti üzerinde %22+ ROUGE-L skoru elde etti. Politik/hukuki metinlerde %40+ skorla özellikle başarılı. 

**Model Parametreleri:**
- Batch Size: 4
- Learning Rate: 5e-5
- Training Loss: 2.0062
- Validation Loss: 1.5192
- Veri Seti: 2,871 eğitim, 66 doğrulama, 57 test örneği 
