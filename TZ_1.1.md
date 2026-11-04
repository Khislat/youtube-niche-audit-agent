# TZ qo'shimchasi: Faza 1.1 (outlier filtri va qidiruv og'ishini kamaytirish)

Bu fayl `TZ.md` ga qo'shimcha. Avval `TZ.md` ni, keyin shu faylni o'qi. Faqat shu yerda yozilgan o'zgarishlarni qil. Faza 2 va 3 ga tegma. Tugagach to'xta va natijani ko'rsat.

## Nima uchun

Faza 1 jonli sinovdan o'tdi ("cozy fantasy ambience" va "japanese zen focus music"). Ikki muammo topildi:
1. Median chegarasi (1 000) past. Medianasi 1 085, o'rtachasi 16 857 bo'lgan kanal outlier deb belgilandi. Bu doimiy outlier emas, bitta-ikkita omadli video.
2. Qidiruv faqat `order=viewCount` bo'yicha. Shuning uchun faqat kamida bitta katta videosi bor kanallar topiladi va outlier ulushi haqiqatdagidan baland chiqadi.

## O'zgarishlar

### 1. Outlier qoidasi (metrics.py, config.py)
Yangi standart qiymatlar:
- `min_median = 3000` (oldin 1000)
- `min_median_avg_ratio = 0.15`

Yangi qoida:
```
is_outlier = age_days <= max_age
         and avg_views_per_video >= min_avg
         and median_views >= min_median
         and median_views / avg_views_per_video >= min_median_avg_ratio
         and top_video_share <= max_top_share
```
`avg_views_per_video` 0 bo'lsa nisbat hisoblanmaydi va kanal outlier emas.
CLI'ga `--min-median` va `--min-ratio` parametrlari qo'shilsin.

### 2. Qidiruvni ikki tartibda bajarish (collect.py)
Har niche uchun `search.list` ikki marta chaqiriladi:
- `order=viewCount` (hozirgidek)
- `order=relevance`

Qolgan parametrlar bir xil (`type=video`, `videoDuration=long`, `publishedAfter`, `regionCode`). Ikki natijadagi video ID'lar birlashtiriladi, takrorlari olib tashlanadi. Har kanal uchun qaysi tartib(lar)da topilgani saqlansin: `found_by` = `"views"`, `"relevance"` yoki `"both"`.
Kvota hisobi to'g'ri yuritilsin (har qidiruv 100). Bir audit taxminan 230-250 kvota olishi kutiladi.

### 3. Yangi ustunlar (metrics.py, store.py, CSV)
- `recent10_median_views`: oxirgi 10 yuklamaning medianasi (kanal hozir ham ishlayaptimi yoki bir marta portlab to'xtaganmi ko'rinadi). Yuklamalar 10 tadan kam bo'lsa, borlari bo'yicha hisoblanadi.
- `found_by`: yuqorida yozilgan.
- `median_avg_ratio`: median / o'rtacha.

`channels` jadvaliga shu uchta ustun qo'shilsin. Eski bazadagi jadval bilan mos bo'lishi uchun oddiy migratsiya yoz (`ALTER TABLE ... ADD COLUMN`), mavjud ma'lumotni o'chirma.

### 4. Niche darajasidagi yangi ko'rsatkich
`outlier_ratio_by_source`: faqat `relevance` orqali topilgan kanallar ichida outlier ulushi (og'ishsiz baho uchun). Konsol chiqishida va CSV tepasida (yoki summary_json'da) alohida ko'rsatilsin:
```
Outlier ulushi (hammasi): 23%
Outlier ulushi (faqat relevance): 11%
```
`relevance`'da topilgan kanal soni 10 dan kam bo'lsa, ogohlantirish chiqarilsin: "tanlama kichik".

### 5. Konsol jadvali
Mavjud ustunlarga `Oxirgi10 median` va `Topilgan` (found_by) qo'shilsin.

## Testlar
`tests/test_metrics.py` ga qo'shilsin:
- median 1085, o'rtacha 16857 bo'lgan kanal outlier EMAS (ikkala yangi shart ham buzilgan).
- median 8000, o'rtacha 20000, top ulush 0.3, yosh 100 kun bo'lgan kanal outlier.
- o'rtacha 0 bo'lsa xato chiqmasligi.
- `found_by` birlashtirish: bir kanal ikkala qidiruvda topilsa `"both"`.
- `recent10_median_views` 10 tadan kam yuklamada ishlashi.
- Eski bazani migratsiya qilish xato bermasligi.
Mavjud 33 test o'tishda davom etsin (kerak bo'lsa yangi standart qiymatlarga moslab yangila, sababini yoz).

## Qabul mezonlari
- `python main.py "cozy fantasy ambience" --no-claude` va `python main.py "japanese zen focus music" --no-claude` ishlaydi, `--refresh` bilan.
- Konsolda `Outlier ulushi (hammasi)` va `(faqat relevance)` ikkalasi chiqadi.
- CSV'da `found_by`, `recent10_median_views`, `median_avg_ratio` ustunlari bor.
- Sarflangan kvota har audit uchun taxminan 230-250 atrofida.
- Barcha testlar o'tadi.

## Qilinmaydi
Claude API, ball hisoblash, Notion, Telegram, yangi kutubxonalar. Ikkilanganda taxmin qilma, so'ra.
