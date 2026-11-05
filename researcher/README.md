# Tadqiqotchi — Faza 1 / 1.2

YouTube Data API orqali niche kanallarini yig'adi, outlier'larni hisoblaydi, AI/brend belgilarini va takliflarni kod bilan chiqaradi. Claude, Notion va Telegram keyingi fazalar.

## O'rnatish

```bash
cd researcher
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` ichiga `YT_API_KEY` yozing. `.env` gitga kirmaydi.

## Ishga tushirish

Bitta niche:

```bash
python main.py "cozy fantasy ambience" --no-claude
```

Ko'p niche:

```bash
python main.py --file niches.txt --no-claude --suggest
```

`niches.txt`: har qatorda bitta niche. Bo'sh qatorlar va `#` bilan boshlangan qatorlar o'tkaziladi.

Foydali bayroqlar: `--days 180`, `--max-age 365`, `--min-avg 5000`, `--min-median 3000`, `--min-ratio 0.15`, `--pages 1`, `--region US`, `--refresh`, `--max-quota 8000`, `--suggest`, `--auto-suggest N`, `--min-channels 2`.

`--max-quota`: shu ishga tushirishda jami kvota limiti. Keyingi niche boshlanishidan oldin `joriy + 250` limitdan oshsa, to'xtaydi (oldin saqlangan natijalar qoladi).

`--suggest`: outlier top-videolar sarlavha/teglaridan taklif CSV. `--auto-suggest N`: eng yaxshi N ta iborani shu navbatda audit qiladi (faqat 1 daraja chuqurlik).

Tugatilgan audit `--refresh`siz o'tkazib yuboriladi. Kvota tugasa, chala natija saqlanadi.

## Natija

Har niche uchun:

- `output/<niche>_<sana>.csv` — kanallar (outlier tepada). Tepada izoh: **belgi yo'qligi AI yoki brend ishlatilmaganini isbotlamaydi**; ko'p kanallar buni yozmaydi.
- `output/<niche>_<sana>_top_videos.csv` — har outlier kanalning eng ko'p ko'rilgan 5 ta videosi.
- `output/<niche>_<sana>_suggestions.csv` — `--suggest` bilan.
- `data/researcher.db`

Ko'p niche tugagach: `output/compare_<sana>.csv` (relevance outlier % bo'yicha tartib).

## `keywords.json`

AI va brend kalit so'zlari hamda taklif uchun stop-so'zlar shu faylda. Matn skaneri faqat **butun so'z/ibora** (`\b`) bo'yicha qidiradi. `"ai"` alohida ro'yxatda yo'q (juda noaniq). Faylni tahrirlang va qayta ishga tushiring — qo'shimcha kutubxona kerak emas.

## Hisoblash (qisqa)

Qidiruv: `viewCount` + `relevance`. Outlier qoidalari: yosh, o'rtacha, median (`>= 3000`), median/o'rtacha (`>= 0.15`), top ulush (`<= 0.6`).

Kanal `ai_flag_share` / `brand_flag_share`: oxirgi 50 yuklama ichida kamida bitta AI yoki brend termini bor videolar ulushi.

Bir niche uchun kvota odatda ~230–250 (qo'shimcha `snippet` maydonlari qo'shimcha kvota olmaydi).

## Test

```bash
pytest
```

## Hali yo'q

Claude tahlili, thumbnail/rasm tahlili, ball, Notion, Telegram, autocomplete/scraping.
