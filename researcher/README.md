# Tadqiqotchi — Faza 1

Bitta niche uchun YouTube Data API orqali kanallarni yig'adi, raqamlarni kod bilan hisoblaydi, SQLite'ga yozadi va CSV chiqaradi. Claude, Notion va Telegram keyingi fazalar.

## O'rnatish

```bash
cd researcher
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` ichiga `YT_API_KEY` yozing. Boshqa kalitlar Faza 1 da ishlatilmaydi. `.env` gitga kirmaydi.

## Ishga tushirish

```bash
python main.py "cozy fantasy ambience" --no-claude
```

Foydali bayroqlar: `--days 180`, `--max-age 365`, `--min-avg 5000`, `--min-median 3000`, `--min-ratio 0.15`, `--pages 1`, `--region US`, `--refresh`.

Tugatilgan audit qayta ishga tushirilsa, dastur `oldin tekshirilgan` deb to'xtaydi. Qayta yig'ish uchun `--refresh`. Kvota tugab, natija chala saqlangan bo'lsa, keyingi ishga tushirish to'silmaydi.

## Natija

- Konsolda qisqa jadval va `Sarflangan kvota`.
- `output/<niche>_<sana>.csv` — outlier kanallar tepada. Sana kompyuterning mahalliy sanasi.
- `data/researcher.db` — `audits`, `channels`, `videos`, `expansions`.

## Hisoblash

Qidiruv ikki marta ketadi: `viewCount` va `relevance`. Takror video ID'lar tushadi. Har kanal `found_by` oladi: `views`, `relevance` yoki `both`. Har bir `search.list` 100 kvota.

Kanal yoshi, video soni, jami ko'rish, o'rtacha ko'rish, kunlik ko'rish va obunachi `channels.list` dan. Oxirgi 50 ta yuklama namunasi median, top ulush, o'rtacha davomiylik va haftalik yuklashni beradi. `recent10_median_views` shu namunaning eng yangi 10 tasidan (kam bo'lsa, boridan).

Yuklama ro'yxati faqat `yosh <= max_age` va `o'rtacha >= min_avg` bo'lgan kanallarga so'raladi. Outlier bo'lishi uchun yosh, o'rtacha, median (`>= 3000`), `median / o'rtacha >= 0.15` va top ulush (`<= 0.6`) kerak. O'rtacha 0 bo'lsa nisbat hisoblanmaydi va kanal outlier emas. Yashirin obunachi `None`.

`Outlier ulushi (faqat relevance)` — relevance qidiruvi qaytargan kanallar (`relevance` va `both`) ichidagi outlier ulushi. Bu kanallar 10 tadan kam bo'lsa, konsolda `tanlama kichik` chiqadi.

`uploads_per_week` — namuna soni, eng eski va eng yangi yuklama orasidagi haftalar soniga bo'lingan. Bitta yuklama yoki bir xil vaqtdagi yuklamalar uchun qiymat yo'q. Kanal yoshi 0 bo'lsa, kunlik ko'rish hisoblanmaydi.

`channels_analyzed` — qidiruvdan chiqqan barcha kanallar. `new_channels` — yoshi `max_age` dan oshmaganlar.

## Test

```bash
pytest
```

Jonli YouTube kalitisiz ham o'tadi: API javoblari mock qilinadi.

## Hali yo'q

Faza 2 (niche kengaytirish, sarlavha/thumbnail, ball) va Faza 3 (Notion, Telegram) qurilmagan. `--no-claude` siz ishga tushirilsa ham hozir faqat yig'ish bajariladi.
