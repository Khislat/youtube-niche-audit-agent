# TZ: Tadqiqotchi agent (YouTube niche audit)

Bu fayl Cursor uchun texnik topshiriq. Repo ildiziga `TZ.md` nomi bilan qo'y va Cursor'ga shunday de:
"TZ.md ni to'liq o'qi. Faqat Faza 1 ni qur. Har faza oxirida to'xta va natijani menga ko'rsat. TZ'dan chetga chiqsang, avval so'ra."

## 1. Maqsad

Foydalanuvchi bitta niche yozadi, masalan `python main.py "cozy fantasy ambience"`. Dastur:
1. YouTube Data API orqali shu niche bo'yicha yosh va kichik, lekin ko'rishi baland kanallarni (outlier) topadi.
2. Raqamlarni kod bilan hisoblaydi.
3. Sarlavha, thumbnail va IP xavfini Claude API bilan tahlil qiladi.
4. Bitta sahifalik hisobotni Notion'ga yozadi va Telegram'ga qisqa xabar yuboradi.
5. Natijani SQLite'ga saqlaydi (takror audit qilmaslik va keyin trendni ko'rish uchun).

Asosiy qoida: **raqamlarni faqat kod hisoblaydi. Claude API raqam o'ylab topmaydi, faqat berilgan raqamlarni sharhlaydi.**

## 2. Texnik talablar

- Python 3.11+, virtual muhit (`venv`).
- Kutubxonalar: `requests`, `python-dotenv`, `anthropic`, `pytest`. SQLite uchun standart `sqlite3`. Kerak bo'lmasa boshqasini qo'shma.
- YouTube Data API v3 (rasmiy). **Scraping yo'q.**
- Hamma secret `.env` faylida. `.env` `.gitignore`'da bo'lsin, `.env.example` repoga tushsin.
- Kod izohlari va log xabarlari ingliz tilida, foydalanuvchiga chiqadigan hisobot o'zbek tilida.

`.env.example`:
```
YT_API_KEY=
ANTHROPIC_API_KEY=
CLAUDE_MODEL=          # model nomi shu yerdan o'qiladi, kodga qattiq yozilmaydi
NOTION_TOKEN=
NOTION_PARENT_PAGE_ID= # hisobot sahifalari shu sahifa ostida yaratiladi
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

## 3. Papka tuzilmasi

```
researcher/
  main.py
  config.py
  expand.py
  collect.py
  metrics.py
  analyze.py
  score.py
  report.py
  store.py
  prompts/
  tests/
  .env.example
  requirements.txt
  README.md
```

## 4. CLI

```
python main.py "niche" [--days 180] [--max-age 365] [--min-avg 5000]
                       [--interest 1-5] [--pages 1] [--region US]
                       [--no-claude] [--no-notion] [--no-telegram] [--refresh]
```

- `--no-claude`: faqat Faza 1 natijasi (CSV), Claude API chaqirilmaydi.
- `--refresh`: oldin tekshirilgan niche'ni qayta tekshiradi. Bo'lmasa xotiradan topilsa, "oldin tekshirilgan" deb chiqib to'xtaydi.
- `--interest`: foydalanuvchining shaxsiy qiziqishi (standart 3).

## 5. Modullar

### config.py
Standart qiymatlar: `days=180`, `max_age=365`, `min_avg=5000`, `min_median=1000`, `max_top_share=0.6`, `pages=1`, `max_videos_per_channel=50`. Ball og'irliklari bir joyda (5.7 ga qarang).

### collect.py (Faza 1)
- `search_videos(query, days, pages, region) -> list[video_id]`: `search.list`, `type=video`, `order=viewCount`, `videoDuration=long`, `publishedAfter`. Har chaqiruv 100 kvota.
- `get_videos(ids) -> list[dict]`: `videos.list` (`snippet,statistics,contentDetails`), 50 tadan partiya bilan.
- `get_channels(ids) -> list[dict]`: `channels.list` (`snippet,statistics,contentDetails`).
- `get_channel_uploads(channel_id, n=50) -> list[dict]`: `playlistItems.list` (kanalning uploads playlist'i, `contentDetails.relatedPlaylists.uploads`), keyin `videos.list` bilan statistika.
- Kvota hisoblagichi: har chaqiruvdan keyin sarflangan kvota yig'iladi va oxirida chop etiladi. Kvota tugasa (403 `quotaExceeded`) dastur tushunarli xabar bilan to'xtaydi va shu paytgacha yig'ilganini saqlaydi.
- Qayta urinish (retry): vaqtincha xatolar (5xx) uchun 3 marta, kutish bilan.
- Kanallar uchun `get_channel_uploads` faqat birinchi filtrdan (yosh va o'rtacha ko'rish) o'tgan kanallarga chaqiriladi, kvota tejash uchun.

### metrics.py (Faza 1)
Har kanal uchun (oxirgi 50 yuklama bo'yicha):
- `age_days`, `videos`, `total_views`
- `avg_views_per_video = total_views / videos`
- `median_views` (oxirgi 50 yuklama)
- `top_video_share = eng ko'p ko'rilgan video / yuklamalar yig'indisi`
- `views_per_day = total_views / age_days`
- `avg_duration_min`, `uploads_per_week` (yuklamalar sanalaridan)
- `subs` (yashirin bo'lsa `None`)
- `is_outlier = age_days <= max_age and avg >= min_avg and median_views >= min_median and top_video_share <= max_top_share`

Niche darajasida:
- `channels_analyzed`, `outliers`, `outlier_ratio = outliers / channels_analyzed`
- `new_channels` (yoshi <= max_age bo'lgan kanallar soni)
- `median_outlier_views_per_day`
- `typical_duration_min`, `typical_uploads_per_week` (outlier'lar medianasi)

Barcha formulalar uchun `tests/test_metrics.py`da kichik sintetik misollar bilan test yoz.

### store.py (Faza 1)
SQLite fayl `data/researcher.db`. Jadvallar:
- `audits(id, niche, created_at, params_json, quota_used, summary_json)`
- `channels(audit_id, channel_id, title, url, age_days, videos, total_views, avg_views, median_views, top_share, views_per_day, avg_duration_min, uploads_per_week, subs, is_outlier)`
- `videos(audit_id, channel_id, video_id, title, views, duration_min, published_at, thumbnail_url)`
- `expansions(audit_id, term)`

Funksiyalar: `init_db()`, `save_audit(...)`, `find_audit(niche)`, `recent_audits(n)`.

Faza 1 natijasi: `output/<niche>_<sana>.csv` (kanallar jadvali, outlier'lar tepada) va konsolga qisqa jadval.

### expand.py (Faza 2)
Claude API orqali niche'dan 10-15 ta yaqin qidiruv so'zi chiqaradi. Javob qat'iy JSON ro'yxat bo'lishi kerak. Yaroqsiz JSON bo'lsa bir marta qayta so'raydi. Oldin tekshirilgan so'zlar (xotiradan) chiqarib tashlanadi.

### analyze.py (Faza 2)
Faqat har niche bo'yicha eng ko'p ko'rilgan 10 ta outlier video uchun.
Claude API'ga beriladi: sarlavhalar, ko'rishlar, thumbnail URL'lar (rasm sifatida), va metrics.py natijasi (raqamlar tayyor holda).
Claude qaytaradi (JSON):
- `title_patterns`: takrorlanadigan sarlavha tuzilmalari
- `keywords`: takrorlanadigan kalit so'zlar
- `thumbnail_style`: umumiy uslub
- `ip_risks`: brend/franshiza nomlari ro'yxati va qaysi videolarda
- `ai_feasibility` (1-5) va sababi: AI musiqa va rasm bilan shu uslubni qilish qanchalik oson
- `differentiation` (1-5) va sababi: yangi kanal qanday ajralib turishi mumkin
- `verdict_text`: "kirishga arziydimi" haqida 3-5 gaplik xulosa, faqat berilgan raqamlarga tayanib

Prompt matnlari `prompts/` papkasida alohida fayllarda. Claude'ga aniq qoida yoziladi: "Berilmagan raqamni o'ylab topma. Ma'lumot yetarli bo'lmasa, shuni yoz."

### score.py (Faza 2)
Beshta mezon, har biri 1-5:
- `demand`: `median_outlier_views_per_day` bo'yicha chegaralar (config'da)
- `supply_gap`: `outlier_ratio` bo'yicha chegaralar
- `ai_feasibility`: Claude'dan
- `differentiation`: Claude'dan
- `interest`: `--interest`
`total` = og'irlikli o'rtacha (standart: hammasi teng). Kichik tanlama bo'lsa (`channels_analyzed < 10`) hisobotga ogohlantirish qo'shiladi.

### report.py (Faza 3)
Notion: `NOTION_PARENT_PAGE_ID` ostida har audit uchun alohida sahifa, o'zbek tilida. Tuzilma:
1. Sarlavha: niche, sana, umumiy ball
2. Xulosa (verdict_text)
3. Raqamlar jadvali (top outlier'lar: havola, yosh, video soni, o'rtacha, median, ko'rish/kun)
4. To'yinganlik
5. Format (uzunlik, chastota)
6. Sarlavha va thumbnail tahlili
7. IP xavflari
8. Ogohlantirishlar (kichik tanlama va h.k.)
9. Parametrlar va sarflangan kvota

Telegram: bitta qisqa xabar: niche, umumiy ball, 1-2 ta eng yaxshi outlier havolasi, Notion sahifasi havolasi.

## 6. Fazalar va qabul mezonlari

**Faza 1: yig'ish va hisoblash** (`--no-claude` rejimi)
- `python main.py "cozy fantasy ambience" --no-claude` ishlaydi, CSV yaratadi, kvotani chop etadi.
- `tests/test_metrics.py` o'tadi.
- Natija SQLite'ga yoziladi, qayta ishga tushirilsa "oldin tekshirilgan" deydi.

**Faza 2: tahlil**
- Niche kengaytirish, thumbnail va sarlavha tahlili, ball hisoblanadi.
- Claude javobi JSON sxemasiga mos kelmasa, bir marta qayta so'raladi, keyin aniq xato beriladi.

**Faza 3: hisobot**
- Notion sahifasi yaratiladi, Telegram xabari boradi. `--no-notion` va `--no-telegram` ishlaydi.

**Faza 4: avto-rejim** (alohida TZ, hozir qilinmaydi)

## 7. Chegaralar (qilinmaydi)

- Daromad, RPM yoki CTR taxmini.
- Obunachi bo'yicha tahlil (faqat ma'lumot sifatida saqlanadi).
- Izohlar, tag'lar, transkriptlar.
- Dashboard va grafiklar.
- Scraping.
- O'z kanal tahlili.
- Server va n8n (Faza 4).

## 8. Cursor uchun qoidalar

- Bir vaqtda bitta fazani qur, oxirida to'xta.
- Faza 1 uchun real YouTube API key kerak. Key bo'lmasa, API javoblarini mock qilib test yoz va foydalanuvchiga key kerakligini ayt.
- Hech qachon secret'larni kodga, logga yoki commit'ga yozma.
- Mavjud `audit.py` (Maslahatchi yozgan qoralama, sinalmagan) faqat g'oya uchun, undan nusxa ko'chirish shart emas.
- Ikkilanganda taxmin qilma, so'ra.
