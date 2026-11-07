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

## Audio Audit Lab (Faza 2b)

Lokal audio faylni til modelsiz o'lchaydi. Internet, YouTube va LLM chaqiruvi yo'q.

### O'rnatish

`ffmpeg` tizimda bo'lishi kerak (`brew install ffmpeg`). Bo'lmasa buyruq tushunarli xato beradi.

```bash
cd researcher
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-audio.txt
```

Ixtiyoriy (sekin, og'ir):

```bash
pip install -r requirements-audio-extra.txt
```

`--stems` uchun Demucs `htdemucs_6s` og'irliklari tahlil paytida yuklanmaydi. Bir marta, o'rnatish bosqichida:

```bash
python -c "from demucs.pretrained import get_model; get_model('htdemucs_6s')"
```

`--tags` uchun Essentia modellari `data/audio_models/` ichiga qo'yiladi (kod ularni yuklab olmaydi):

- `discogs-effnet-bs64-1.pb`
- `genre_discogs400-discogs-effnet-1.pb` va `.json`
- `mtg_jamendo_instrument-discogs-effnet-1.pb` va `.json`
- `mtg_jamendo_moodtheme-discogs-effnet-1.pb` va `.json`

**Essentia modellari CC BY-NC-SA 4.0 (tijoriy litsenziya alohida mavjud). Tijoriy foydalanishdan oldin shartlarni tekshiring.**

### Ishga tushirish

```bash
python audio_lab.py analyze --input track.wav --out output/audio
python audio_lab.py analyze --input papka --clip 30 90 --stems --tags
python audio_lab.py profile --dir papka --name uslub
python audio_lab.py compare --track track.wav --profile output/audio/style_profile_uslub.json
```

Uzun miksni treklarga ajratish:

```bash
python audio_lab.py segment --input mix.wav --expect-tracks 12-15 --min-track 90
python audio_lab.py analyze --input mix.wav --segment --sr 44100
python audio_lab.py analyze --input mix.wav --sample-times 5:00,15:00,25:00,40:00,55:00 --clip-len 240
python audio_lab.py profile --dir papka --name uslub --per-track
python audio_lab.py summarize --report output/audio/<papka>/report.json --out output/audio/<papka>/summary.md
```

`summarize` katta `report.json` dan 120 qatordan oshmaydigan `summary.md` yozadi. `--report` berilmasa `output/audio/` ichidagi eng yangi hisobot olinadi. Timeline va grafik nuqtalari kirmaydi. Parcha guruhlari silhouette emas, kosinus masofa `1.0` kesimi bilan olinadi va `[PROKSI, kichik tanlama]` deb belgilanadi.

Segmentatsiya o'zi 22050 Hz da ishlaydi. `--sr` faqat har trekning chastota va spektr o'lchovlariga tegishli; default 22050. Tiniqlik tahlili uchun 44100 tavsiya etiladi. 15 daqiqadan uzun faylda `analyze` ogohlantiradi, lekin `--segment` berilmasa segmentatsiyani o'zi yoqmaydi. `--expect-tracks` trek sonini majburlamaydi.

`--boundaries manual.csv` (`start_s` ustuni) avtomatik chegarani almashtiradi. Ikkala manba berilsa, qo'lda berilgan chegaralar ishlatiladi. `--tracklist-text` foydalanuvchi joylagan matndan vaqtni oladi va avtomatik chegaralar bilan ±5 s da solishtiriladi. `--export-segments` har trekni `output/audio/<fayl>_<hash>/segments/NN.wav` ga kesadi; default o'chiq.

Balandlik `ffmpeg ebur128` bilan o'lchanadi.

Boshqa odamning musiqasini repo ildiziga qo'ymang. Alohida treklar `researcher/data/audio_inputs/<papka>/` ichida qoladi. Tayyor uzun miksni shu papkaga yoki repodan tashqaridagi yo'lga qo'yish ham mumkin. `wav`, `mp3`, `flac`, `m4a`, `webm` va shu kabi audio gitignore da. `git add -f` bilan majburan qo'shmang.

`--stems` va `--tags` default o'chiq. Bir xil fayl va parametrlar qayta hisoblanmaydi; `--refresh` majbur qiladi. Raqamlar 4 o'nli yaxlitlanadi.

### Chegara nomzodlari (ishonchsiz, tracklist yoki qo'lda tuzatish tavsiya etiladi)

`strength` chegaraning kuchini ko'rsatadi, to'g'riligini emas; A va B real mixlarda soxta chegaralar ham `high` bo'lgan.

Qattiq kesish va 0.5 soniyadan uzun jimlik aniqroq topiladi. Crossfade va bir-biriga o'xshash tembrli treklarda chegara bir necha soniyaga surilishi yoki umuman tushib qolishi mumkin. Shubhali joyda `manual.csv` yoki tracklist matnidan foydalaning. Takror trek va `style_groups` proksi.

O'lchangan natija, 4 soniyalik crossfade, `--min-track 90`, `k=2.6`. A va B turli treklardan yig'ilgan. B sozlamada ishlatilmagan.

| Mix | ±3 s precision | ±3 s recall | ±5 s precision | ±5 s recall |
| --- | --- | --- | --- | --- |
| A (6 trek, ccMixter) | 0.50 | 0.80 | 0.50 | 0.80 |
| B (4 trek, Finn Anderson) | 0.14 | 0.33 | 0.29 | 0.67 |

`--sample-times` uzun fayldan ko'rsatilgan vaqtlarda `--clip-len` (default 240 s) parchalarni kesib, har birini alohida tahlil qiladi. Fayl oxiridan oshgan vaqt ogohlantirish bilan tashlanadi. Parchalar orasidagi feature masofa va `style_groups` proksi. `profile --per-track` har parchani bitta birlik sifatida oladi.

### Metrika turlari

| kind | Hisobotdagi belgi | Misollar |
| --- | --- | --- |
| measured | [O'LCHANGAN] | LUFS, LRA, true peak, RMS, dinamik diapazon, chastota bandlari, centroid, rolloff, flatness, pauza, stereo |
| estimate | [TAXMINIY] | past-o'tkazish chegarasi, tempo, tonallik, shovqin pog'onasi, stemlar |
| proxy | [PROKSI] | onset tezligi, IOI, takrorlanish |
| probabilistic | [EHTIMOLLIK] | Essentia janr, asbob, kayfiyat teglari |

Har bir sonli metrikada `value`, `unit`, `kind`, `note` bor. `report.md` oxirida **Cheklovlar** bo'limi bor: tempo/tonallik ambientda ishonchsiz, stemlarda artefakt, teglar ehtimollik, tahlil "nega yoqadi"ga javob bermaydi.

