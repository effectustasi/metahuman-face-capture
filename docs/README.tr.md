# facecap

Webcam veya video → ARKit blendshape skorlari → MetaHuman face board → RigLogic.

Ust klasordeki "film karesi secici" projesinin devami sayilabilir: o araç filmden yuz
rekonstruksiyonu icin kare seciyordu, bu araç ortaya cikan MetaHuman'i oynatiyor.

## Mimari — iki surec

MediaPipe **Blender'in python'una kurulmaz**; protobuf/numpy catismasi Blender'i bozar.

```
  [detector/.venv  Python 3.11]          [Blender 5.0  Python 3.11]
   webcam / video                         meta_human_dna 0.5.4
        |                                          ^
   mediapipe FaceLandmarker                        |
        |  51 ARKit skoru + kafa matrisi           |
        |                                   blender_addon/charface
        +-- offline: takes/*.jsonl  ------->  charface.bake_take
        +-- canli:   UDP :11111    -------->  charface.live
                                                   |
                        core/  (mapping, filtre, take semasi — stdlib, ortak)
```

`core/` iki tarafin da import ettigi ortak koddur; esleme mantigi tek yerde durur.

## Zincir nasil calisiyor

```
ARKit pozu ──(Epic posemap)──> raw CTRL_expressions.* ──(DNA GUI→raw tersi)──> GUI ekseni
                                                                                    │
                                                     face board pose_bone.location ─┘
                                                                                    │
                                                        RigLogic ──> kemik + shape key + maske
```

Raw kontrole **dogrudan yazilmaz** — addon her degerlendirmede `mapGUIToRawControls()` cagirip
raw degerleri GUI'den yeniden uretiyor, yazdigin deger ayni karede eziliyor. Ayrinti ve
dosya:satir referanslari: [docs/api-notes.md](docs/api-notes.md).

## Kurulum

### 1. Detector — ✅ kurulu

Python 3.11.9, mediapipe 0.10.35, opencv 4.11 `detector/.venv` icinde; model
`detector/models/face_landmarker.task` (3.7 MB) indirildi.

Sifirdan kurmak gerekirse:

```bash
py -3.11 -m venv detector/.venv
detector/.venv/Scripts/python -m pip install -r detector/requirements.txt
curl -L -o detector/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 2. Test ortami

```bash
py -m venv .venv
.venv/Scripts/python -m pip install pytest
```

### 3. Blender addon

`blender_addon/charface` klasorunu Blender'a addon olarak kur (veya
`scripts/addons` altina symlink). Character DNA Pro'nun **ustune** biner, onun yerine gecmez.

## Kullanim

### Arayuz

**`facecap.bat`** -- cift tikla. Kamera secimi, Blender'a gonderme, kayit ve
canli kanal barlari tek pencerede.

```
+-------------------------+---------------------+
|                         |  Kamera  [0] [Tara] |
|       onizleme          |  UDP 127.0.0.1:11111|
|                         |  [x] Onizleme       |
|                         |  [x] Filtre         |
|                         |  [ ] Ham landmark   |
|                         |  [ ] Dosyaya kaydet |
|                         |     [ BASLAT ]      |
|                         |  yuz VAR | 28 fps   |
+-------------------------+---------------------+
| Agiz / cene                                   |
|  mouthSmileRight ##########      0.96 (0.96)  |
|  jawOpen         ###             0.27 (0.28)  |
| Goz                                           |
|  eyeBlinkRight   ######          0.64 (0.77)  |
| Kas / burun                                   |
|  browDownRight   ###             0.37 (0.76)  |
+-----------------------------------------------+
```

**Kanal barlari kalibrasyonun can damari.** Bir ifade yaptiginda hangi kanalin
ne kadar tetiklendigini gormeden hangi kanalin olu oldugunu anlamak mumkun
degil; eski .bat dosyalari calisiyordu ama hicbir sey soylemiyordu.

Barlar HAM degeri degil **notrden sapmayi** gosteriyor, ve bolge bolge. Tek bir
"en yuksek 8" listesi ise yaramiyordu: MediaPipe notr yuzde bile goz/kas
kanallarini yuksek veriyor. Gercek cekimde, kullanici ifadesiz dururken:

    eyeSquintLeft 0.473   browDownRight 0.395   browDownLeft 0.305
    eyeLookUpLeft 0.301   eyeBlinkLeft  0.290   eyeSquintRight 0.267

Tam sekiz kanal surekli 0.2 uzerinde; agiz kanallari notrde 0.00 olmasina ragmen
listeye hic giremiyordu. Goz de yaniltiyordu: `eyeBlinkLeft` 0.290'da basliyor,
yani 0.87'lik ham deger aslinda 0.58'lik bir kirpma. Parantez icindeki ham deger
kanalin doyuma gidip gitmedigini gormek icin duruyor.

Notr, baslattiktan sonraki ilk 30 kareden ogreniliyor -- o sirada **ifadesiz
dur**. Isik veya mesafe degisirse `Notru Yeniden Al`.

Onizleme 15 fps'e kisili (her kareyi Tk'ye cizmek yakalamadan pahali).
`cv2.imshow` kullanilmiyor: HighGUI kendi olay dongusunu isletiyor ve Tk
mainloop ile yan yana Windows'ta kilitlenebiliyor.

`facecap.bat` uc eski .bat dosyasinin isini de yapiyor (`canli_baslat`,
`kamera_tara`, `cozucu_cekim`). `iphone_test.bat` ayri duruyor.

### Komut satiri

Video dosyasindan cekim uret:

```bash
detector/.venv/Scripts/python detector/detect.py video --input cekim.mp4 --out takes/cekim.jsonl
```

### Keyframe yazma

Iki yol var. **Canli kayit** normal kullanim icin, bake tekrar denemek icin.

#### 1. Canli kayit (timeline'a dogrudan)

1. `facecap.bat` → BASLAT
2. Blender: **Canli Yuz Yakalama**
3. Oynatma kafasini istedigin kareye koy
4. **`Timeline'a Kaydet`** tusuna bas (kirmizi olur), oyna, tekrar bas

Keyframe'ler sen oynarken yaziliyor, oynatma kafasi ilerliyor. Panel kac kare
yazildigini canli gosteriyor.

Zamanlama DUVAR SAATINDEN, paket sayacindan degil: paketler 30 fps geliyor,
sahne genelde 24 fps. Paket basina bir kare yazmak animasyonu %25 yavaslatirdi.
Ayni sahne karesine birden fazla paket dustugunde eksenlerin MUTLAK DEGERCE en
buyugu tutuluyor -- sonuncuyu almak 89 ms suren goz kirpma tepesini
kaybettiriyor (headless testte dogrulandi: 30 paket -> 25 keyframe, tepe 0.9
korundu).

Sahne araligi sadece GENISLETILIYOR, daraltilmiyor; mevcut animasyonunu
kirpmasin diye. Gercek aralik panelde yaziyor.

#### 2. Kayittan bake

Cekimi dosyaya alip sonra islemek istersen (ayni cekimi farkli kalibrasyonla
tekrar denemek icin faydali):

1. Arayuzde **Dosyaya kaydet** isaretli, BASLAT, oyna, DURDUR
2. Blender: N-panel → **charface** → *Cekimi Face Board'a Bake Et* → .jsonl sec

Secenekler:

| | |
|---|---|
| `Baslangic Karesi` | animasyonun basladigi sahne karesi |
| `Notr Kalibrasyonu` | ilk 30 kareyi notr kabul edip cikarir. **Acik birak** |
| `Tek Kanal` | sadece bu ARKit kanali (once `jawOpen` ile dogrula) |
| `One-Euro Filtre` | kayitta filtre kullanmadiysan burada ac |
| `Sol/Sag Aynala` | kamera goruntusunde kullanicinin solu karakterin saginda |

**Cekim zaman damgasina gore sahne fps'ine oturtuluyor**, kare kareye 1:1
degil. Olculdu: webcam 30 fps veriyor, Blender'in varsayilan sahnesi 24 fps --
1:1 yazim 37 saniyelik cekimi 46 saniye yapiyor, %25 yavas. Kare araligi da
sabit degil (12-50 ms olculdu). Ayni sahne karesine birden fazla cekim karesi
duserse kanallarin en yuksegi aliniyor; sonuncuyu almak 89 ms suren goz kirpma
tepesini kaybettiriyor. Mantik `core/take.py:plan_scene_frames`, testleri
`tests/test_take.py`.

Bake bitince operator kac kare yazdigini ve kacini birlestirdigini raporluyor.
Sahnenin `frame_start`/`frame_end` araligi da otomatik ayarlaniyor.

**Ilk denemede `Tek Kanal` alanina `jawOpen` yaz.** Cene aciliyorsa zincir saglamdir, alani
bosaltip tekrar bake et. Hepsini birden baglarsan hangi katmanin bozuk oldugu anlasilmaz.

Canli:

```bash
detector/.venv/Scripts/python detector/detect.py live --udp 127.0.0.1:11111 --preview
```

Blender'da *Canli Yuz Yakalama* (ESC ile cikis).

### Kafa kalibrasyonu

N-panel > charface > **Kafa Kalibrasyonu** alt paneli. Ayarlar SAHNEDE durur, yani
canli yayin devam ederken degistirilebilir ve aninda etki eder -- operator ozelligi
olsaydi her denemede kapat/ac gerekirdi.

Panelde:

- **Canli okuma**: yaw / pitch / roll derece cinsinden. Kalibrasyonun can damari --
  hangi acinin ne kadar gittigini gormeden hangi eksenin yanlis oldugu tahmin isi olur.
  Ayrica paket sayaci ve "yuz VAR/YOK".
- **Notru Yeniden Al**: notr ilk 30 kareden kuruluyor; o sirada duzgun oturmuyorsan
  kafa surekli kaymis durur. Duz bak, bas, o an notr olur.
- **Aci basina esleme**: her aci icin hedef kemik ekseni + `Ters` + `Kazanc`.
  Kazanc 0 o aciyi kapatir, 2 iki katina cikarir.
- **Eksen testi**: X / Y / Z tuslari kafayi o eksende dondurur, hangisinin ne yaptigini
  gormek icin. `Sifirla` notre dondurur.

### Kanal araligi (ROM cekimi)

Kanallar 1.0'a ulasmiyor -- gercek iPhone kaydinda olculdu: `eyeBlink` **0.917**'de
doyuyor, `mouthSmile` 0.865. Rig 1.0'da tam acilacak sekilde kurulu oldugundan **goz
hicbir zaman tam kapanmiyor**.

Yuz Kalibrasyonu panelinde:

1. `Aralik Ogreniliyor` tusuna bas
2. Tum ifadeleri **sonuna kadar** yap (goz kirp, agzi ac, gulumse, kaslari kaldir...)
3. Tusu kapat
4. **`Araliklari Kaydet`** -- yoksa Blender kapaninca kaybolur

Kaydedilen profil `//charface_profile.json` (sahnenin yaninda; sahne kaydedilmemisse
proje kokunde). Canli yakalama baslarken varsa otomatik yukleniyor.

Kaydetme mevcut profili ezmiyor: elle ayarladigin kazanc/olu bolge korunuyor,
sadece `input_max` guncelleniyor.

Kalibrasyon sirasi: once eksen testiyle hangi eksenin ne yaptigini ogren, sonra aci
eslemesini kur, sonra `Ters` ile yonu duzelt, en son `Kazanc` ile miktari ayarla.

Operator sadece UDP dinler; paketi kimin gonderdigini bilmez. Kaynak uc secenek:

| Kaynak | Durum |
|---|---|
| `scripts/replay_take.py` | ✅ calisiyor — kayitli cekimi gercek zamanli basar, sahte detector |
| `detector/detect.py live` | ✅ webcam — kurulu, gercek fotografta dogrulandi, canli akis henuz denenmedi |
| iPhone Live Link Face | ✅ **destekleniyor** — cozucu 546 gercek pakete karsi dogrulandi |

### iPhone (Live Link Face)

Operator paketi **otomatik taniyor**: JSON ise webcam/replay, ikili ise Live Link Face.
Ayar yok, ayni tus ayni port.

iPhone'un webcam'e gore artisi:

| | webcam (MediaPipe) | iPhone (ARKit) |
|---|---|---|
| kanal | 51 | 52 |
| `tongueOut` | yok | **var** |
| `mouthClose` | yok (elle yazildi) | **gercek olcum** |
| kararlilik | RGB tahmini | TrueDepth sensoru |
| goz yonu | blendshape'ten | ayri 6 kanal |

Kurulum:

1. App Store > **Live Link Face** (Epic Games, ucretsiz). iPhone X ve sonrasi gerekiyor.
2. Sol ust dislii > **Live Link** > **Add Target** > PC'nin IP'si, port **11111**
3. Telefon ve PC ayni agda olmali
4. Dogrula: `iphone_test.bat` (ya da `python scripts/llf_probe.py`)
5. Blender'da *Canli Yuz Yakalama*

**Paket duzeni Epic tarafindan resmi belgelenmemis**, topluluk kaynaklari da celisiyor.
Cozucu gercek bir iPhone kaydindan alinan **546 pakete** karsi kosuldu: hepsi cozuldu,
yedek yola hic dusulmedi. Ayrinti: [docs/api-notes.md](docs/api-notes.md) §12.

`llf_probe.py` hangi yolun kullanildigini yine de soyluyor; `scan` derse duzen senin
iOS surumunde farkli demektir.

## Sahne agirsa (Sahne Performansi paneli)

N-panel > charface > **Sahne Performansi**. Canli yakalamayla ilgisi yok, her
zaman kullanilabilir.

### Sac particle'i ekleyince kasiyorsa

Sebep sudur: sac emitter mesh'e bagli ve mesh RigLogic ile surekli deforme
oluyor; her deformasyonda tum teller yeniden hesaplaniyor. Asil yuk `count`
alaninda gorunmuyor, **cocuk particle'larla carpiliyor**:

    2000 tel x 25 cocuk = 50.000 tel, her biri 2^3 segment

Panel bu carpimi hesaplayip gosteriyor. **`Saci Hafiflet`** viewport degerlerini
dusuruyor (`display_percentage` 10, `child_percent` 0, `display_step` 1) --
olculdu: 50.000 tel -> 200. **Render etkilenmez**; `render_step` ve
`rendered_child_count` ayri alanlar, onlara dokunulmuyor.

Yedek particle settings datablock'unda saklaniyor, dosyayi kaydedip acsan da
`Geri al` birebir eski degerlere donuyor.

### RigLogic otomatik degerlendirme

Addon'un `evaluate_dependency_graph` bayragi acikken RigLogic **her depsgraph
guncellemesinde** kosuyor (`meta_human_dna/rig_logic.py:37`) -- obje secmek,
kare degistirmek, particle duzenlemek, hepsi 800+ kemikli rigi yeniden
hesaplatiyor. Addon bunu sadece bake/import sirasinda programatik kapatiyor,
hicbir arayuzde acik degil.

Panelden kapatabilirsin; yuz kendiliginden guncellenmez, `Simdi Degerlendir`
ile elle tetiklersin. Sac/modelleme isi yaparken kapali tutmak mantikli.

### Canli yakalamada

**Onizleme Hz** viewport guncellemesini kisar. **Kaydi etkilemez** -- keyframe'ler
gelen degerlerden yaziliyor, ekranda gorunenden degil; viewport 15 Hz'e dusse de
animasyon tam hizda kaydediliyor. Panelde olculen degerler var: rig kac ms,
uygulama kac ms, bu hizda saniyede kac kare yetisiyor.

## Klasorler

| | |
|---|---|
| `core/` | Ortak mantik: esleme, one-euro filtre, kalibrasyon, take semasi. Stdlib-only. |
| `detector/` | MediaPipe tarafi. Ayri venv. |
| `blender_addon/charface/` | Bake ve canli operator'ler + panel. |
| `mapping/arkit_to_mh.json` | Uretilen esleme tablosu. Elle duzenlenmez. |
| `mapping/_generated/` | DNA dokumu, build raporu, introspection ciktisi. |
| `gui.py` | Tk arayuzu. detector/.venv ile kosar; `facecap.bat` baslatir. |
| `scripts/` | Faz 0 introspection, DNA dokumu, tablo uretimi, fixture, UDP replay/monitor, addon kurulumu. |
| `tests/` | Blender gerektirmeyen birim testleri. |

## Cozucu (deneysel)

ARKit eslemesi kanal kanal calisir: "jawOpen 0.42 gordum, cene kontrolune 0.42 yaz".
Cozucu tersini yapar: "bu landmark dizilimini hangi kontrol kombinasyonu uretir".
MetaHuman Animator'un yaptigi is bu.

**Ileri model karakterin KENDI DNA'sindan cikarilmali.** Varsayilan `head.dna` ile
cikarildiginda Blender dunyasina oturtma olcegi 0.010565 cikiyordu (0.01 olmali,
%5.7 sapma), artik hata 2.75 mm. Ayni olcum Icardi DNA'siyla 0.009994 ve 0.47 mm --
5.8 kat iyi. Baska bir kafayi modellemek sessizce olan, ama her seyi bozan hata.

```bash
blender --background --python scripts/05_extract_behavior.py -- <karakterin head.dna> mapping/_generated/behavior.npz
blender --background --python scripts/04_extract_face_model.py -- <karakterin head.dna> mapping/_generated/face_model_landmarks.npz --vertices mapping/_generated/correspondence/correspondence.json
blender --background --python scripts/06_dump_reference.py -- <karakterin head.dna> mapping/_generated/riglogic_reference.npz 32
```

Sonra landmark'li cekim al ve coz:

```bash
detector/.venv/Scripts/python detector/detect.py live --out takes/deneme.jsonl --landmarks --preview
python scripts/10_solve_take.py --take takes/deneme.jsonl
```

`--landmarks` 478 ham landmark'i da yazar (kareyi ~30 kat buyutur, o yuzden
varsayilan degil). Cekimin **ilk 30 karesi ifadesiz** olmali; notr oradan
ogreniliyor ve kimlik farki ona gore iptal ediliyor.

### Karsilik NOTR ve ONDEN render'dan kurulmali

Ilk karsilik ~17 derece donuk kafayla kuruldu. MediaPipe daha cok landmark
buluyordu (473 -> 478) ama uzak yanaktaki landmark'lar onden atilan isinla
YAKIN taraftaki vertexlere baglaniyordu. Olculen bedel:

| | donuk kafa | notr + teget filtresi |
|---|---|---|
| landmark | 478 | 447 |
| DNA -> Blender oturma artigi | 0.471 mm | **0.028 mm** |
| sol / sag landmark dengesi | 176 / 260 | **202 / 207** |

`Kafayi Render Et` artik kafa rotasyonunu ve tum yuz ifadesini render suresince
sifirliyor, sonra geri koyuyor -- karakter nasil duruyorsa dursun. Ayrica isinin
yuzeye ~70 dereceden yatik geldigi landmark'lar eleniyor (`facing < 0.35`).

**Landmark SAYISI hedef degil.** Az ve dogru baglanmis landmark, cok ve yanlis
baglanmistan iyidir.

### Gercek yuzle olculen durum (1114 karelik cekim)

| kare | ifade | taban (kontrol yok) | ARKit eslemesi | cozucu |
|---|---|---|---|---|
| 143 | gulumseme | 5.360 | 7.139 | **3.335** |
| 187 | cene acik | 1.488 | 3.893 | **0.670** |
| 246 | goz kirpma | 1.291 | 1.887 | **1.100** |
| 775 | dudak buzustur | 1.847 | 2.917 | **0.838** |

Cozucu ARKit eslemesini her karede ACIK ARA geciyor. ARKit tabandan bile kotu
cikiyor cunku agzi dogru buluyor ama yuzun geri kalanina sahte hareket basiyor:
gozlemde 0.7 mm oynayan 57 landmark'i 4.9 mm oynatiyor (kosinus -0.19).

**Ama cozucu de hareketin ancak %32-51'ini acikliyor.** Bu bir optimizasyon
sorunu DEGIL: duzenlilestirmeyi tamamen kapatip 2000 iterasyon ve 136 kontrolle
kosunca sonuc DAHA KOTU (3.438 vs 3.365 mm). Tavan modelin kendisinde.

Kalan acik: gercek bir insanin landmark hareketini belirli bir MetaHuman rigine
tasimak rijit hizalama + delta aktariminden fazlasini gerektiriyor. MetaHuman
Animator bunu oyuncuya OZEL bir DNA cikararak cozuyor (kimlik cozumu), sonra
ifadeyi o rig uzerinde cozup hedefe retarget ediyor. Bizde o adim yok.

Denendi ve ISE YARAMADI (tekrar denenmesin diye): pozu ifadeyle birlikte cozmek
(`solve(joint_pose=True)`) -- gulumseme karesinde 3.475 -> 4.026 mm, poz ve ifade
gradyani birbirini yiyor.

## Tabloyu yeniden uretmek

```bash
blender --background --python scripts/01_dump_dna_controls.py -- <head.dna> mapping/_generated/dna_controls.json
python scripts/02_build_mapping.py --posemap <posemap.json> --controls mapping/_generated/dna_controls.json
```

Kapsama: **51/51** MediaPipe kanali + `tongueOut` (iPhone icin). `mouthClose` Epic'in
asset'inden turetilemedigi icin elle yazildi -- kaynagi ve guven sinirlari icin
[docs/api-notes.md](docs/api-notes.md) §8 ve §11.

## Durum

| Faz | Durum |
|---|---|
| 0 — Kesif | ✅ `docs/api-notes.md`, dosya:satir referansli |
| 1 — Detector | ✅ kurulu ve dogrulandi — gercek fotografta 51 skor + kafa quaternion'i uretti |
| 2 — Mapping | ✅ 51/51 kanal, 88 eksen; `mouthClose` elle yazildi (§11) |
| 3 — Blender uygulayici | ✅ canli yol gercek MetaHuman sahnesinde dogrulandi (UDP replay -> yuz oynadi); bake operator'u henuz denenmedi |
| 4 — Kalibrasyon/filtre | ✅ panelden canli ayarlanan kafa + yuz kalibrasyonu, grup kazanclari, uc bantli one-euro |
| 6 — Turetilmis kontroller | ✅ goz kapagi/bakis baglantisi, kapak baskisi, mikro-sakkad, kanal araligi (api-notes §15) |
| 7 — Turevlenebilir rig | ✅ RigLogic PyTorch'ta yeniden yazildi, 3.3e-07 hata, gradyan 4.5 ms (§14) |
| 8 — Landmark karsiligi | ✅ 477/478 landmark, baryantrik agirlikli, karakterin kendi kafasindan (§15) |
| 9 — Cozucu | ✅ iki asamali (L1 + yanlilik giderme), sentetik testte 0.002 kontrol hatasi (§16) |
| 10 — Kimlik ayirma | ✅ `core/observation.py` — gozlem/model notru farki iptal |
| 11 — Gercek yuzle deneme | ⚠️ calisiyor ama hareketin ancak %32-51'ini acikliyor (asagi) |
| 5 — Cikis | addon'un kendi bake/export'u kullanilir, ek kod yok |
