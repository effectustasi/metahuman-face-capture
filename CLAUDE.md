# facecap

ARKit yuz yakalama → MetaHuman face board. Ust klasordeki "film karesi secici"
projesinden **ayridir**, o projenin kurallari buraya uygulanmaz.

## Ortam

**Kullanicinin calistigi surum: Blender 5.0.** Kurulu Blender'lar ve addon durumlari farkli,
karistirma:

| Blender | Python | DNA addon | Sahne property |
|---|---|---|---|
| **5.0 ← kullanilan** | **3.11.13** | `meta_human_dna` **0.5.4** (legacy addon, etkin) | `scene.meta_human_dna` |
| 5.1 | 3.13.9 | Character DNA Pro **0.8.7** (extension, etkin) | `scene.character_dna_pro` |
| 4.4 / 5.2 | — | yok | — |

`docs/api-notes.md` cogunlukla **0.8.7** kaynagindan cikarildi; 0.5.4 karsiliklari §9'da.
charface ikisini de destekler (`SCENE_PROPERTY_CANDIDATES`, `INSTANCE_LIST_CANDIDATES`).

- charface Blender 5.0'a **kurulu ve etkin**: `scripts/addons/charface_loader.py` yukleyici
  dosyasi projeye isaret eder, kod kopyalanmaz. Kaynagi duzenle → Blender'i yeniden baslat.
  Yeniden kurmak/kaldirmak: `python scripts/install_addon.py --version 5.0 [--uninstall]`
- Blender'in python'una **ASLA** pip ile paket kurma. MediaPipe ayri venv'de (`detector/.venv`).
- Testler icin `facecap/.venv` (Python 3.14): pytest + numpy + **torch 2.11.0+cu128**.
  `requirements-dev.txt`e bak. torch ust projeden ODUNC ALINMIYOR, burada kurulu —
  `core/riglogic_torch.py` facecap'in gercek bagimliligi ve ust projenin venv'i
  degisirse sessizce bozulurdu. Blackwell (sm_120) icin **cu128 index'i sart**.
- `core/` cogunlukla stdlib; iki istisna numpy kullaniyor (`facemodel.py`) ve torch
  (`riglogic_torch.py`). Blender numpy'i paketiyle getiriyor, torch'a Blender'da
  ihtiyac yok — cozucu Blender disinda kosuyor.
- Detector venv'i **Python 3.11** olmali: MediaPipe 3.13/3.14 wheel'i yok, ayrica addon'un DNA
  binding'leri py311/py313 olarak derlenmis. **Makinede su an sadece 3.14 kurulu, 3.11 gerekiyor.**

## Kurallar

- **Addon API adlarini EZBERDEN YAZMA.** Once `docs/api-notes.md`'ye bak, yoksa kurulu addon
  kaynaginda grep'le. Bulamazsan dur ve sor.
- **Dogru kopyayi oku.** Diskte uc kopya var:
  - ✅ `5.0/scripts/addons/meta_human_dna` — **0.5.4, kullanicinin fiilen kullandigi surum**
  - ✅ `5.1/extensions/polyhammer_com/character_dna_pro` — 0.8.7, kayitli repo
  - ❌ `5.1/extensions/polyhammer/character_dna_pro` — kayitsiz kopya, bayat
  - ❌ `5.1/scripts/addons/meta_human_dna` — 0.5.4'un ikinci kopyasi
  Bir davranis 5.0'da mi 5.1'de mi dogrulandi, her zaman belirt. **API'ler farkli.**
- **Isimler tek kaynaktan gelir.** ARKit/GUI kontrol isimleri `mapping/arkit_to_mh.json`'dan.
  Sahne property adi yoklanir (`character_dna_pro`, `character_dna`, `meta_human_dna`),
  sabitlenmez. Instance listesi adi da (`rig_instance_list` / `rig_logic_instance_list`).
  Kodda kontrol ismi string literal'i yok.
- **Raw kontrol yazma, face board (GUI) yaz.** Gerekcesi `docs/api-notes.md` §4 —
  `mapGUIToRawControls()` raw degerleri ayni karede eziyor.
- Operator icinde `bpy.ops` zincirleme cagirma; dogrudan veri API'si kullan.
- **Bake sirasinda her karede sahne guncelleme.** `evaluate_dependency_graph` bayragini `False`
  yap, tum kareleri yaz, sonunda `True` yapip tek sefer `evaluate()` cagir (api-notes §5).
- Tek kanalla dogrula, sonra genislet. `charface.bake_take` operator'unde `single_channel`
  alani bunun icin: once `jawOpen`, dogrula, sonra ac. 51 kanali birden baglarsan hangi
  katmanin (tespit / esleme / eksen yonu / olcek) bozuk oldugunu ayirt edemezsin.

## Degisiklik sonrasi

```bash
.venv/Scripts/python -m pytest tests/ -q
```

Blender tarafina dokunduysan ek olarak headless smoke:

```bash
blender --background --python scripts/00_introspect.py -- /tmp/introspect.json
```

## Bilinen tuzaklar

- MediaPipe `categories[0] == "_neutral"`, atlanir → 51 skor (52 degil).
- MediaPipe **tongueOut uretmez**, iPhone uretir. Kaynak ayrimini `tongueOut`un varligindan
  yapiyoruz (panelde "kaynak" satiri).
- Live Link Face kanal sirasi **gercek CSV basligindan** alindi, ezberden degil:
  `JawRight` `JawLeft`ten ONCE geliyor. Ezberden yazilirsa cene sag/sol takla atar.
- Live Link paket duzeni resmi belgelenmemis; `core/livelink.py` yapisal cozum tutmazsa
  sondan tarayan yedek yola dusuyor. **Gercek telefonla dogrulanmadi.**
- `cheekPuff` gurultulu.
- **MediaPipe notr yuzde bile goz/kas kanallarini yuksek veriyor.** Gercek cekimde olculdu
  (kullanici ifadesiz dururken): `eyeSquintLeft` **0.473**, `browDownRight` 0.395,
  `eyeBlinkLeft` **0.290**, `eyeSquintRight` 0.267. Agiz kanallari ise notrde 0.00.
  Ham degeri esik/siralama icin kullanan her sey bu yuzden bozuk calisir -- notrden
  SAPMA kullan. `gui.py` barlari ve `core/calibration.py` olu bolgesi bunun icin.
- `mouthClose` eslemesi Epic'in asset'inden TURETILMEDI, anlamdan elle yazildi
  (`scripts/02_build_mapping.py:AUTHORED`). Tabloyu yeniden uretirken kaybolmasin.
- ARKit `...Left/Right` ↔ MetaHuman raw `...L/R`. Tablo **MediaPipe isimleriyle** anahtarlandi,
  boylece detector→mapping arasinda cevirme katmani yok. Bozmayin.
- `eyeBlinkLeft` ve `eyeWideLeft` **ayni** GUI eksenini ters yonde suruyor. Toplanmalari sart,
  `max` alinirsa yanlis. (api-notes §7)
- Sol/sag ayna: kamera goruntusunde kullanicinin solu karakterin sagidir. `mirror` bayragi var.
- **MediaPipe `tongueOut` URETMEZ.** Dil hareketi bu yolla imkansiz, eksik ozellik degil model
  siniri. Sadece iPhone ARKit veriyor, o da desteklenmiyor (README).
- Kafa rotasyonu face board'a degil `head` kemigine yazilir ve **bagil** olmalidir (api-notes §10).
- **Rig'e dokunan operator'de `bl_options`'a UNDO koymadan once dusun.** Redo panelinden ozellik
  degistirmek Blender'i "geri al + yeniden calistir" dongusune sokuyor; undo sahne
  datablock'larini yeniden kuruyor ama addon RigLogic'in C++ nesnelerini python sozlugunde
  onbellege aliyor. Bayat isaretci -> sonraki `evaluate()` sureci **sert dusuruyor**, python
  hatasi gelmiyor. Tani/deneme operator'lerinde UNDO kullanma, secenekleri panelde ayri tuslara
  bol. Bake gibi geri alinabilir olmasi sart olanlarda UNDO kalir ama `evaluate()` sarilir.
- **Sadece Blender 5.1'de:** Character DNA Pro kayit sirasinda `docker` calistirmaya calisiyor
  (`editors/raw_control_editor/nls_worker/docker/image.py:201`), makinede Docker yok →
  konsolda `WinError 2` traceback'i. Zararsiz, bizim kodumuzla ilgisi yok. 5.0'da olmaz.
- 0.5.4 binding'leri dizileri **numpy skalari** olarak donduruyor; json'a yazmadan once
  `int()`/`float()`'a cevir (`scripts/01_dump_dna_controls.py`).
- Blender konsolunda `biome_reader` unregister hatasi da baska bir addon'dan, ilgisiz.
- **Bash'te calisma dizini kayiyor.** Bu oturumda uc kez bir onceki komuttan kalan
  dizinde calisildi ve `./.venv` UST PROJENIN venv'ini gosterdi: bir kez test dosyasi
  yanlis projeye yazildi, bir kez `pip install` yanlis ortama gitti ("already satisfied"
  deyip sessizce hicbir sey yapmadi). **Venv ve script yollarini MUTLAK yaz.**
- **Modal operator soket sizdirir.** Blender bir modal'i olduren her yolda `cancel()`
  cagirmiyor; UDP soketi acik kalip portu tutuyor ve sonraki baslatma `WinError 10048`
  aliyor. Cozum modul duzeyinde `_LIVE_SESSION` izi + `charface.stop_live` operator'u +
  `unregister()` icinde kapatma. Ayni deseni yeni modal operator yazarken tekrarla.

## Bu proje ne zaman YANLIS arac

Prodüksiyon kalitesi yuz animasyonu hedefliyorsan MetaHuman Animator ile cekip Unreal Level
Sequence'ten export edilen face board animasyonunu addon'un Animation panelinden ice aktarmak
**sifir kodla** daha iyi sonuc verir. Bu pipeline canli/etkilesimli senaryolar ve iPhone'suz
calismak icin mantikli.
