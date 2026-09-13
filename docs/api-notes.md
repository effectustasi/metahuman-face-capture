# DNA addon'u — API notlari (Faz 0)

Bu dosya **okunarak** cikarildi, tahmin yok. Her iddianin yaninda dosya:satir var.

> **§1–8 Character DNA Pro 0.8.7 (Blender 5.1) kaynagindan cikarildi.**
> Kullanicinin fiilen calistigi surum **Blender 5.0 + meta_human_dna 0.5.4**; mimari ayni,
> isimler farkli. Karsilik tablosu ve 5.0'da yapilan dogrulamalar: **§9**.

| | |
|---|---|
| Addon | Character DNA Pro **0.8.7** |
| Modul | `bl_ext.polyhammer_com.character_dna_pro` |
| Kurulum yolu | `%APPDATA%\Blender Foundation\Blender\5.1\extensions\polyhammer_com\character_dna_pro` |
| Blender | 5.1.2, Python 3.13.9 |
| Dogrulama tarihi | 2026-08-18 |

Asagida `<addon>` = yukaridaki kurulum yolu.

> **Dikkat:** Diskte baska kopyalar da var. `5.1/extensions/polyhammer/character_dna_pro`
> kayitli repo **degil**, bayat. `scripts/addons/meta_human_dna` ise 0.5.4 — eski isim,
> **API'si farkli** (bkz. §9). Grep atarken yanlis kopyayi okumak kolay; 5.1'de kayitli
> repo adi `polyhammer_com`.

---

## 1. Rig instance'a erisim

`RigInstance` bir `PropertyGroup`, sahnedeki listede duruyor:

- Sinif: `<addon>/rig_instance.py:314`
- Liste: `<addon>/properties.py:456` → `rig_instance_list`
- Aktif indeks: `<addon>/properties.py:457` → `rig_instance_list_active_index`

Sahne property adi **iki tane**: `constants.py:313` `SIBLING_EDITIONS = {"character_dna": "character_dna_pro", ...}`
ve `properties.py:519-521` her iki ismi de `bpy.types.Scene` uzerine bagliyor. Introspection
ciktisi ikisinin de var oldugunu dogruladi. Bu yuzden kodda tek isim sabitlenmez:

```python
for name in ("character_dna_pro", "character_dna"):
    properties = getattr(context.scene, name, None)
    ...
```

## 2. Face board

`instance.face_board` bir armature **Object**'i (`rig_instance.py:157-168`, animation_data/action
uzerinden kullaniliyor). GUI kontrol isimleri dogrudan bu armature'un pose bone adlaridir.

## 3. Degerlerin yazildigi yer

`update_head_gui_control_values(override_values)` — `<addon>/rig_instance.py:1508`

```python
for index in range(self.head_dna_reader.getGUIControlCount()):
    full_name = self.head_dna_reader.getGUIControlName(index)   # "CTRL_C_jaw.ty"
    control_name, axis = full_name.split(".")
    axis = axis.rsplit("t", -1)[-1].lower()                      # "ty" -> "y"
    if override_values:
        value = override_values.get(control_name, {}).get(axis)
        ...
    else:
        pose_bone = self.face_board.pose.bones.get(control_name)
        value = getattr(pose_bone.location, axis)
    self.head_instance.setGUIControl(index, value)
```

Iki giris yolu var:

1. **Pose bone location yazmak** (override yok) — addon kemikten okur. `charface` bunu kullanir:
   keyframe'lenebilir, viewport'ta gorunur, addon'un kendi bake/export akisiyla uyumlu.
2. `override_values` gecmek — addon'un kendi yorumu: *"can be used for baking the values to an action"*.
   Kemige dokunmadan degerlendirme yapar.

Ters yon de var: `apply_gui_controls_to_face_board()` — `rig_instance.py:1575` RigLogic'teki GUI
degerlerini face board kemiklerine geri yazar.

## 4. Neden raw kontrol yazmiyoruz — planin B secenegi burada calismiyor

Plan "Raw Control'e yaz, Unreal'in mapping'iyle birebir ayni olur" diyordu. Kaynakta durum tersine:

**(a) `update_head_raw_control_values` ifade kontrollerini zaten atliyor** — `rig_instance.py:1456`:

```python
control_name, axis = full_name.split(".")
if not axis.startswith("q"):    # <-- sadece quaternion eksenleri
    continue
```

DNA'da raw kontrol isimleri `CTRL_expressions.browDownL` formatinda; `split(".")` sonrasi
axis = `browDownL`, `q` ile baslamiyor → **atlaniyor**. Bu fonksiyon sadece RBF/kemik surucusu
quaternion kontrolleri icin. 263 raw kontrolun 251'i `CTRL_expressions.*`, yani buradan yazilmiyor.

**(b) Yazilsa bile ayni karede eziliyor** — `evaluate()` (`rig_instance.py:2000`) sirasi:

```
update_head_gui_control_values()      # GUI degerleri
head_manager.mapGUIToRawControls()    # <-- raw kontroller GUI'den YENIDEN uretilir
update_head_raw_control_values()      # sadece evaluate_rbfs acikken, sadece .q*
head_manager.calculate()
```

`setRawControl()` ile yazilan `CTRL_expressions.*` degeri `mapGUIToRawControls()` tarafindan
uzerine yazilir. Raw yazmak icin `evaluate()` akisini tamamen bypass etmek gerekirdi — o zaman da
kemik/shape key/texture mask guncellemelerini elle yapmak zorunda kalirdik.

**Sonuc: face board (GUI) dogru giris noktasi.** Planin A secenegi.

## 5. Evaluation tetikleme ve tuzagi

`evaluate(component, dependency_graph)` — `rig_instance.py:2000`.

Basinda bir bayrak var, atlanmasi bake performansini oldurur:

```python
if window_manager_properties.evaluate_dependency_graph:
    window_manager_properties.evaluate_dependency_graph = False
    ...
finally:
    window_manager_properties.evaluate_dependency_graph = True
```

Bayrak `False` iken `evaluate()` **hicbir sey yapmadan doner**. Bake sirasinda bunu bilerek
`False` yapip 150 kareyi yazar, sonunda `True`ya cevirip **tek sefer** `evaluate()` cagiririz.
Aksi halde her `keyframe_insert` depsgraph handler'ini tetikler ve RigLogic bosuna yuzlerce kez kosar.

## 6. DNA okuma

`get_dna_reader()` — `<addon>/dna_io/misc.py:41`.

**API 0.5.4'ten 0.8.7'ye degisti**, ezberden yazilmaz:

| | 0.5.4 (eski) | 0.8.7 (guncel) |
|---|---|---|
| modul | `riglogic` | `dna` |
| open mode | `riglogic.OpenMode.Binary` | `dna.OpenMode_Binary` |
| data layer | `riglogic.DataLayer.All` | `dna.DataLayer_All` |

Binding'ler derlenmis `.pyd`: `<addon>/bindings/windows/x64/{py311,py313}/`. `bindings/__init__.py`
bir `IsolatedModuleLoader` meta_path hook'u kuruyor, bu yuzden `sys.path`e ekleyip
`import dna` demek **calismaz** — paket uzerinden gitmek gerekiyor:

```python
importlib.import_module(f"{addon_module}.bindings").dna
```

Modul adi da sabitlenmez, `addon_utils.modules()` ile aranir (bkz. `scripts/01_dump_dna_controls.py`).

## 7. DNA'dan olculen gercek sayilar

`scripts/01_dump_dna_controls.py` ciktisi (varsayilan MetaHuman head.dna):

| | |
|---|---|
| GUI kontrol | **174** — `CTRL_L_brow_raiseIn.ty` formatinda, face board kemik adlari |
| Raw kontrol | **263** — `CTRL_expressions.browRaiseInL` formatinda (251'i `CTRL_expressions.*`) |
| GUI→raw satiri | 259, surulen raw kontrol 251 |

GUI→raw parcali dogrusal: `raw = slope * gui + cut`, `gui ∈ [from, to]`.

- 243 raw kontrolun **tek** GUI girdisi var, cogu identity (slope=1, cut=0, 0..1)
- 8 tanesinin iki girdisi var: `mouthLipsSticky*Ph1/Ph2`, `neckSwallowPh1..4` — ayni slider
  uzerinde ucgen rampalar (faz kontrolleri). Tersi belirsiz, tabloya alinmiyor.
- 88 satir identity degil. **Onemli kalip:** bir slider'in negatif yarisi baska bir raw
  kontrolu suruyor:

  ```
  CTRL_expressions.eyeBlinkL  <- CTRL_L_eye_blink.ty  from=0..1   slope=+1
  CTRL_expressions.eyeWidenL  <- CTRL_L_eye_blink.ty  from=-1..0  slope=-1
  ```

  Yani **ARKit'in `eyeBlinkLeft` ve `eyeWideLeft` kanallari ayni ekseni ters yonde suruyor.**
  Bagimsiz yazilamazlar; toplanmalari gerekir. `core/mapping.py` bunu yapiyor, testi
  `tests/test_mapping.py::test_blink_ve_wide_toplanip_birbirini_goturuyor`.

## 8. Mapping tablosunun kaynagi ve guveni

`mapping/arkit_to_mh.json`, `scripts/02_build_mapping.py` ile uretiliyor:

```
ARKit pozu --(Epic posemap)--> raw CTRL_expressions.* --(GUI→raw tersi)--> GUI ekseni
```

Kaynak: `Dylanyz/ARKitRemap` reposundaki `PA_MetaHuman_ARKit_Mapping.posemap.json`.

**Guven notu:** bu tablo Epic'in asset'inin birebir kopyasi **degil**, editor python'uyla
**orneklenerek** cikarilmis (`poseIndexToTimeRule: i/(N-1)`, baseline cikarma). Sonuclari:

- posemap'teki 77 curve isminin **76'si** DNA'daki raw kontrollere birebir eslesti
  (`ctrl_expressions_browraiseinl` ↔ `CTRL_expressions.browRaiseInL`; kucuk harf + nokta→alt cizgi).
  Eslesmeyen tek isim `ctrl_expressions_tonguerolldown`, varsayilan DNA'da yok.
- `browLateralL` 0.031 agirlikla `cheekPuff`, `tongueOut` gibi alakasiz pozlarda gorunuyor —
  ornekleme artefakti. `epsilon=0.05` ile eleniyor (11 kayit).
- Pose asset'inde `Pose_4`..`Pose_17` isimli 14 girdi var, **hepsi ayni degerlere** ornekleniyor
  ve `tongueOut` ile ozdes. Ayirt edilememis ekstra dil pozlari; tabloya alinmiyor.
- **Kapsama: MediaPipe'in 51 blendshape'inin 50'si.** Eksik olan tek kanal `mouthClose`
  (posemap'te sifirdan farkli kaydi yok). `tongueOut` tabloda var ama MediaPipe uretmiyor —
  sadece iPhone Live Link icin.

Dogrulanmadi: agirliklarin Unreal'deki gorsel sonuca birebir esitligi. Epic'in asset'ini
Unreal'den T3D olarak export edip `--posemap` yerine onu vermek daha saglam olur; sema ayni kalir.

---

## 9. MetaHuman DNA 0.5.4 karsiliklari (Blender 5.0 — fiilen kullanilan surum)

Yukaridaki referanslar 0.8.7 icin. Kullanicinin Blender 5.0'inda **0.5.4** kurulu ve mimari ayni,
isimler farkli. `<addon50>` = `%APPDATA%\Blender Foundation\Blender\5.0\scripts\addons\meta_human_dna`.

| Kavram | 0.5.4 (Blender 5.0) | 0.8.7 (Blender 5.1) |
|---|---|---|
| Ana dosya | `rig_logic.py` | `rig_instance.py` |
| Instance sinifi | `RigLogicInstance` — `rig_logic.py:387` | `RigInstance` — `rig_instance.py:314` |
| Sahne property | `scene.meta_human_dna` (`MetahumanSceneProperties`) | `scene.character_dna_pro` / `character_dna` |
| Instance listesi | `rig_logic_instance_list` — `properties.py:213` | `rig_instance_list` — `properties.py:456` |
| GUI yazma | `update_head_gui_control_values()` — `rig_logic.py:1453` | `rig_instance.py:1508` |
| Raw yazma (sadece `.q*`) | `rig_logic.py:1401` | `rig_instance.py:1456` |
| `evaluate()` | `rig_logic.py:1925` | `rig_instance.py:2000` |
| DNA reader modulu | `riglogic.OpenMode.Binary`, `DataLayer.All` | `dna.OpenMode_Binary`, `DataLayer_All` |
| Binding'ler | `bindings/windows/amd64/` **sadece cp311** | `bindings/windows/x64/{py311,py313}/` |

Blender 5.0'in python'u **3.11**, 0.5.4'un binding'leri de cp311 → uyumlu.

Degismeyenler (charface'in dayandigi her sey):
- `instance.face_board` armature Object'i, GUI kontrol isimleri = pose bone adlari
- `evaluate()` icindeki sira: GUI → `mapGUIToRawControls()` → raw → `calculate()`,
  yani **§4'teki "raw yazma" gerekcesi 0.5.4'te de gecerli** (`rig_logic.py:1503`)
- `window_manager.<addon>.evaluate_dependency_graph` bayragi (§5)

Blender 5.0'da dogrulandi: charface yuklendi, `MetahumanSceneProperties` bulundu, WM bayragi
mevcut, mapping yuklendi (51 kanal / 64 kemik), `jawOpen=1.0 -> CTRL_C_jaw.y=0.999999`,
`eyeBlinkLeft+eyeWideLeft 0.5 -> 0.0`.

---

## 10. Kafa pozu — nereye yazilir

DNA'da 263 raw kontrolun **12'si** ifade degil, kemik quaternion'u:

```
neck_01.qx/qy/qz/qw   neck_02.qx/qy/qz/qw   head.qx/qy/qz/qw
```

Yani **surucu kemikler: `head`, `neck_01`, `neck_02`**. Bunlarin rotasyonu RigLogic'e
**girdi** -- boyun/kafa duzelticilerini tetikliyorlar (`update_head_raw_control_values`
sadece `.q*` eksenlerini isliyor, bkz. §4a).

Kritik olan: RigLogic 870 eklemin **hepsini** suruyor, `head` dahil. Buna ragmen bu uc kemige
yazmak guvenli, cunku addon bunlari bilerek atliyor:

```python
# 0.5.4  rig_logic.py:1630   (0.8.7'de rig_instance.py icinde ayni kontrol)
if name in self.head_driver_bone_names:
    continue
```

`head_driver_bone_names` (`rig_logic.py:1079`) tam olarak yukaridaki `.q*` isimlerinden
uretiliyor. Yani surucu kemiklere yazilan rotasyon **ezilmiyor**; diger 867 eklem RigLogic'in
cikardigi degerle yeniden yaziliyor.

`charface` bu yuzden kafayi `head` kemigine yaziyor (`resolve_head_bone`), ve yazmadan once
kemigin gercekten surucu listesinde oldugunu dogruluyor.

**Iki tuzak:**

1. **Mutlak degil bagil yaz.** MediaPipe kameraya gore mutlak yonelim veriyor; dogrudan
   yazilirsa karakterin kafasi kameranin onunde nasil duruyorsan oyle sabitlenir.
   `core/head.py:HeadTracker` ilk 30 kareden notr referans kurup farki uyguluyor.
2. **Eksen eslemesi rig'e gore degisiyor.** Tek dogru donusum yok; operator'de
   `head_axis` preset'i olarak disari acildi (varsayilan `X-ZY`). Kafa yanlis yone
   donuyorsa buradan degistirilir -- kodda sabit degil.

### Olculen eksen eslemesi

Gercek MetaHuman rig'inde iki gozlemle kuruldu (teorik degil):

| Gozlem | Sonuc |
|---|---|
| yaw, kemigin Z eksenine gonderildi -> karakter asagi/yukari bakti | **Z = pitch** |
| yaw->Y ve roll->X iken ikisi birbirinin yerine calisti | **Y = roll, X = yaw** |

Yani kafa kemiginin yerel eksenleri: **X = yaw, Y = roll, Z = pitch**.

Bu, Blender'in "kemik yerel Y ekseni boyunca uzanir" kuralindan cikan sezgiye uymuyor
(Y twist olsaydi yaw beklenirdi). MetaHuman kemikleri Unreal'den geldigi icin yonelim
farkli. Tam da bu yuzden kodda sabitlenmedi: `core/head.py:DEFAULT_MAPPING` sadece
baslangic degeri, gercek ayar sahnede (`scene.charface`) ve panelden degistiriliyor.

**Ayar sahnede saklandigi icin** koddaki varsayilan degistiginde mevcut .blend dosyalari
eski degeri tutmaya devam eder; `charface.reset_calibration` operator'u ikisini esitler.

Dogrulanmadi: isaretler (`Ters` bayraklari). Eksenler oturduktan sonra yon kontrolu gerekiyor.


---

## 11. Agiz/dudak kapsamasi

DNA'da agiz-dudak-cene-dil bolgesinde **110 GUI kontrolu** var; posemap'ten turetilen
tablo bunlarin **46'sini** suruyor. Kalan 64'u MediaPipe'in uretmedigi ince kontroller
(lipBite, thickness, sticky, cornerSharpness, tongue detaylari) -- bunlar 51 ARKit
kanalindan turetilemez, normal.

**Ama biri onemliydi ve eksikti:** `CTRL_L/R_mouth_lipsTogetherU/D`.

ARKit'te bunu `mouthClose` surer: "cene aciligindan BAGIMSIZ olarak dudaklarin kapanma
miktari". Bu olmadan "m", "b", "p" seslerinde cene hafif acikken dudaklarin kapanmasi
gerekirken agiz acik kaliyor -- konusmayi yapay gosteren birincil sebep.

Epic'in pose asset'inde `MouthClose` icin sifirdan farkli **hicbir kayit yok**, yani
turetilemedi. `scripts/02_build_mapping.py:AUTHORED` icinde **anlamdan elle yazildi**
(dort lipsTogether kontrolu, agirlik 1.0) ve tabloda `authoredChannels` olarak
isaretlendi. Epic'in gercek agirliklari farkli olabilir.

Kapsama artik **51/51**, kanal sayisi 84 -> 88.

### Ikinci sorun: pucker/funnel cakismasi

`mouthPucker` alti kontrolu suruyor ve bunlarin ikisi (`funnelU/D`) `mouthFunnel`
tarafindan da suruluyor. MediaPipe ikisini sik sik birlikte tetikliyor; toplama +
kirpma kuralinda toplam doyuma gidip agzi buzuk kilitliyor. Cozum kod degil ayar:
panelde `Dudaklar` kazanci ile kisiliyor.

### Ucuncu sorun: filtre

Tek one-euro ayari tum yuze uymuyor. Konusma saniyede birkac kez sekil degistiriyor;
kaslara uygun yumusatma dudaklari lapa yapiyor. `core/filters.py` artik uc grup
kullaniyor: blink (min_cutoff 5.0), agiz/cene (3.0), geri kalan (1.0).


---

## 12. Live Link Face paket duzeni — DOGRULANDI

Duzen Epic tarafindan resmi belgelenmemis ve topluluk kaynaklari **celisiyor**:
`lucasjinreal/llv` README'si isimlerin degerlerden SONRA geldigini soyluyor, digerleri ONCE.

`aelzeiny/Animoji` reposunda gercek bir iPhone'dan netcat ile yakalanmis ham UDP kaydi var
(`experimentation/livelink.udp`, 179 KB). Cozucu ona karsi kosuldu:

| | |
|---|---|
| Sirali cozulen paket | **546** |
| Basarisiz | **0** |
| Kullanilan yol | hepsi `structured` (yedek tarayiciya hic dusmedi) |
| Paket boyutu | 319 bayt (1 + 4+36 + 4+13 + 16 + 1 + 61*4) |

Yani **isimler degerlerden ONCE geliyor**, llv'nin README'si yanlis. Dogru duzen:

```
uint8         surum (6)
int32 + bytes cihaz kimligi (UUID, 36 bayt)
int32 + bytes konu adi
int32         kare numarasi
float         alt kare
int32         fps
int32         payda
uint8         kanal sayisi (61)
float[61]     degerler
```

Hepsi big-endian. `abclop99/livelinkface/pylivelinkface.py:218-242` ayni sirayi kullaniyor,
bagimsiz ikinci dogrulama.

Kayittan bir kare `tests/fixtures/livelink_frame.bin` olarak saklandi (cihaz UUID'si ve telefon
adi ayni uzunlukta yer tutucuyla degistirildi -- baskasinin kimligini repoya gommeyiz).
O karede `TongueOut = 1.0`, yani dil kanalinin gercekten calistigi da dogrulanmis oldu.


---

## 13. Kendi cozucumuz — ileri model

Hedef: MHA'nin yaptigi isi kendimiz yapmak. Ozu analysis-by-synthesis --
gozlemi (landmark/derinlik) rig'in parametre uzayinda en iyi aciklayan
kontrol vektorunu bulmak. Isin zor kismi olan **ileri model** bizde zaten
var: DNA + RigLogic.

### Hiz olcumu (Blender 5.0, gercek DNA)

| | |
|---|---|
| Tek RigLogic degerlendirmesi | **0.234 ms** |
| Cikti okumayla | **0.486 ms** |
| Numerik Jacobian, 174 parametre | 85 ms/iterasyon |
| Tek kare, 10 iterasyon | 0.8 s |
| 300 kare (10 sn @30fps) | ~4 dakika |

Offline cozum icin fazlasiyla yeterli.

### RigLogic vertex vermiyor

`getRawJointOutputs()` + `getBlendShapeOutputs()` veriyor, vertex konumu
degil. Zinciri kendimiz kuruyoruz (`core/facemodel.py`):

```
notr vertex + blendshape deltalari  ->  linear blend skinning  =  vertex
```

Joint cikti semantigi addon kaynagindan okundu (`rig_logic.py:1613`),
tahmin degil:

- joint basina **9** deger: oteleme(3) + rotasyon(3) + olcek(3)
- hepsi **delta**, uygulama `notr + delta`
- rotasyon **derece**, Euler sirasi **XYZ**
- `SCALE_FACTOR = 100` (`constants.py:38`), notr rotasyonlar da derece
  (`dna_io/importer.py:476`)

### Bolge secimi: geometri degil, skin agirligi

Ilk deneme jointlerin notr dunya konumlarina yakinliga bakiyordu ve
**0 vertex** buldu. Sebep: DNA jointleri ebeveyne gore tutuyor ve dogru
dunya konumu icin ebeveyn ROTASYONLARININ da zincirlenmesi gerekiyor;
sadece oteleme toplamak konumlari kaydiriyor.

Skin agirligi hem bu sorunu atlatiyor hem daha dogru bir tanim: bir vertex,
agirliginin cogu goz kapagi jointlerine gidiyorsa goz kapagindadir.
Bu rig'in kendi tanimi, geometrik tahmin degil.

`--region eyelid --threshold 0.5` -> **2168 vertex**, 737 blendshape
hedefinden **311**'i bu bolgeye dokunuyor.

### Dogrulama

Ileri modelin dogrulugu tek bir testle kanitlanir: sifir kontrolde cikti
notr vertexlerle ayni olmali.

| | |
|---|---|
| `world @ bind^-1` birim matristen sapma | **5.7e-14** (float64 epsilonu) |
| Notr yeniden uretimde mutlak hata | 2.4e-05 cm = **0.00024 mm** |
| Bagil hata | 1.46e-07 |
| float32 epsilonu | 1.19e-07 |

Yani **matris zinciri tam dogru**; kalan sapma yalnizca DNA'nin float32
saklanmasindan geliyor ve olcecegimiz her seyin cok altinda.

Deform suresi 2168 vertex icin 3.3 ms.

### Hizlandirma -- olculdu

Ilk deform 3.04 ms'ti ve **%54'u 870 jointlik hiyerarsi zinciriydi**. Vertex
sayisini dusurmek fayda etmedi (100 vertexte bile 1.96 ms) cunku zincir
sabit maliyet.

Iki gozlem cozdu:

1. Goz kapagi bolgesi 870 jointin sadece **128'ini** kullaniyor (skinlenen
   95 + atalari). Gerisini hesaplamak gereksiz.
2. Zincir sadece **8 seviye** derinlikte. Ayni seviyedeki jointler ayni anda
   hesaplanabilir -> python dongusu joint sayisi kadar degil DERINLIK kadar.

| | once | sonra |
|---|---|---|
| joint zinciri | 1.65 ms | **0.097 ms** (17x) |
| toplam deform | 3.04 ms | **1.46 ms** (2.1x) |
| notr hata | 2.381e-05 | 2.381e-05 (degismedi) |

### Gercek zaman butcesi (30 fps = 33 ms)

Iterasyon maliyeti = (serbest parametre + 1) x (RigLogic 0.486 ms + deform 1.46 ms):

| serbest parametre | 2 iterasyon | |
|---|---|---|
| 4 | 19.5 ms | **gercek zaman** |
| 8 | 35.1 ms | sinirda |
| 16 | 66.3 ms | offline |
| 174 | 682 ms | offline |

Yani **kucuk parametre kumesiyle gercek zaman bugun CPU'da mumkun.** Goz
kapagi durumu tam olarak bu: 4 kontrol.

### GPU nerede yardim eder, nerede etmez

**Etmez:** RigLogic derlenmis bir CPU kutuphanesi (Epic/Poly Hammer ikilisi).
GPU'ya tasinamaz. Numerik Jacobian N+1 **sirali** RigLogic cagrisi gerektiriyor
ve yuksek parametre sayisinda baskin maliyet bu.

**Eder:** Ileri modeli PyTorch'ta **turevlenebilir** yeniden yazarsak autograd
tek geri gecisle tam gradyan veriyor -- N+1 carpani **tamamen kalkiyor**.
Bu 2x degil, O(N) -> O(1) kazanci. Ustune GPU kareleri batch'liyor.

Parcalari zaten elimizde: GUI->raw parcali dogrusal diziler (§7), joint
gruplari (`getJointGroupValues`, dogrusal matrisler), blendshape deltalari,
skin agirliklari. En zor kisim olan semantik tersine muhendislik bitti.

Makinede torch 2.11.0+cu128 ve RTX 5070 Ti (sm_120) calisir durumda.

### Kalanlar

- Optimizasyon dongusu (Gauss-Newton / least squares)
- Landmark <-> vertex karsiligi (24049 vertexten hangisi MediaPipe'in
  hangi noktasi)
- Derinlik verisi (Record3D gibi; Live Link'in MHA modu kapali format)
- Zamansal duzenlileştirme


---

## 14. Turevlenebilir RigLogic (PyTorch)

`core/riglogic_torch.py` RigLogic'i sifirdan yeniden yaziyor. Amac autograd:
numerik Jacobian N+1 **sirali** RigLogic cagrisi gerektiriyor ve RigLogic
derlenmis CPU kodu -- paralellestirilemiyor.

### Zincir (API yoklanarak cikarildi)

```
GUI (174)
  |  parcali dogrusal:  raw = slope * gui + cut,  gui in [from, to]
raw (263)
  |  PSD (545): refere edilen kontrollerin CARPIMI
  |  RBF poz kontrolleri (18): joint rotasyonlarindan, disaridan verilir
kontrol (826)
  |  122 joint grubu -> tek (7830 x 826) matrise indirilebilir
joint deltalari (7830)   +   blendshape agirliklari (782, birebir gather)
```

### Test iki hata yakaladi

**1. Kontrol vektoru 808 degil 826.** Once raw + PSD varsaymistim. Joint
grubu girdi indeksleri 825'e gidince patladi. Eksik olan
`getRBFPoseControlCount() = 18` -- RBF poz kontrolleri PSD'lerin ardina
ekleniyor.

**2. `psdValues` bir CARPAN DEGIL.** Her terimi kendi degeriyle carpiyordum.
"Hepsi +1" ornegi patladi (PSD 453: sutun 191 deger 4.0, sutun 108 deger
1.0 -> benim 4.0, RigLogic 1.0). Dort hipotez 32 gercek ornege karsi
denendi:

| yorum | hata |
|---|---|
| carpim x deger | 3.0e+00 |
| carpim / deger | 7.5e-01 |
| carpim x ilk deger | 3.0e+00 |
| **sadece carpim** | **2.2e-08** |

Ikisini de yakalayan sey **gercek RigLogic'e karsi dogrulama** oldu; rastgele
gercekci girdiler her ikisini de gizliyordu (hata 1e-7), sadece "hepsi +1"
gibi sinir durumu ortaya cikardi.

### Dogrulama sonucu

32 ornek (notr, hepsi +1, hepsi -1, 29 rastgele):

| | |
|---|---|
| joint deltalari, bagil hata | **3.3e-07** |
| blendshape, mutlak hata | **3.0e-08** |
| notr girdi | tam sifir |

float32 kaynak hassasiyeti mertebesinde -- yeniden yazim dogru.

### Hiz

122 grup ayri ayri hesaplanirsa 121 kernel baslatmasi oluyor; tek kare
gradyaninda bu tamamen overhead. Gruplari tek (7830 x 826) matriste
birlestirmek (25 MB, %14.9 dolu) cozuyor:

| | 121 grup | tek GEMM |
|---|---|---|
| CPU tek kare ileri+geri | 11.40 ms | **3.55 ms** |
| CUDA tek kare ileri+geri | 37.70 ms | **4.15 ms** |

**Tam 174-parametre gradyani, tek kare:**

| | sure | numerik Jacobian'a gore |
|---|---|---|
| numerik (RigLogic C++) | 341.1 ms | 1x |
| autograd CPU | **4.51 ms** | **75x** |
| autograd CUDA | 5.46 ms | 62x |

Ikisi de 30 fps butcesinde (33 ms). Tek karede CPU daha hizli -- GPU'nun
kernel baslatma maliyeti bu boyutta baskin. **GPU batch'te kazaniyor:**
64 karelik batch'te kare basina 0.71 ms (CPU 1.77 ms).

Yani: gercek zaman tek kare -> CPU, offline toplu cozum -> GPU.


---

## 15. ARKit'in ulasamadigi kontrolleri turetme

Olculdu: DNA'da **23 goz kontrolu** var, ARKit tablosuyla sadece **8'i**
suruluyor. Kalan 15'i (goz kapaklari, kapak baskisi, goz bebegi, kirpik
tweaker'lari) ARKit'in 52 kanalinda karsiligi olmadigi icin olu duruyor.

Olculemezler ama TURETILEBILIRLER -- anatomi bunlari bakis yonune baglar.
`core/realism.py` esleme ciktisindan turetip ustune ekliyor; kaynagi
(webcam / iPhone / replay) umursamiyor.

### Yonler DNA'dan dogrulandi

| GUI ekseni | negatif | pozitif |
|---|---|---|
| `CTRL_*_eye.ty` | asagi bak | yukari bak |
| `CTRL_*_eye_eyelidU.ty` | ust kapak **KALKAR** (`eyeUpperLidUp`) | kapak gevser (`eyeRelax`) |
| `CTRL_*_eye_eyelidD.ty` | alt kapak iner | alt kapak kalkar |
| `CTRL_*_eye_lidPress.ty` | — | kapak baskisi |

Yani **yukari bakis pozitif ama kapagi kaldirmak icin eyelidU NEGATIF**
olmali; kural `eyelidU = -k * bakis`. Isaret ters yazilirsa goz yukari
bakarken kapak inip goz kapanmis gorunur -- testler bunu tutuyor.

Kirpma sirasinda takip soneriyor (`1 - blink`), yoksa kapak takibi
kapanan gozu geri aciyor.

### Kanal araligi duzeltmesi

Gercek iPhone kaydinda olculdu: kanallar 1.0'a **ulasmiyor**.

| kanal | gercek tepe |
|---|---|
| eyeBlinkLeft/Right | **0.917** |
| mouthSmileLeft | 0.865 |
| browInnerUp | 0.858 |
| jawOpen | 0.973 |

Kisi gozunu tamamen kapatiyor ama ARKit 0.92 diyor; rig 1.0'da tam
kapanacak sekilde kurulu oldugundan **goz kapagi her zaman biraz acik
kaliyor**. "Olu bakis" hissinin somut kaynagi bu.

`ChannelProfile.input_max` gozlenen tepeyi 1.0'a normalize ediyor.
`RangeLearner` bir ROM cekiminden ogreniyor; `floor=0.35` altindaki
kanallara dokunmuyor (hic tetiklenmemis kanali normalize etmek 20x
kazanc demek olur ve gurultuyu patlatir).

### Kirpma egrisi -- KASITLI OLARAK DOKUNULMADI

Ilk hipotezim ARKit'in kirpma egrisini yeniden sekillendirmekti. Gercek
kayitta olculdu ve hipotez CURUTULDU:

| | olculen | literatur |
|---|---|---|
| kapanma | 89 ms | 50-100 ms |
| acilma | 194 ms | 150-300 ms |
| oran | 2.19 | ~2-3 |
| kirpma sikligi | 19.8/dk | 15-20/dk |

ARKit kirpmayi zaten fizyolojik olarak dogru veriyor. `realism.blink_curve`
duruyor ama varsayilani kimlik.

Buna karsilik gercek bir kusur bulundu: canli modal kuyrugu bosaltip
**sadece son paketi** kullaniyordu. Kapanma fazi 60 fps'te 3-7 kare, yani
tepe noktasi atilan pakette kalabiliyordu. Artik bir tikte birden fazla
paket gelirse kanal basina maksimum aliniyor.


---

## 16. Cozucu

`core/solver.py` -- gozlemden rig kontrollerini cozer. Zincir bastan sona
turevlenebilir:

```
GUI (174) -> TorchRigLogic -> joint/blendshape -> TorchFaceModel -> vertex
          -> baryantrik -> landmark -> kayip
```

### Gozlem hizalamasi

MediaPipe landmark'lari kafa pozunu iceriyor ve kamera ic parametrelerini
bilmiyoruz. 2D yeniden izdusum yerine once **rijit hizalama** (Procrustes:
donme + oteleme + tek olcek) yapiliyor; kafa pozu ve olcek ayiklaninca
geriye sadece ifade kaliyor.

### Iki asamali cozum -- olculerek kuruldu

Problem **kotu kosullu**: farkli kontrol kombinasyonlari benzer landmark
dizilimi uretiyor. Sadece L2 ile cozuldugunde 6 kontrollu bir ifadede
**23 kontrol yanlislikla** tetikleniyordu.

Gercek ifadeler az sayida kontrol kullanir; bu bir onsel ve L1 ile
dayatilabilir. Sentetik gidis-donus testiyle olculdu:

| L2 | L1 | RMS | kontrol hatasi | yanlis tetik |
|---|---|---|---|---|
| 1e-4 | 0 | 0.118 mm | 0.093 | 23 |
| 1e-3 | 0 | 0.183 mm | 0.184 | 3 |
| **1e-4** | **5e-4** | **0.112 mm** | **0.066** | **1** |
| 1e-4 | 3e-3 | 0.436 mm | 0.212 | 0 |

Sadece L2'yi buyutmek her seyi bastiriyor (hata 2 katina cikiyor); L1
yanlis tetikleri kirparken dogrulugu da **iyilestiriyor**.

L1'in bilinen yan etkisi tahminleri sifira dogru buzmesi (kirpma 0.50
yerine 0.435 cikiyordu). Careci standart: L1 aktif kumeyi bulsun, sonra
sadece o kume uzerinde L1'siz yeniden coz.

| asama | ort. kontrol hatasi | RMS |
|---|---|---|
| L1 | 0.066 | 0.1118 mm |
| **+ yanlilik giderme** | **0.002** | **0.0060 mm** |

### Sentetik gidis-donus sonucu

Hedef 6 kontrol; cozucu 6 kontrol buldu, hepsi 0.006 hata payinda:

| kontrol | hedef | bulunan |
|---|---|---|
| CTRL_C_jaw | 0.60 | 0.600 |
| CTRL_L_mouth_cornerPull | 0.80 | 0.800 |
| CTRL_R_mouth_cornerPull | 0.80 | 0.794 |
| CTRL_L_eye_blink | 0.50 | 0.499 |
| CTRL_R_eye_blink | 0.50 | 0.500 |
| CTRL_L_brow_raiseIn | 0.70 | 0.697 |

Sure: 6.6 sn (300 + 150 iterasyon, CPU). Offline cozum icin uygun;
gercek zaman icin iterasyon sayisi dusurulup onceki kareden sicak
baslatilmali.

### Dogrulanmadi

Gercek MediaPipe gozlemiyle test edilmedi. Sentetik test ileri modelin
kendi ciktisini kullaniyor, yani **model hatasi yok**; gercek veride
landmark gurultusu, karsilik hatasi ve rig'in oyuncunun yuzunu tam
temsil edememesi devreye girer.
