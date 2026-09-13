"""charface -- ARKit cekimlerini MetaHuman face board'a uygular.

Character DNA (Pro) addon'unun USTUNE biner, onun yerine gecmez. Yaptigi
tek sey face board pose bone'larinin location degerlerini yazmaktir; kemik
transformu, shape key ve wrinkle map hesabini RigLogic'e birakir.

Neden face board (GUI) yazip raw kontrol yazmiyoruz: addon'un evaluate()
akisi her guncellemede mapGUIToRawControls() cagiriyor, yani dogrudan
yazilan raw degerler ayni karede uzerine yaziliyor. Gerekce ve dosya:satir
referanslari: docs/api-notes.md.
"""

bl_info = {
    "name": "charface",
    "description": "ARKit yuz yakalama -> MetaHuman face board",
    "author": "facecap",
    "version": (0, 1, 0),
    "blender": (4, 5, 0),
    "category": "Animation",
}

import sys
from pathlib import Path

import bpy

# core/ paylasilan mantik: esleme, filtre, take formati. Blender'in
# python'una hicbir sey pip ile kurulmaz, core stdlib disi bagimlilik
# kullanmaz.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.calibration import ChannelProfile, Profile, RangeLearner, build_neutral  # noqa: E402
from core.filters import ChannelFilters  # noqa: E402
from core.head import AXES, DEFAULT_MAPPING, HeadTracker, quaternion_from_axis, to_blender_quaternion  # noqa: E402
from core.livelink import DecodeError, to_frame_payload  # noqa: E402
from core.mapping import Mapping, split_axis_key  # noqa: E402
from core.realism import RealismLayer, merge  # noqa: E402
from core.take import plan_scene_frames, read_take  # noqa: E402

# Addon'un sahne property'si surumden surume degisiyor, tek isim sabitlenmez:
#
#   character_dna_pro / character_dna   Character DNA (Pro) 0.8.x   Blender 5.1
#   meta_human_dna                      MetaHuman DNA 0.5.4         Blender 5.0
#
# Instance listesinin adi da 0.5.4 -> 0.8.x arasinda degisti.
SCENE_PROPERTY_CANDIDATES = ("character_dna_pro", "character_dna", "meta_human_dna")
INSTANCE_LIST_CANDIDATES = ("rig_instance_list", "rig_logic_instance_list")


def get_addon_properties(container):
    """Addon'un property group'unu sahne ya da window manager uzerinde bul."""
    for name in SCENE_PROPERTY_CANDIDATES:
        properties = getattr(container, name, None)
        if properties is not None:
            return properties
    return None


def get_rig_instance(context):
    """Aktif rig instance'ini bul. Bulunamazsa None."""
    for name in SCENE_PROPERTY_CANDIDATES:
        scene_properties = getattr(context.scene, name, None)
        if scene_properties is None:
            continue
        for list_name in INSTANCE_LIST_CANDIDATES:
            instances = getattr(scene_properties, list_name, None)
            if not instances:
                continue
            index = getattr(scene_properties, f"{list_name}_active_index", 0)
            if 0 <= index < len(instances):
                return instances[index]
            return instances[0]
    return None


# RigLogic'in EZMEDIGI kemikler. Addon update_head_bone_transforms() icinde
# bunlari `if name in self.head_driver_bone_names: continue` ile atliyor;
# cunku bunlarin rotasyonu RigLogic'e GIRDI (DNA'da head.qx/qy/qz/qw diye
# raw kontrol olarak duruyorlar). Dolayisiyla buraya yazmak guvenli ve
# ustelik boyun/kafa duzelticilerini de tetikliyor.
HEAD_BONE_CANDIDATES = ("head", "neck_02", "neck_01")


def resolve_head_bone(instance, preferred: str = ""):
    """Kafa rotasyonunun yazilacagi pose bone'u bul.

    Addon'un kendi surucu kemik listesi varsa onu dogrula; yoksa bilinen
    adaylara dus. Isim uydurmuyoruz -- adaylar DNA'dan okundu.
    """
    head_rig = getattr(instance, "head_rig", None)
    if head_rig is None:
        return None, "rig instance'in head rig'i yok"

    driver_names = set(getattr(instance, "head_driver_bone_names", []) or [])

    names = [preferred] if preferred.strip() else list(HEAD_BONE_CANDIDATES)
    for name in names:
        pose_bone = head_rig.pose.bones.get(name)
        if pose_bone is None:
            continue
        if driver_names and name not in driver_names:
            return None, (
                f"'{name}' RigLogic surucu kemigi degil; yazilan rotasyon ezilir. "
                f"Surucu kemikler: {sorted(driver_names)[:5]}"
            )
        return pose_bone, None

    return None, f"kafa kemigi bulunamadi (denenen: {names})"


def apply_head_rotation(pose_bone, quaternion) -> None:
    """Kemige quaternion yaz. Rotasyon modu quaternion'a alinir --
    euler'de kaldiysa yazdigimiz deger goz ardi edilir."""
    if pose_bone.rotation_mode != "QUATERNION":
        pose_bone.rotation_mode = "QUATERNION"
    pose_bone.rotation_quaternion = quaternion


def resolve_bones(face_board, mapping: Mapping):
    """Tabloda gecen kemikleri face board uzerinde bul.

    Eksik kemikleri sessizce atlamak yerine dondururuz: DNA surumu ile face
    board surumu uyusmadiginda kullanicinin bunu bilmesi gerekiyor.
    """
    from core.realism import EYE_SIDES

    # Turetilmis eksenler tabloda gecmeyen kemiklere dokunuyor (goz
    # kapaklari, kapak baskisi); onlari da cozmemiz lazim.
    from core.realism import LIPS_TOGETHER_AXES

    derived = {
        key.rsplit(".", 1)[0]
        for side in EYE_SIDES
        for key in (side["upper"], side["lower"], side["press"])
    }
    derived |= {key.rsplit(".", 1)[0] for key in LIPS_TOGETHER_AXES}

    found, missing = {}, []
    for bone_name in sorted(set(mapping.bones) | derived):
        pose_bone = face_board.pose.bones.get(bone_name)
        if pose_bone:
            found[bone_name] = pose_bone
        else:
            missing.append(bone_name)
    return found, missing


# Kalibrasyon ayarlari SAHNEDE durur, operator ozelliginde degil.
# Sebep: operator ozelligi ancak operator baslatilirken secilebiliyor;
# canli yayin devam ederken degistirilemiyor. Kalibrasyon tam da bakarken
# ayar yapmayi gerektiriyor, o yuzden sahneye tasindi -- canli modal her
# karede buradan okuyor ve degisiklik aninda etki ediyor.

_ANGLE_ITEMS = [(a, a, "") for a in AXES] + [("NONE", "Kapali", "")]


class CharfaceSettings(bpy.types.PropertyGroup):
    port: bpy.props.IntProperty(name="UDP Port", default=11111, min=1024, max=65535)  # type: ignore[valid-type]
    smooth: bpy.props.BoolProperty(name="One-Euro Filtre", default=True)  # type: ignore[valid-type]
    mirror: bpy.props.BoolProperty(
        name="Sol/Sag Aynala",
        description="Kamera goruntusunde kullanicinin solu karakterin saginda kalir",
        default=False,
    )  # type: ignore[valid-type]

    head_motion: bpy.props.BoolProperty(name="Kafa Hareketi", default=True)  # type: ignore[valid-type]
    head_influence: bpy.props.FloatProperty(
        name="Genel Etki",
        description="0 = kafa sabit, 1 = tam takip",
        default=1.0,
        min=0.0,
        max=2.0,
        subtype="FACTOR",
    )  # type: ignore[valid-type]

    yaw_axis: bpy.props.EnumProperty(name="Eksen", items=_ANGLE_ITEMS, default=DEFAULT_MAPPING["yaw"][0])  # type: ignore[valid-type]
    yaw_invert: bpy.props.BoolProperty(name="Ters", default=False)  # type: ignore[valid-type]
    yaw_gain: bpy.props.FloatProperty(name="Kazanc", default=1.0, min=0.0, max=3.0)  # type: ignore[valid-type]

    pitch_axis: bpy.props.EnumProperty(name="Eksen", items=_ANGLE_ITEMS, default=DEFAULT_MAPPING["pitch"][0])  # type: ignore[valid-type]
    pitch_invert: bpy.props.BoolProperty(name="Ters", default=False)  # type: ignore[valid-type]
    pitch_gain: bpy.props.FloatProperty(name="Kazanc", default=1.0, min=0.0, max=3.0)  # type: ignore[valid-type]

    roll_axis: bpy.props.EnumProperty(name="Eksen", items=_ANGLE_ITEMS, default=DEFAULT_MAPPING["roll"][0])  # type: ignore[valid-type]
    roll_invert: bpy.props.BoolProperty(name="Ters", default=False)  # type: ignore[valid-type]
    roll_gain: bpy.props.FloatProperty(name="Kazanc", default=1.0, min=0.0, max=3.0)  # type: ignore[valid-type]

    test_angle: bpy.props.FloatProperty(name="Test Acisi", default=25.0, min=-90.0, max=90.0)  # type: ignore[valid-type]

    # --- yuz ifadesi kalibrasyonu (grup bazli)
    # Kanal kanal ayar 51 kaydirici demek. Insanlar "agiz fazla oynuyor"
    # diye dusunuyor, "mouthStretchLeft yuksek" diye degil.
    jaw_gain: bpy.props.FloatProperty(name="Cene", default=1.0, min=0.0, max=3.0)  # type: ignore[valid-type]
    lips_gain: bpy.props.FloatProperty(name="Dudaklar", default=1.0, min=0.0, max=3.0)  # type: ignore[valid-type]
    lip_close_gain: bpy.props.FloatProperty(
        name="Dudak Kapanma",
        description=(
            "mouthClose kanali. 'm', 'b', 'p' seslerinde cene acikken dudaklarin "
            "kapanmasini saglar. Dusuk kalirsa agiz surekli acik gorunur"
        ),
        default=1.0,
        min=0.0,
        max=3.0,
    )  # type: ignore[valid-type]
    eyes_gain: bpy.props.FloatProperty(name="Gozler", default=1.0, min=0.0, max=3.0)  # type: ignore[valid-type]
    brows_gain: bpy.props.FloatProperty(name="Kaslar", default=1.0, min=0.0, max=3.0)  # type: ignore[valid-type]

    mouth_deadzone: bpy.props.FloatProperty(
        name="Agiz Olu Bolge",
        description=(
            "MediaPipe dinlenme halinde bile agiz kanallarinda kucuk degerler "
            "donduruyor; bu esigin altini sifirlar. Yuksek tutarsan konusma da kirpilir"
        ),
        default=0.06,
        min=0.0,
        max=0.4,
    )  # type: ignore[valid-type]
    # --- ARKit'in kanali olmayan kontrolleri turetme
    realism: bpy.props.BoolProperty(
        name="Turetilmis Goz Kontrolleri",
        description=(
            "DNA'daki 23 goz kontrolunden 15'ini ARKit suremiyor (kanali yok). "
            "Bunlari bakis yonunden ve kisilmadan turet"
        ),
        default=True,
    )  # type: ignore[valid-type]
    eyelid_follow: bpy.props.FloatProperty(
        name="Kapak Bakisi Takip",
        description=(
            "Ust kapagin dikey bakisi izleme miktari. 0'da kapak sabit kalir ve "
            "asagi bakarken goz bosluga dalmis gorunur"
        ),
        default=0.85,
        min=0.0,
        max=2.0,
    )  # type: ignore[valid-type]
    lower_lid_follow: bpy.props.FloatProperty(
        name="Alt Kapak Takip", default=0.25, min=0.0, max=1.5
    )  # type: ignore[valid-type]
    lid_press: bpy.props.FloatProperty(
        name="Kisilmada Kapak Baskisi", default=0.5, min=0.0, max=2.0
    )  # type: ignore[valid-type]
    pucker_close: bpy.props.FloatProperty(
        name="Buzusturmede Dudak Kapanma",
        description=(
            "Opucuk/buzusturmede dudaklari birlestirir. 0'da agzin ortasinda "
            "bosluk kalir cunku pucker funnel'i (acik O) suruyor ama hicbir sey "
            "dudaklari kapatmiyor"
        ),
        default=0.7,
        min=0.0,
        max=2.0,
    )  # type: ignore[valid-type]
    saccades: bpy.props.FloatProperty(
        name="Mikro-sakkad",
        description="Gercek goz sabit bakarken bile kucuk siciramalar yapar; 0 = tamamen sabit",
        default=1.0,
        min=0.0,
        max=3.0,
    )  # type: ignore[valid-type]

    # --- kanal araligi
    use_channel_range: bpy.props.BoolProperty(
        name="Kanal Araligi Duzeltmesi",
        description=(
            "Kanallar 1.0'a ulasmiyor (olculdu: eyeBlink 0.917'de doyuyor), bu yuzden "
            "goz tam kapanmiyor. Ogrenilen tepeyi 1.0'a normalize et"
        ),
        default=True,
    )  # type: ignore[valid-type]
    learning_range: bpy.props.BoolProperty(name="Aralik Ogreniliyor", default=False)  # type: ignore[valid-type]
    profile_path: bpy.props.StringProperty(
        name="Profil",
        description="Ogrenilen kanal araliklarinin saklandigi dosya",
        default="//charface_profile.json",
        subtype="FILE_PATH",
    )  # type: ignore[valid-type]

    expression_neutral: bpy.props.BoolProperty(
        name="Notr Yuzu Ogren",
        description="Canli baslarken ilk 30 karenin ortalamasini notr kabul edip cikar",
        default=True,
    )  # type: ignore[valid-type]

    preview_hz: bpy.props.IntProperty(
        name="Onizleme Hz",
        description=(
            "Viewport'u saniyede en fazla bu kadar guncelle. 0 = her karede. "
            "KAYDI ETKILEMEZ -- keyframe'ler tam hizda yazilir. Agir sahnede "
            "(particle sac, kalabalik armature) 15-20 iyi bir baslangic"
        ),
        default=0,
        min=0,
        max=60,
    )  # type: ignore[valid-type]

    light_capture: bpy.props.BoolProperty(
        name="Yakalamada Sahneyi Hafiflet",
        description=(
            "Canli yakalama boyunca particle sistemlerini gizler ve Simplify'i "
            "acar (cocuk particle 0, subdivision 0). Durdurunca geri alinir"
        ),
        default=True,
    )  # type: ignore[valid-type]

    recording: bpy.props.BoolProperty(
        name="Timeline'a Kaydet",
        description=(
            "Canli yakalama sirasinda keyframe yazar. Oynatma kafasi neredeyse "
            "oradan baslar; kapatinca sahne araligi yazilan kareye ayarlanir"
        ),
        default=False,
    )  # type: ignore[valid-type]


def get_settings(context):
    return getattr(context.scene, "charface", None)


def settings_profile(settings, neutral: dict | None = None) -> Profile:
    """Sahne ayarlarindan calisir bir Profile kur.

    Grup bazli kazanc: mouthClose'u ayri grup tuttuk cunku dudak kapanmasi
    diger agiz kanallariyla ayni olcekte davranmiyor -- konusmada onun
    kuvvetli olmasi gerekiyor, digerlerinin kisilmasi.
    """
    if settings is None:
        return Profile(neutral=neutral or {})

    deadzone = settings.mouth_deadzone
    channels = dict(_CHANNEL_RANGES["profiles"]) if settings.use_channel_range else {}

    return Profile(
        neutral=neutral or {},
        channels=channels,
        groups={
            "jaw": ChannelProfile(gain=settings.jaw_gain, deadzone=deadzone),
            "lips": ChannelProfile(gain=settings.lips_gain, deadzone=deadzone),
            "lipClose": ChannelProfile(gain=settings.lip_close_gain, deadzone=deadzone * 0.5),
            "eyes": ChannelProfile(gain=settings.eyes_gain),
            "brows": ChannelProfile(gain=settings.brows_gain),
        },
        mirror=settings.mirror,
    )


def settings_head_mapping(settings) -> dict:
    """Sahne ayarlarini core.head'in bekledigi esleme sozluguene cevir.
    Kazanc isarete carpilir: kazanc 0 o aciyi kapatir, 2 iki katina cikarir."""
    if settings is None:
        return dict(DEFAULT_MAPPING)
    return {
        "yaw": (
            "" if settings.yaw_axis == "NONE" else settings.yaw_axis,
            (-1.0 if settings.yaw_invert else 1.0) * settings.yaw_gain,
        ),
        "pitch": (
            "" if settings.pitch_axis == "NONE" else settings.pitch_axis,
            (-1.0 if settings.pitch_invert else 1.0) * settings.pitch_gain,
        ),
        "roll": (
            "" if settings.roll_axis == "NONE" else settings.roll_axis,
            (-1.0 if settings.roll_invert else 1.0) * settings.roll_gain,
        ),
    }


# Canli oturumun son durumu -- panelde okunur gostermek icin. Sahne
# property'sine her karede yazmak undo yigini ve depsgraph acisindan
# pahali, bu yuzden modul duzeyinde tutuluyor.
# ROM cekiminden ogrenilen kanal tepeleri. Sahne property'sine yazmak
# yerine modul duzeyinde tutuluyor -- her karede guncelleniyor.
_CHANNEL_RANGES: dict = {"learner": None, "profiles": {}, "path": None}

_LIVE_STATUS = {
    "packets": 0,
    "badPackets": 0,
    "merged": 0,
    "source": "-",
    "pitch": 0.0,
    "yaw": 0.0,
    "roll": 0.0,
    "face": False,
    "tongue": 0.0,
    "ranges": "-",
    "recorded": "-",
    "evalMs": 0.0,      # RigLogic + depsgraph
    "applyMs": 0.0,     # kemiklere yazma + kayit
    "budgetHz": 0.0,    # bu hizda kac kare/sn yetisir
    "skipped": 0,       # onizleme kismasi yuzunden atlanan kare
}


def decode_packet(payload: bytes, frame_number: int) -> dict | None:
    """Gelen UDP paketini semaya cevir. Kaynak ne olursa olsun.

    Iki format destekleniyor ve AYIRT EDILMESI gerekmiyor:

    * `detector/detect.py` ve `scripts/replay_take.py` duz JSON yolluyor
    * iPhone'daki **Live Link Face** kendi ikili paketini yolluyor

    JSON once denenir cunku ucuz ve kesin ('{' ile basliyor). Tutmazsa
    ikili cozucuye dusuyoruz. Boylece kullanici tarafinda ayar yok:
    hangi kaynagi acarsan onu dinliyor.
    """
    import json

    if payload[:1] == b"{":
        try:
            return json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    try:
        import time

        return to_frame_payload(payload, time.time() % 86400.0, frame_number)
    except (DecodeError, ValueError, IndexError):
        return None


class HeadMappingMixin:
    """Kafa eslemesini SAHNE ayarlarindan alir.

    Operator ozelliginde tutulsaydi canli yayin sirasinda degistirilemezdi.
    """

    def head_mapping(self, context) -> dict:
        return settings_head_mapping(get_settings(context))


def resolve_profile_path(settings) -> Path:
    """Ayardaki yolu mutlak hale getir.

    Varsayilan `//charface_profile.json` yani .blend dosyasinin yaninda.
    Sahne hic kaydedilmemisse Blender `//` yi cozemez; o zaman projeye dus.
    """
    raw = (settings.profile_path if settings else "").strip() or "//charface_profile.json"
    if raw.startswith("//") and not bpy.data.filepath:
        return PROJECT_ROOT / "charface_profile.json"
    return Path(bpy.path.abspath(raw))


def render_head_reference(context, out_dir: Path, resolution: int = 1024):
    """Karakterin kafasini onden render eder ve kamera parametrelerini yazar.

    Cozucunun landmark <-> vertex karsiligini kurmasi icin gerekli: MediaPipe
    bu render'da yuzu bulur, sonra her landmark'tan isin atip mesh'e carptigi
    vertex bulunur. Karsilik jenerik bir yuz modeliyle degil KARAKTERIN KENDI
    kafasiyla kurulmus olur.

    Sahneyi kirletmez: gecici kamera/isik eklenir, render ayarlari degistirilir,
    sonra hepsi geri alinir.
    """
    import json

    from mathutils import Vector

    instance = get_rig_instance(context)
    mesh = getattr(instance, "head_mesh", None) if instance else None
    if mesh is None:
        meshes = [o for o in context.scene.objects if o.type == "MESH"]
        if not meshes:
            raise RuntimeError("sahnede mesh yok")
        mesh = max(meshes, key=lambda o: len(o.data.vertices))

    scene = context.scene
    render = scene.render

    # --- geri almak icin mevcut durumu sakla
    previous = {
        "camera": scene.camera,
        "engine": render.engine,
        "x": render.resolution_x,
        "y": render.resolution_y,
        "percent": render.resolution_percentage,
        "format": render.image_settings.file_format,
        "filepath": render.filepath,
        "transparent": render.film_transparent,
    }

    # --- kafa ve ifade GECICI olarak notre alinir
    #
    # Olculdu ve pahaliya mal oldu: ilk karsilik ~17 derece donuk kafayla
    # kuruldu. MediaPipe daha cok landmark buluyor (473 -> 478) ama uzak
    # yanaktaki landmark'lar onden atilan isinla YAKIN taraftaki vertexlere
    # baglaniyor. Sonuc: tek tarafli bir kontrol (R_mouth_cornerPull) 331
    # landmark oynatiyordu ve bunlarin x ortalamasi yuzun tam ortasindaydi --
    # imkansiz. Cozucu simetrik bir gulumsemeyi hicbir kontrol kombinasyonuyla
    # aciklayamiyordu.
    #
    # Landmark SAYISI degil, baglanma dogrulugu onemli. O 5 fazla landmark
    # yaklasik 100 tanesinin yanlis baglanmasina mal oldu.
    posed = []
    head_bone, _ = resolve_head_bone(instance) if instance else (None, None)
    if head_bone is not None:
        if head_bone.rotation_mode != "QUATERNION":
            head_bone.rotation_mode = "QUATERNION"
        posed.append((head_bone, "rotation_quaternion", list(head_bone.rotation_quaternion)))
        head_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)

    face_board = getattr(instance, "face_board", None) if instance else None
    if face_board is not None:
        for pose_bone in face_board.pose.bones:
            location = list(pose_bone.location)
            if any(abs(v) > 1e-6 for v in location):
                posed.append((pose_bone, "location", location))
                pose_bone.location = (0.0, 0.0, 0.0)

    camera_data = bpy.data.cameras.new("charface_tmp_cam")
    camera = bpy.data.objects.new("charface_tmp_cam", camera_data)
    light_data = bpy.data.lights.new("charface_tmp_light", type="SUN")
    light = bpy.data.objects.new("charface_tmp_light", light_data)
    scene.collection.objects.link(camera)
    scene.collection.objects.link(light)

    try:
        # notre alma RigLogic'i yeniden degerlendirmeli, yoksa bounding box
        # ve render hala eski pozu gosterir
        context.view_layer.update()

        corners = [mesh.matrix_world @ Vector(c) for c in mesh.bound_box]
        center = sum(corners, Vector()) / 8
        size = max((c - center).length for c in corners)

        # Kadraj IKI TUR OLCULEREK ayarlandi, tahminle degil.
        #
        # Bounding box boyun ve omuzlari da iceriyor, bu yuzden ham merkez
        # yuzun altinda kaliyor -- ilk denemede yuz karenin sadece %44'unu
        # kapliyordu ve merkezi %31'deydi.
        #
        # 1. tur (hedef +0.24, mesafe 1.64): yuz %88 oldu ama CENE TASTI --
        #    6 landmark (148, 152, 175, 176, 377, 400) kare disina cikti.
        # 2. tur: landmark kutusundan geri hesaplandi, %72 hedeflendi ki
        #    cene ve alin icin pay kalsin.
        #
        # 85mm lens + 36mm sensor -> 23.9 derece gorus alani.
        # Daha cok piksel = daha hassas landmark = daha iyi karsilik, ama
        # tasma karsiligi tamamen bozar; pay birakmak daha degerli.
        target = center + Vector((0.0, 0.0, size * 0.18))
        camera.location = target + Vector((0.0, -size * 2.0, 0.0))
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera_data.lens = 85.0  # portre lensi, perspektif bozulmasi az

        light.location = camera.location + Vector((0.0, 0.0, size))
        light.rotation_euler = camera.rotation_euler
        light_data.energy = 4.0

        scene.camera = camera
        render.resolution_x = render.resolution_y = resolution
        render.resolution_percentage = 100
        render.image_settings.file_format = "PNG"
        render.film_transparent = False
        try:
            render.engine = "BLENDER_EEVEE_NEXT"
        except TypeError:
            render.engine = "BLENDER_EEVEE"

        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / "head_front.png"
        render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)

        (out_dir / "camera.json").write_text(
            json.dumps(
                {
                    "meshObject": mesh.name,
                    "resolution": [resolution, resolution],
                    "location": list(camera.location),
                    "rotationEuler": list(camera.rotation_euler),
                    "lens": camera_data.lens,
                    "sensorWidth": camera_data.sensor_width,
                    "headCenter": list(center),
                    "cameraTarget": list(target),
                    "headSize": size,
                    # Cozucu KARAKTERIN KENDI DNA'sini kullanmali. Varsayilan
                    # head.dna ile cozunce olcek %5.7 kayiyor ve artik 2.75 mm
                    # kaliyor -- baska bir kafayi modellemis oluyoruz.
                    "headDnaPath": str(getattr(instance, "head_dna_file_path", "") or ""),
                    "meshMatrixWorld": [list(row) for row in mesh.matrix_world],
                    # kac kanal notre alindi -- karsilik dogrulanirken
                    # "render notr muydu" sorusunun cevabi burada dursun
                    "neutralized": len(posed),
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        return image_path, mesh.name
    finally:
        # --- pozu geri koy
        for pose_bone, attribute, value in reversed(posed):
            try:
                setattr(pose_bone, attribute, value)
            except (RuntimeError, ReferenceError):
                pass

        # --- sahneyi bulundugu gibi birak
        for obj, data in ((camera, camera_data), (light, light_data)):
            try:
                scene.collection.objects.unlink(obj)
            except (RuntimeError, ReferenceError):
                pass
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.cameras.remove(camera_data, do_unlink=True)
        bpy.data.lights.remove(light_data, do_unlink=True)

        scene.camera = previous["camera"]
        render.engine = previous["engine"]
        render.resolution_x = previous["x"]
        render.resolution_y = previous["y"]
        render.resolution_percentage = previous["percent"]
        render.image_settings.file_format = previous["format"]
        render.filepath = previous["filepath"]
        render.film_transparent = previous["transparent"]


def build_correspondence(context, folder: Path):
    """Landmark <-> vertex karsiligini isin atarak kurar.

    Her landmark'tan kameradan mesh'e isin atilir, carptigi ucgen ve
    BARYANTRIK AGIRLIKLAR kaydedilir.

    Neden en yakin vertex degil: landmark bir vertexin tam ustune
    oturmuyor, ucgenin ortasina dusuyor. En yakini almak yarim vertex
    araligi kadar hata demek. Uc vertexin agirlikli ortalamasi hem daha
    dogru hem cozucu icin turevlenebilir.
    """
    import json

    from mathutils import Vector

    landmarks_path = folder / "head_front.landmarks.json"
    camera_path = folder / "camera.json"
    for path in (landmarks_path, camera_path):
        if not path.exists():
            raise RuntimeError(f"eksik dosya: {path.name}")

    landmark_data = json.loads(landmarks_path.read_text(encoding="utf-8"))
    camera_info = json.loads(camera_path.read_text(encoding="utf-8"))

    mesh_object = context.scene.objects.get(camera_info["meshObject"])
    if mesh_object is None:
        raise RuntimeError(f"mesh sahnede yok: {camera_info['meshObject']}")

    scene = context.scene
    camera_data = bpy.data.cameras.new("charface_corr_cam")
    camera = bpy.data.objects.new("charface_corr_cam", camera_data)
    scene.collection.objects.link(camera)

    try:
        camera.location = Vector(camera_info["location"])
        camera.rotation_euler = camera_info["rotationEuler"]
        camera_data.lens = camera_info["lens"]
        camera_data.sensor_width = camera_info["sensorWidth"]
        context.view_layer.update()

        frame = camera_data.view_frame(scene=scene)
        top_right, bottom_right, bottom_left, top_left = frame

        # DEGERLENDIRILMIS mesh kullan, taban mesh degil.
        #
        # Ilk surumde isini deforme olmus (armature + shape key) geometriye
        # atip ucgen testini TABAN vertexlerle yapiyordum. Kafa poz almissa
        # ikisi tutmuyor: 468 karsiligin 468'i de yedek yola dustu, yani
        # baryantrik hesap hic calismadi ve agirliklar [1,0,0] kaldi.
        #
        # Topoloji degismedigi icin (shape key ve armature vertex sayisini
        # korur) indeksler ayni; sadece KONUMLARI degerlendirilmisten almak
        # gerekiyor.
        depsgraph = context.evaluated_depsgraph_get()
        evaluated = mesh_object.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()

        matrix = mesh_object.matrix_world
        inverse = matrix.inverted()
        origin_world = camera.matrix_world.translation

        records, missed = [], []
        for index, point in enumerate(landmark_data["landmarks"]):
            u, v = point["x"], point["y"]
            if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                missed.append({"index": index, "reason": "kare disinda"})
                continue

            # MediaPipe y'yi yukaridan asagi verir, view_frame asagidan yukari
            top = top_left.lerp(top_right, u)
            bottom = bottom_left.lerp(bottom_right, u)
            target = camera.matrix_world @ top.lerp(bottom, v)
            direction = (target - origin_world).normalized()

            hit, location, _, face_index = evaluated.ray_cast(
                inverse @ origin_world, inverse.to_3x3() @ direction
            )
            if not hit:
                missed.append({"index": index, "reason": "mesh'e carpmadi"})
                continue

            hit_world = matrix @ location
            polygon = mesh.polygons[face_index]
            indices = list(polygon.vertices)

            # Teget carpmalari at.
            #
            # Isin yuzeye neredeyse paralel geldiginde bir piksellik landmark
            # hatasi yuzeyde santimetrelerce kayma demek; siluete yakin
            # baglantilar bu yuzden guvenilmez. Ayrica MediaPipe siluetteki
            # landmark'i zaten kenara "yapistiriyor", gercek yuzey noktasina
            # degil.
            #
            # 0.35 = ~70 derece gelis acisi. Daha siki bir esik burun ve cene
            # kenarlarini da atiyordu; daha gevsek olani siluet gurultusunu
            # iceri aliyor.
            normal_world = (matrix.to_3x3() @ polygon.normal).normalized()
            facing = -normal_world.dot(direction)
            if facing < 0.35:
                missed.append(
                    {"index": index, "reason": f"teget carpma (cos={facing:.2f})"}
                )
                continue

            # Baryantrik agirliklari DOGRUDAN hesapla.
            #
            # `intersect_point_tri` katı: nokta ucgenin duzleminden azicik
            # sapsa None donuyor ve her sey yedek yola dusuyor. Kendimiz
            # hesaplayip toleransla kabul ediyoruz, ve hicbir ucgen tam
            # icermezse EN IYI olani (en az disari tasan) seciyoruz --
            # boylece agirliklar yine anlamli kaliyor.
            chosen, weights, best_error = None, None, None
            for k in range(1, len(indices) - 1):
                a, b, c = indices[0], indices[k], indices[k + 1]
                pa = matrix @ mesh.vertices[a].co
                pb = matrix @ mesh.vertices[b].co
                pc = matrix @ mesh.vertices[c].co

                v0, v1, v2 = pb - pa, pc - pa, hit_world - pa
                d00, d01, d11 = v0.dot(v0), v0.dot(v1), v1.dot(v1)
                d20, d21 = v2.dot(v0), v2.dot(v1)
                denominator = d00 * d11 - d01 * d01
                if abs(denominator) < 1e-12:
                    continue  # dejenere ucgen

                beta = (d11 * d20 - d01 * d21) / denominator
                gamma = (d00 * d21 - d01 * d20) / denominator
                alpha = 1.0 - beta - gamma

                # ucgenin disina ne kadar tasiyor
                error = -min(0.0, alpha, beta, gamma)
                if best_error is None or error < best_error:
                    best_error = error
                    chosen = [a, b, c]
                    weights = [round(alpha, 6), round(beta, 6), round(gamma, 6)]
                if error <= 1e-6:
                    break

            if chosen is None:
                nearest = min(indices, key=lambda i: (matrix @ mesh.vertices[i].co - hit_world).length)
                chosen, weights = [int(nearest)] * 3, [1.0, 0.0, 0.0]

            records.append(
                {
                    "landmark": index,
                    "vertices": [int(x) for x in chosen],
                    "weights": weights,
                    "position": [round(c, 6) for c in hit_world],
                    "facing": round(facing, 4),
                }
            )

        out_path = folder / "correspondence.json"
        out_path.write_text(
            json.dumps(
                {
                    "meshObject": mesh_object.name,
                    "landmarkCount": len(landmark_data["landmarks"]),
                    "resolved": len(records),
                    "missed": missed,
                    "correspondence": records,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        return out_path, len(records), len(landmark_data["landmarks"]), missed
    finally:
        try:
            evaluated.to_mesh_clear()
        except (NameError, RuntimeError):
            pass
        bpy.data.objects.remove(camera, do_unlink=True)
        bpy.data.cameras.remove(camera_data, do_unlink=True)


class CHARFACE_OT_build_correspondence(bpy.types.Operator):
    """Landmark dosyasindan vertex karsiligini kurar (once render + 08_landmarks)"""

    bl_idname = "charface.build_correspondence"
    bl_label = "Karsiligi Kur"
    bl_options = {"REGISTER"}

    def execute(self, context):
        folder = PROJECT_ROOT / "mapping" / "_generated" / "correspondence"
        try:
            path, resolved, total, missed = build_correspondence(context, folder)
        except Exception as error:  # noqa: BLE001 - sahneye/dosyaya bagli
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        level = "INFO" if resolved == total else "WARNING"
        self.report({level}, f"{resolved}/{total} landmark cozuldu -> {path.name}")
        print(f"[charface] karsilik: {resolved}/{total} -> {path}")
        for item in missed[:5]:
            print(f"[charface]   landmark {item['index']}: {item['reason']}")
        return {"FINISHED"}


class CHARFACE_OT_render_head(bpy.types.Operator):
    """Kafayi onden render eder (cozucu icin landmark karsiligi kurmaya yarar)"""

    bl_idname = "charface.render_head"
    bl_label = "Kafayi Render Et"
    bl_options = {"REGISTER"}

    def execute(self, context):
        out_dir = PROJECT_ROOT / "mapping" / "_generated" / "correspondence"
        try:
            image_path, mesh_name = render_head_reference(context, out_dir)
        except Exception as error:  # noqa: BLE001 - sahneye bagli, ne gelecegi belirsiz
            self.report({"ERROR"}, f"render basarisiz: {error}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"{mesh_name} -> {image_path}")
        print(f"[charface] render: {image_path}")
        print("[charface] sonraki adim:")
        print(f"[charface]   detector/.venv/Scripts/python scripts/08_landmarks.py \"{image_path}\"")
        return {"FINISHED"}


class CHARFACE_OT_save_ranges(bpy.types.Operator):
    """Ogrenilen kanal araliklarini dosyaya kaydeder"""

    bl_idname = "charface.save_ranges"
    bl_label = "Araliklari Kaydet"
    bl_options = {"REGISTER"}

    def execute(self, context):
        learner = _CHANNEL_RANGES["learner"]
        if learner is None or not learner.maxima:
            self.report({"WARNING"}, "Ogrenilmis aralik yok. Once 'Aralik Ogreniliyor' ile ROM cek.")
            return {"CANCELLED"}

        path = resolve_profile_path(get_settings(context))
        try:
            count = learner.save(path)
        except OSError as error:
            self.report({"ERROR"}, f"yazilamadi: {error}")
            return {"CANCELLED"}

        _CHANNEL_RANGES["path"] = str(path)
        self.report({"INFO"}, f"{count} kanal -> {path}")
        return {"FINISHED"}


class CHARFACE_OT_load_ranges(bpy.types.Operator):
    """Kaydedilmis kanal araliklarini yukler"""

    bl_idname = "charface.load_ranges"
    bl_label = "Araliklari Yukle"
    bl_options = {"REGISTER"}

    def execute(self, context):
        path = resolve_profile_path(get_settings(context))
        loaded = load_ranges_from(path)
        if loaded is None:
            self.report({"WARNING"}, f"profil yok: {path}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"{loaded} kanal yuklendi: {path}")
        return {"FINISHED"}


def load_ranges_from(path: Path) -> int | None:
    """Profildeki kanal araliklarini bellege al. Yoksa None."""
    if not path.exists():
        return None
    try:
        profile = Profile.load(path)
    except (OSError, ValueError, TypeError):
        return None
    _CHANNEL_RANGES["profiles"] = dict(profile.channels)
    _CHANNEL_RANGES["path"] = str(path)
    _LIVE_STATUS["ranges"] = f"{len(profile.channels)} kanal yuklendi"
    return len(profile.channels)


class CHARFACE_OT_clear_ranges(bpy.types.Operator):
    """Ogrenilen kanal araliklarini sifirlar"""

    bl_idname = "charface.clear_ranges"
    bl_label = "Araliklari Sifirla"
    bl_options = {"REGISTER"}

    def execute(self, context):
        _CHANNEL_RANGES["learner"] = None
        _CHANNEL_RANGES["profiles"] = {}
        _CHANNEL_RANGES["path"] = None
        _LIVE_STATUS["ranges"] = "-"
        self.report({"INFO"}, "Kanal araliklari sifirlandi.")
        return {"FINISHED"}


class CHARFACE_OT_reset_calibration(bpy.types.Operator):
    """Eksen eslemesini koddaki varsayilana dondurur.

    Ayarlar sahnede saklandigi icin koddaki varsayilan degistiginde mevcut
    sahneler eski degeri tutmaya devam eder. Bu tus ikisini eslestirir.
    """

    bl_idname = "charface.reset_calibration"
    bl_label = "Varsayilan Eksenlere Don"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = get_settings(context)
        if settings is None:
            self.report({"ERROR"}, "Ayarlar bulunamadi.")
            return {"CANCELLED"}

        for name in ("yaw", "pitch", "roll"):
            axis, sign = DEFAULT_MAPPING[name]
            setattr(settings, f"{name}_axis", axis)
            setattr(settings, f"{name}_invert", sign < 0)
            setattr(settings, f"{name}_gain", 1.0)
        settings.head_influence = 1.0

        self.report(
            {"INFO"},
            "Eksenler varsayilana dondu: "
            + ", ".join(f"{n}={DEFAULT_MAPPING[n][0]}" for n in ("yaw", "pitch", "roll")),
        )
        return {"FINISHED"}


class CHARFACE_OT_reset_neutral(bpy.types.Operator):
    """Kafa notrunu yeniden ogrenir.

    Notr, canli oturumun ilk 30 karesinden kuruluyor. O sirada duzgun
    oturmuyorsan kafa surekli kaymis durur. Bu tus referansi sifirlar --
    duz bak, bas, o an notr kabul edilir.
    """

    bl_idname = "charface.reset_neutral"
    bl_label = "Notru Yeniden Al"
    bl_options = {"REGISTER"}

    def execute(self, context):
        operator = _LIVE_SESSION.get("operator")
        tracker = _LIVE_SESSION.get("tracker")
        if tracker is None and operator is None:
            self.report({"WARNING"}, "Canli oturum yok. Once canli yakalamayi baslat.")
            return {"CANCELLED"}

        if tracker is not None:
            tracker.reset()
        if operator is not None:
            # yuz notrunu de sifirla -- kafa duzelirken agiz kaymis kalmasin
            operator._neutral_samples = []
            operator._profile.neutral = {}

        self.report({"INFO"}, "Notr sifirlandi. Ifadesiz dur ve duz bak.")
        return {"FINISHED"}


class CHARFACE_OT_test_head_axis(bpy.types.Operator):
    """Kafa kemigini secilen yerel eksende dondurur -- hangi eksenin ne
    yaptigini GORMEK icin. Tahmin etmek yerine olcup ayarliyoruz.

    UNDO KASITLI OLARAK YOK. Redo panelinden ozellik degistirmek Blender'i
    "geri al + yeniden calistir" dongusune sokuyor; undo sahne datablock'larini
    yeniden kuruyor ama addon RigLogic'in C++ nesnelerini python sozlugunde
    onbellege aliyor -- o onbellek bayat isaretci tutup sonraki evaluate()
    cagrisinda sureci dusuruyordu. Bu yuzden eksen redo panelinden degil,
    panelde ayri tuslardan seciliyor.
    """

    bl_idname = "charface.test_head_axis"
    bl_label = "Kafa Eksenini Test Et"
    bl_options = {"REGISTER"}

    axis: bpy.props.EnumProperty(
        name="Kemik Ekseni",
        items=[(a, a, "") for a in AXES] + [("RESET", "Sifirla", "")],
        default="X",
        options={"SKIP_SAVE"},
    )  # type: ignore[valid-type]

    angle: bpy.props.FloatProperty(
        name="Aci (derece)",
        default=25.0,
        min=-90.0,
        max=90.0,
        options={"SKIP_SAVE"},
    )  # type: ignore[valid-type]

    def execute(self, context):
        import math

        instance = get_rig_instance(context)
        if instance is None:
            self.report({"ERROR"}, "Rig instance bulunamadi.")
            return {"CANCELLED"}

        head_bone, problem = resolve_head_bone(instance)
        if head_bone is None:
            self.report({"ERROR"}, problem or "kafa kemigi yok")
            return {"CANCELLED"}

        if self.axis == "RESET":
            apply_head_rotation(head_bone, (1.0, 0.0, 0.0, 0.0))
            message = f"{head_bone.name} sifirlandi"
        else:
            quaternion = quaternion_from_axis(AXES.index(self.axis), math.radians(self.angle))
            apply_head_rotation(head_bone, to_blender_quaternion(quaternion))
            message = (
                f"{head_bone.name}: {self.axis} ekseninde {self.angle:.0f} derece. "
                "Ne oldu? egme=pitch, cevirme=yaw, yatirma=roll"
            )

        # Sahneyi guncelle. Addon'un depsgraph handler'i zaten tetiklenir ama
        # arka planda/handler kapaliyken de gorunsun diye acikca cagiriyoruz;
        # C++ tarafi patlarsa python'a hata gelmeyebilir, o yuzden sariyoruz.
        try:
            instance.evaluate(component="head")
        except Exception as error:  # noqa: BLE001 - ne gelecegi addon surumune bagli
            self.report({"WARNING"}, f"evaluate hatasi: {error}")

        self.report({"INFO"}, message)
        return {"FINISHED"}


class CHARFACE_OT_bake_take(HeadMappingMixin, bpy.types.Operator):
    """Bir .jsonl cekimini face board'a keyframe olarak yazar"""

    bl_idname = "charface.bake_take"
    bl_label = "Cekimi Face Board'a Bake Et"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")  # type: ignore[valid-type]
    filter_glob: bpy.props.StringProperty(default="*.jsonl", options={"HIDDEN"})  # type: ignore[valid-type]

    start_frame: bpy.props.IntProperty(name="Baslangic Karesi", default=1)  # type: ignore[valid-type]

    single_channel: bpy.props.StringProperty(
        name="Tek Kanal",
        description=(
            "Doldurulursa sadece bu ARKit kanali uygulanir (ornek: jawOpen). "
            "Once tek kanalla dogrulamak icin -- 51 kanali birden baglarsan "
            "hangi katmanin bozuk oldugunu ayirt edemezsin"
        ),
        default="",
    )  # type: ignore[valid-type]

    calibrate: bpy.props.BoolProperty(
        name="Notr Kalibrasyonu",
        description="Ilk 30 karenin ortalamasini notr kabul edip cikar",
        default=True,
    )  # type: ignore[valid-type]

    smooth: bpy.props.BoolProperty(name="One-Euro Filtre", default=False)  # type: ignore[valid-type]
    mirror: bpy.props.BoolProperty(
        name="Sol/Sag Aynala",
        description="Kamera goruntusunde kullanicinin solu karakterin saginda kalir",
        default=False,
    )  # type: ignore[valid-type]

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        instance = get_rig_instance(context)
        if instance is None:
            self.report({"ERROR"}, "Character DNA rig instance bulunamadi. Once bir MetaHuman ice aktar.")
            return {"CANCELLED"}

        face_board = getattr(instance, "face_board", None)
        if face_board is None:
            self.report({"ERROR"}, "Rig instance'in face board'u yok.")
            return {"CANCELLED"}

        try:
            mapping = Mapping.load()
            frames = list(read_take(self.filepath))
        except (OSError, ValueError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        if not frames:
            self.report({"ERROR"}, "Cekim bos.")
            return {"CANCELLED"}

        bones, missing = resolve_bones(face_board, mapping)
        if missing:
            self.report({"WARNING"}, f"{len(missing)} kemik face board'da yok, atlandi: {missing[:5]}")
        if not bones:
            self.report({"ERROR"}, "Tablodaki hicbir kemik face board'da bulunamadi.")
            return {"CANCELLED"}

        only = {self.single_channel.strip()} if self.single_channel.strip() else None
        if only:
            unknown = only - set(mapping.table)
            if unknown:
                self.report({"ERROR"}, f"tabloda olmayan kanal: {unknown.pop()}")
                return {"CANCELLED"}

        profile = settings_profile(get_settings(context))
        profile.mirror = self.mirror
        if self.calibrate:
            profile.neutral = build_neutral(iter(frames), sample_count=30)

        filters = ChannelFilters() if self.smooth else None

        settings = get_settings(context)
        head_bone, head_tracker = None, None
        if settings is None or settings.head_motion:
            head_bone, problem = resolve_head_bone(instance)
            if head_bone is None:
                self.report({"WARNING"}, f"kafa hareketi atlandi: {problem}")
            else:
                head_tracker = HeadTracker(
                    mapping=self.head_mapping(context),
                    influence=settings.head_influence if settings else 1.0,
                )

        # Bake sirasinda sahne guncellemesini kapat. Acik birakilirsa her
        # keyframe_insert depsgraph handler'ini tetikler ve RigLogic 150 kez
        # bosuna calisir.
        window_manager_properties = get_addon_properties(context.window_manager)
        if window_manager_properties is not None and not hasattr(
            window_manager_properties, "evaluate_dependency_graph"
        ):
            window_manager_properties = None

        previous = None
        if window_manager_properties is not None:
            previous = window_manager_properties.evaluate_dependency_graph
            window_manager_properties.evaluate_dependency_graph = False

        # Cekimi ZAMAN DAMGASINA gore sahne fps'ine otur, kare kareye 1:1
        # degil. Gerekcesi ve olculen sayilar core/take.py:plan_scene_frames.
        scene = context.scene
        fps = scene.render.fps / scene.render.fps_base

        prepared = []
        for take_frame in frames:
            blendshapes = profile.apply(take_frame.bs)
            if filters:
                # filtre cekim sirasina bagli, PLANLAMADAN ONCE uygulanmali
                blendshapes = filters(blendshapes, take_frame.t)
            prepared.append((take_frame.t, blendshapes, take_frame.head))

        plan = plan_scene_frames(prepared, fps, self.start_frame)
        merged_count = len(frames) - len(plan)

        written = 0
        try:
            for scene_frame in sorted(plan):
                blendshapes, head = plan[scene_frame]
                values = mapping.apply(blendshapes, only=only)

                for axis_key, value in values.items():
                    bone_name, axis_index = split_axis_key(axis_key)
                    pose_bone = bones.get(bone_name)
                    if pose_bone is None:
                        continue
                    pose_bone.location[axis_index] = value
                    pose_bone.keyframe_insert(
                        data_path="location", index=axis_index, frame=scene_frame
                    )

                if head_tracker is not None and head:
                    quaternion = head_tracker.feed(head.get("rot"))
                    if quaternion is not None:
                        apply_head_rotation(head_bone, quaternion)
                        head_bone.keyframe_insert(
                            data_path="rotation_quaternion", frame=scene_frame
                        )

                written += 1
        finally:
            if window_manager_properties is not None and previous is not None:
                window_manager_properties.evaluate_dependency_graph = previous

        context.scene.frame_start = self.start_frame
        # plan seyrek olabilir (cekimde duraklama), son kare sayidan turetilemez
        context.scene.frame_end = max(plan) if plan else self.start_frame
        context.scene.frame_set(self.start_frame)

        # tek seferde guncelle
        try:
            instance.evaluate(component="head")
        except Exception as error:  # noqa: BLE001 - addon surumune gore degisiyor
            self.report({"WARNING"}, f"evaluate hatasi: {error}")

        self.report(
            {"INFO"},
            f"{written} kare bake edildi ({len(bones)} kemik"
            + f", {fps:.0f} fps"
            + (f", {len(frames) - len(plan)} kare birlestirildi" if len(plan) < len(frames) else "")
            + (f", sadece {self.single_channel}" if only else "")
            + ")",
        )
        return {"FINISHED"}


# Calisan canli oturumun izi. Modal operator ornekleri arasinda paylasilan
# tek yer burasi: Blender bir modal'i olduren her yolda cancel() cagirmiyor
# (dosya acma, hata, script yeniden yukleme), o durumda soket sizip portu
# tutmaya devam ediyor ve bir sonraki baslatma WinError 10048 aliyor.
_LIVE_SESSION: dict = {"socket": None, "running": False, "tracker": None, "operator": None}

# Canli kayit durumu. Operator ozelligi DEGIL, cunku kullanici kaydi canli
# yayin devam ederken acip kapatabilmeli; operator ozelligi olsaydi her
# denemede yayini kapatip acmak gerekirdi (ayni gerekce kalibrasyon
# ayarlarinda da gecerli).
_RECORD: dict = {
    "active": False,
    "origin": None,   # duvar saati baslangici
    "start": 1,       # ilk sahne karesi
    "frame": None,    # su an bekleyen sahne karesi
    "written": 0,
    "pending": None,
}


def reset_recording() -> None:
    _RECORD.update(active=False, origin=None, start=1, frame=None, written=0, pending=None)


def stop_live_session() -> bool:
    """Sizmis ya da calisan canli oturumu kapat. Kapatildiysa True."""
    existing = _LIVE_SESSION.get("socket")
    _LIVE_SESSION["socket"] = None
    _LIVE_SESSION["running"] = False
    _LIVE_SESSION["tracker"] = None
    _LIVE_SESSION["operator"] = None
    if existing is None:
        return False
    try:
        existing.close()
    except OSError:
        pass
    return True


class CHARFACE_OT_stop_live(bpy.types.Operator):
    """Calisan canli yakalamayi durdurur ve portu serbest birakir"""

    bl_idname = "charface.stop_live"
    bl_label = "Canli Yakalamayi Durdur"

    def execute(self, context):
        if stop_live_session():
            self.report({"INFO"}, "Canli oturum kapatildi, port serbest.")
        else:
            self.report({"INFO"}, "Calisan canli oturum yok.")
        return {"FINISHED"}


class CHARFACE_OT_live(HeadMappingMixin, bpy.types.Operator):
    """UDP'den gelen canli yuz verisini face board'a uygular"""

    bl_idname = "charface.live"
    bl_label = "Canli Yuz Yakalama"

    _timer = None
    _head_bone = None
    _head_tracker = None
    _neutral_samples = None
    _realism = None
    _last_evaluate = 0.0

    def port(self, context) -> int:
        settings = get_settings(context)
        return settings.port if settings else 11111
    _socket = None
    _mapping = None
    _bones = None
    _instance = None
    _filters = None
    _profile = None

    def execute(self, context):
        import socket

        instance = get_rig_instance(context)
        if instance is None or getattr(instance, "face_board", None) is None:
            self.report({"ERROR"}, "Character DNA rig instance / face board bulunamadi.")
            return {"CANCELLED"}

        try:
            self._mapping = Mapping.load()
        except (OSError, ValueError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        self._bones, missing = resolve_bones(instance.face_board, self._mapping)
        if not self._bones:
            self.report({"ERROR"}, "Tablodaki hicbir kemik face board'da bulunamadi.")
            return {"CANCELLED"}
        if missing:
            self.report({"WARNING"}, f"{len(missing)} kemik face board'da yok.")

        settings = get_settings(context)
        self._instance = instance
        self._filters = ChannelFilters() if (settings is None or settings.smooth) else None
        self._profile = settings_profile(settings)
        self._neutral_samples = []

        # Kaydedilmis aralik varsa ve bellekte yoksa otomatik yukle --
        # kullanicinin ROM cekimi Blender yeniden baslayinca kaybolmasin.
        if not _CHANNEL_RANGES["profiles"]:
            loaded = load_ranges_from(resolve_profile_path(settings))
            if loaded:
                self.report({"INFO"}, f"kanal araliklari yuklendi ({loaded} kanal)")

        self._realism = RealismLayer()

        self._head_bone, self._head_tracker = None, None
        if settings is None or settings.head_motion:
            self._head_bone, problem = resolve_head_bone(instance)
            if self._head_bone is None:
                self.report({"WARNING"}, f"kafa hareketi atlandi: {problem}")
            else:
                self._head_tracker = HeadTracker(
                    mapping=self.head_mapping(context),
                    influence=settings.head_influence if settings else 1.0,
                )

        if _LIVE_SESSION["running"]:
            self.report(
                {"ERROR"},
                "Canli yakalama zaten calisiyor. Viewport'a tiklayip ESC'ye bas, "
                "ya da 'Canli Yakalamayi Durdur' tusunu kullan.",
            )
            return {"CANCELLED"}

        # Onceki oturum temiz kapanmadiysa soketi burada birakmis olabilir.
        if stop_live_session():
            self.report({"WARNING"}, "Onceki canli oturumun soketi aciktı, kapatildi.")

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.bind(("0.0.0.0", self.port(context)))
            self._socket.setblocking(False)
        except OSError as error:
            self._socket = None
            self.report(
                {"ERROR"},
                f"port {self.port(context)} acilamadi: {error}. "
                "Baska bir program portu tutuyor olabilir (ornegin ikinci bir Blender "
                "ya da Unreal Live Link). Portu degistirip tekrar dene.",
            )
            return {"CANCELLED"}

        _LIVE_SESSION["socket"] = self._socket
        _LIVE_SESSION["running"] = True
        _LIVE_SESSION["tracker"] = self._head_tracker
        _LIVE_SESSION["operator"] = self
        _LIVE_STATUS.update(
            {
                "packets": 0,
                "badPackets": 0,
                "merged": 0,
                "source": "-",
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "face": False,
                "tongue": 0.0,
                "ranges": "-",
            }
        )

        self._timer = context.window_manager.event_timer_add(1 / 60, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, f"UDP {self.port(context)} dinleniyor. Durdurmak icin ESC.")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC"}:
            self.cancel(context)
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        import json
        import math
        import time

        tick_start = time.perf_counter()

        # Kuyrukta birikeni bosalt -- gecikme birikmesin.
        #
        # Ama SADECE sonuncuyu almak kirpmayi kirpiyor: gercek ARKit
        # kaydinda olculdu, kapanma fazi 50-120 ms yani 60 fps'te 3-7 kare.
        # Blender bir tikte iki paket cekerse tepe noktasi (goz tam kapali
        # ani) atilan pakette kalabiliyor ve goz hic tam kapanmiyor --
        # "olu bakis" etkisinin sebeplerinden biri bu.
        #
        # Cozum: bir tikte birden fazla paket geldiyse kanal basina MAKSIMUM
        # al. Tek paket geldiginde (normal durum) davranis degismiyor; geride
        # kaldigimizda ~16 ms'lik pencerede maksimum, yavas kanallar icin
        # sonuncuyla neredeyse ayni, hizli kanallar icin tepeyi koruyor.
        packets = []
        while True:
            try:
                payload, _ = self._socket.recvfrom(65535)
            except (BlockingIOError, OSError):
                break
            packets.append(payload)

        if not packets:
            return {"PASS_THROUGH"}

        decoded = []
        for payload in packets:
            frame = decode_packet(payload, _LIVE_STATUS["packets"] + 1)
            if frame is None:
                _LIVE_STATUS["badPackets"] += 1
            else:
                decoded.append(frame)

        if not decoded:
            return {"PASS_THROUGH"}

        data = decoded[-1]
        if len(decoded) > 1:
            _LIVE_STATUS["merged"] += len(decoded) - 1
            birlesik = dict(data.get("bs") or {})
            for onceki in decoded[:-1]:
                for kanal, deger in (onceki.get("bs") or {}).items():
                    if deger > birlesik.get(kanal, 0.0):
                        birlesik[kanal] = deger
            data = dict(data, bs=birlesik)

        blendshapes = {k: float(v) for k, v in (data.get("bs") or {}).items()}
        timestamp = float(data.get("t", 0.0))
        head = data.get("head") or {}

        settings = get_settings(context)

        # Ayarlar SAHNEDE, o yuzden kullanici canliyken degistirdiginde
        # aninda etki etmeli. Her karede tazeliyoruz -- ucuz, sadece sozluk.
        if self._head_tracker is not None:
            self._head_tracker.mapping = settings_head_mapping(settings)
            self._head_tracker.influence = settings.head_influence if settings else 1.0
        # Ayarlar canliyken degisebiliyor; profili her karede yeniden kur.
        # Ogrenilen notru koruyoruz, yoksa her karede sifirlanirdi.
        if settings is not None:
            self._profile = settings_profile(settings, neutral=self._profile.neutral)

            # aralik ogrenme (ROM cekimi)
            if settings.learning_range:
                learner = _CHANNEL_RANGES["learner"]
                if learner is None:
                    learner = _CHANNEL_RANGES["learner"] = RangeLearner()
                learner.feed(blendshapes)
                _CHANNEL_RANGES["profiles"] = learner.channel_profiles()
                _LIVE_STATUS["ranges"] = learner.summary()

            # notr ogrenme: ilk 30 karenin ortalamasi
            if settings.expression_neutral and len(self._neutral_samples) < 30:
                self._neutral_samples.append(dict(blendshapes))
                if len(self._neutral_samples) == 30:
                    totals: dict[str, float] = {}
                    for sample in self._neutral_samples:
                        for key, value in sample.items():
                            totals[key] = totals.get(key, 0.0) + value
                    self._profile.neutral = {
                        k: v / len(self._neutral_samples) for k, v in totals.items()
                    }

        _LIVE_STATUS["packets"] += 1
        _LIVE_STATUS["face"] = bool(blendshapes)
        # tongueOut sadece iPhone'dan gelir; kaynak ayrimi icin iyi bir isaret
        if "tongueOut" in blendshapes:
            _LIVE_STATUS["source"] = "iPhone (Live Link)"
            _LIVE_STATUS["tongue"] = blendshapes["tongueOut"]
        else:
            _LIVE_STATUS["source"] = "webcam (MediaPipe)"

        if self._head_tracker is not None and head and (settings is None or settings.head_motion):
            quaternion = self._head_tracker.feed(head.get("rot"))
            if quaternion is not None:
                apply_head_rotation(self._head_bone, quaternion)
            pitch, yaw, roll = self._head_tracker.last_angles
            _LIVE_STATUS["pitch"] = math.degrees(pitch)
            _LIVE_STATUS["yaw"] = math.degrees(yaw)
            _LIVE_STATUS["roll"] = math.degrees(roll)

        blendshapes = self._profile.apply(blendshapes)
        if self._filters:
            blendshapes = self._filters(blendshapes, timestamp)

        values = self._mapping.apply(blendshapes)

        # ARKit'in kanali olmayan kontrolleri turet (goz kapaklari, kapak
        # baskisi, mikro-sakkad) ve esleme ciktisiyla birlestir
        if self._realism is not None and (settings is None or settings.realism):
            self._realism.eyelid_follow = settings.eyelid_follow if settings else 0.85
            self._realism.lower_lid_follow = settings.lower_lid_follow if settings else 0.25
            self._realism.lid_press = settings.lid_press if settings else 0.5
            self._realism.saccades = settings.saccades if settings else 1.0
            self._realism.pucker_close = settings.pucker_close if settings else 0.7
            extra = self._realism.extra_axes(values, timestamp)
            values = merge(values, extra, self._mapping.limits)

        for axis_key, value in values.items():
            bone_name, axis_index = split_axis_key(axis_key)
            pose_bone = self._bones.get(bone_name)
            if pose_bone is not None:
                pose_bone.location[axis_index] = value

        self.record(context, settings, values)
        apply_ms = (time.perf_counter() - tick_start) * 1000.0

        # Onizleme kisma.
        #
        # KAYIT BUNDAN ETKILENMIYOR: keyframe'ler yukarida `values`'tan
        # yaziliyor, ekranda gorunenden degil. Yani agir sahnede viewport 15
        # Hz'e dusse bile animasyon tam cozunurlukte kaydediliyor. Sac/sakal
        # particle'lari ve 800+ kemikli armature ile evaluate() 30 Hz'e
        # yetismiyor; kismazsak gecikme birikiyor ve yuz geriden geliyor.
        limit = settings.preview_hz if settings else 0
        now = time.perf_counter()
        if limit and now - self._last_evaluate < 1.0 / limit:
            _LIVE_STATUS["skipped"] += 1
            return {"PASS_THROUGH"}
        self._last_evaluate = now

        evaluate_start = now
        self._instance.evaluate(component="head")
        evaluate_ms = (time.perf_counter() - evaluate_start) * 1000.0

        # Hareketli ortalama. Tek olcum yaniltici: ilk degerlendirme onbellek
        # kuruyor, arada bir de GC giriyor. Panelde saniyede bir okunuyor,
        # ani sicramalar isaretsiz kalmasin diye yumusatiliyor.
        for key, value in (("evalMs", evaluate_ms), ("applyMs", apply_ms)):
            _LIVE_STATUS[key] = _LIVE_STATUS[key] * 0.9 + value * 0.1
        total = _LIVE_STATUS["evalMs"] + _LIVE_STATUS["applyMs"]
        _LIVE_STATUS["budgetHz"] = 1000.0 / total if total > 0.01 else 0.0

        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()

        return {"PASS_THROUGH"}

    def record(self, context, settings, values) -> None:
        """Canliyken timeline'a keyframe yazar.

        Zamanlama DUVAR SAATINDEN geliyor, paket sayacindan degil: paketler
        30 fps geliyor, sahne genelde 24 fps ve arada kare dusebiliyor. Paket
        basina bir sahne karesi yazmak animasyonu yavaslatir -- ayni hata
        bake yolunda olculmustu, 37 saniyelik cekim 46 saniye suruyordu.

        Ayni sahne karesine birden fazla paket dustugunde eksenlerin MUTLAK
        DEGERCE en buyugu tutuluyor. Sonuncuyu almak 89 ms suren goz kirpma
        tepesini kaybettiriyor. Eksenler isaretli oldugu icin (bir eksen iki
        yone gidiyor) max degil abs-max.
        """
        import time

        if settings is None or not settings.recording:
            if _RECORD["active"]:
                self.finish_recording(context)
            return

        scene = context.scene
        fps = scene.render.fps / scene.render.fps_base
        now = time.time()

        if not _RECORD["active"]:
            # kayit oynatma kafasinin durdugu yerden baslar
            _RECORD.update(
                active=True,
                origin=now,
                start=scene.frame_current,
                frame=None,
                written=0,
                pending=None,
            )

        target = _RECORD["start"] + round((now - _RECORD["origin"]) * fps)

        pending = _RECORD["pending"]
        if pending is None or _RECORD["frame"] != target:
            if pending is not None:
                self.write_keyframes(context, _RECORD["frame"], pending)
            _RECORD["frame"] = target
            _RECORD["pending"] = dict(values)
        else:
            for axis_key, value in values.items():
                if abs(value) > abs(pending.get(axis_key, 0.0)):
                    pending[axis_key] = value

    def write_keyframes(self, context, scene_frame, values) -> None:
        for axis_key, value in values.items():
            bone_name, axis_index = split_axis_key(axis_key)
            pose_bone = self._bones.get(bone_name)
            if pose_bone is None:
                continue
            pose_bone.location[axis_index] = value
            pose_bone.keyframe_insert(data_path="location", index=axis_index, frame=scene_frame)

        if self._head_bone is not None:
            self._head_bone.keyframe_insert(
                data_path="rotation_quaternion", frame=scene_frame
            )

        _RECORD["written"] += 1
        context.scene.frame_current = scene_frame

    def finish_recording(self, context) -> None:
        """Kayit kapatilinca son bekleyen kareyi yaz ve sahne araligini kur."""
        if _RECORD["pending"] is not None and _RECORD["frame"] is not None:
            self.write_keyframes(context, _RECORD["frame"], _RECORD["pending"])
        if _RECORD["written"]:
            scene = context.scene
            scene.frame_start = min(scene.frame_start, _RECORD["start"])
            scene.frame_end = max(scene.frame_end, _RECORD["frame"])
            _LIVE_STATUS["recorded"] = (
                f"{_RECORD['written']} kare yazildi ({_RECORD['start']}-{_RECORD['frame']})"
            )
        reset_recording()

    def cancel(self, context):
        if _RECORD["active"]:
            self.finish_recording(context)
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._socket = None
        stop_live_session()


class CHARFACE_PT_panel(bpy.types.Panel):
    bl_label = "charface"
    bl_idname = "CHARFACE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "charface"

    def draw(self, context):
        layout = self.layout
        instance = get_rig_instance(context)
        settings = get_settings(context)

        if instance is None:
            layout.label(text="Rig instance yok", icon="ERROR")
            layout.label(text="Once Character DNA ile MetaHuman ice aktar")
            return

        layout.label(text=f"Rig: {getattr(instance, 'name', '?')}", icon="OUTLINER_OB_ARMATURE")

        box = layout.box()
        if _LIVE_SESSION["running"]:
            row = box.row()
            row.label(text="Canli yakalama calisiyor", icon="REC")
            row.operator(CHARFACE_OT_stop_live.bl_idname, text="", icon="SNAP_FACE")
            box.label(text="Durdurmak icin viewport'ta ESC")

            if settings is not None:
                record = box.row()
                record.scale_y = 1.4
                record.alert = settings.recording
                record.prop(
                    settings,
                    "recording",
                    toggle=True,
                    icon="REC" if settings.recording else "RADIOBUT_OFF",
                    text="KAYDEDILIYOR" if settings.recording else "Timeline'a Kaydet",
                )
                if settings.recording:
                    box.label(
                        text=f"kare {_RECORD['frame'] or context.scene.frame_current}"
                        f"  ({_RECORD['written']} yazildi)"
                    )
                else:
                    box.label(text="Oynatma kafasini konumlandir, sonra bas")
                if _LIVE_STATUS["recorded"] != "-":
                    box.label(text=_LIVE_STATUS["recorded"], icon="CHECKMARK")
        else:
            if settings is not None:
                box.prop(settings, "port")
            box.operator(CHARFACE_OT_live.bl_idname, icon="RADIOBUT_ON")
        box.operator(CHARFACE_OT_bake_take.bl_idname, icon="IMPORT")

        box = layout.box()
        box.label(text="Cozucu hazirligi", icon="RESTRICT_RENDER_OFF")
        box.operator(CHARFACE_OT_render_head.bl_idname, icon="RENDER_STILL")
        box.label(text="2. adim terminalde: scripts/08_landmarks.py")
        box.operator(CHARFACE_OT_build_correspondence.bl_idname, icon="SNAP_VERTEX")


class CHARFACE_PT_face(bpy.types.Panel):
    bl_label = "Yuz Kalibrasyonu"
    bl_idname = "CHARFACE_PT_face"
    bl_parent_id = "CHARFACE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "charface"

    def draw(self, context):
        layout = self.layout
        settings = get_settings(context)
        if settings is None:
            layout.label(text="Ayarlar yuklenemedi", icon="ERROR")
            return

        box = layout.box()
        box.label(text="Agiz", icon="USER")
        box.prop(settings, "jaw_gain", slider=True)
        box.prop(settings, "lips_gain", slider=True)
        box.prop(settings, "lip_close_gain", slider=True)
        box.prop(settings, "mouth_deadzone", slider=True)

        box = layout.box()
        box.label(text="Ust yuz")
        box.prop(settings, "eyes_gain", slider=True)
        box.prop(settings, "brows_gain", slider=True)

        layout.prop(settings, "expression_neutral")

        box = layout.box()
        box.label(text="Kanal araligi", icon="ARROW_LEFTRIGHT")
        box.label(text="Kanallar 1.0'a ulasmiyor, goz tam kapanmiyor")
        box.prop(settings, "use_channel_range")
        row = box.row()
        row.prop(settings, "learning_range", toggle=True, icon="REC")
        row.operator(CHARFACE_OT_clear_ranges.bl_idname, text="", icon="X")
        if _LIVE_STATUS["ranges"] != "-":
            box.label(text=_LIVE_STATUS["ranges"])
        box.label(text="Ac, tum ifadeleri sonuna kadar yap, kapat")

        row = box.row(align=True)
        row.operator(CHARFACE_OT_save_ranges.bl_idname, icon="FILE_TICK")
        row.operator(CHARFACE_OT_load_ranges.bl_idname, icon="FILE_REFRESH")
        box.prop(settings, "profile_path", text="")
        if _CHANNEL_RANGES["path"]:
            box.label(text=Path(_CHANNEL_RANGES["path"]).name, icon="CHECKMARK")

        box = layout.box()
        box.label(text="Turetilmis goz kontrolleri", icon="HIDE_OFF")
        box.label(text="ARKit'in kanali olmayan 15 kontrol")
        box.prop(settings, "realism")
        column = box.column()
        column.active = settings.realism
        column.prop(settings, "eyelid_follow", slider=True)
        column.prop(settings, "lower_lid_follow", slider=True)
        column.prop(settings, "lid_press", slider=True)
        column.prop(settings, "saccades", slider=True)
        column.separator()
        column.prop(settings, "pucker_close", slider=True)


class CHARFACE_PT_calibration(bpy.types.Panel):
    bl_label = "Kafa Kalibrasyonu"
    bl_idname = "CHARFACE_PT_calibration"
    bl_parent_id = "CHARFACE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "charface"

    def draw(self, context):
        layout = self.layout
        settings = get_settings(context)
        if settings is None:
            layout.label(text="Ayarlar yuklenemedi", icon="ERROR")
            return

        # Canli okuma kalibrasyonun can damari: hangi acinin ne kadar
        # gittigini gormeden hangi eksenin yanlis oldugu tahmin isi olur.
        box = layout.box()
        if _LIVE_SESSION["running"]:
            box.label(
                text=f"paket {_LIVE_STATUS['packets']}"
                + ("   yuz VAR" if _LIVE_STATUS["face"] else "   yuz YOK"),
                icon="INFO",
            )
            box.label(text=f"kaynak: {_LIVE_STATUS['source']}")
            if _LIVE_STATUS["badPackets"]:
                box.label(text=f"cozulemeyen paket: {_LIVE_STATUS['badPackets']}", icon="ERROR")
            if _LIVE_STATUS["merged"]:
                # geride kaliyoruz demektir; tepe korunuyor ama viewport yavas
                box.label(text=f"birlestirilen kare: {_LIVE_STATUS['merged']}", icon="TIME")
            if _LIVE_STATUS["source"].startswith("iPhone"):
                box.label(text=f"dil (tongueOut) {_LIVE_STATUS['tongue']:.2f}")
            column = box.column(align=True)
            column.label(text=f"yaw    {_LIVE_STATUS['yaw']:+6.1f}°  saga sola cevirme")
            column.label(text=f"pitch  {_LIVE_STATUS['pitch']:+6.1f}°  asagi yukari")
            column.label(text=f"roll   {_LIVE_STATUS['roll']:+6.1f}°  yana yatirma")
            box.operator(CHARFACE_OT_reset_neutral.bl_idname, icon="LOOP_BACK")
        else:
            box.label(text="Canli degil - sayilar canliyken gorunur", icon="INFO")

        layout.prop(settings, "head_motion")

        column = layout.column()
        column.active = settings.head_motion
        column.prop(settings, "head_influence", slider=True)

        for label, prefix in (
            ("Yaw - saga sola cevirme", "yaw"),
            ("Pitch - asagi yukari", "pitch"),
            ("Roll - yana yatirma", "roll"),
        ):
            angle_box = column.box()
            angle_box.label(text=label)
            row = angle_box.row(align=True)
            row.prop(settings, f"{prefix}_axis", text="")
            row.prop(settings, f"{prefix}_invert", toggle=True, icon="ARROW_LEFTRIGHT")
            angle_box.prop(settings, f"{prefix}_gain", slider=True)

        box = layout.box()
        box.label(text="Eksen testi", icon="ORIENTATION_GIMBAL")
        box.label(text="Bas ve karakter ne yapiyor bak")
        box.prop(settings, "test_angle", slider=True)
        row = box.row(align=True)
        for axis in AXES:
            row.operator(CHARFACE_OT_test_head_axis.bl_idname, text=axis).axis = axis
        box.operator(
            CHARFACE_OT_test_head_axis.bl_idname, text="Sifirla", icon="LOOP_BACK"
        ).axis = "RESET"

        layout.operator(CHARFACE_OT_reset_calibration.bl_idname, icon="FILE_REFRESH")

        layout.separator()
        layout.prop(settings, "mirror")
        layout.prop(settings, "smooth")


# --------------------------------------------------------------------------
# Sahne performansi
# --------------------------------------------------------------------------
#
# MetaHuman sahnesi agirdir ve agirligin buyuk kismi GORUNMEZ bir anahtara
# bagli: addon'un `evaluate_dependency_graph` bayragi. Acikken RigLogic HER
# depsgraph guncellemesinde kosuyor (rig_logic.py:37) -- yani objeyi
# secmek, kare degistirmek, particle duzenlemek, hepsi 800+ kemikli rigi
# yeniden hesaplatiyor. Addon bunu sadece bake/import sirasinda programatik
# kapatiyor, hicbir arayuzde acik degil.
#
# Buradaki tuslar kalici sahne ayarlarini degistirir; canli yakalamayla
# ilgisi yoktur, her zaman kullanilabilir.


def riglogic_auto(context):
    """Addon'un otomatik RigLogic degerlendirmesi acik mi. Yoksa None."""
    properties = get_addon_properties(context.window_manager)
    if properties is None or not hasattr(properties, "evaluate_dependency_graph"):
        return None
    return properties.evaluate_dependency_graph


def particle_modifiers(context):
    for obj in context.scene.objects:
        for modifier in getattr(obj, "modifiers", []):
            if modifier.type == "PARTICLE_SYSTEM":
                yield obj, modifier


class CHARFACE_OT_toggle_riglogic(bpy.types.Operator):
    """RigLogic'in her sahne degisikliginde kosmasini ac/kapat"""

    bl_idname = "charface.toggle_riglogic"
    bl_label = "RigLogic Otomatik Degerlendirme"

    def execute(self, context):
        properties = get_addon_properties(context.window_manager)
        if properties is None or not hasattr(properties, "evaluate_dependency_graph"):
            self.report({"ERROR"}, "addon'un evaluate_dependency_graph bayragi yok")
            return {"CANCELLED"}
        properties.evaluate_dependency_graph = not properties.evaluate_dependency_graph
        state = "acik" if properties.evaluate_dependency_graph else "KAPALI"
        self.report({"INFO"}, f"RigLogic otomatik degerlendirme {state}")
        return {"FINISHED"}


class CHARFACE_OT_evaluate_now(bpy.types.Operator):
    """Rigi bir kez degerlendir (otomatik kapaliyken yuzu guncellemek icin)"""

    bl_idname = "charface.evaluate_now"
    bl_label = "Simdi Degerlendir"

    def execute(self, context):
        instance = get_rig_instance(context)
        if instance is None:
            self.report({"ERROR"}, "rig instance bulunamadi")
            return {"CANCELLED"}
        try:
            instance.evaluate(component="head")
        except Exception as error:  # noqa: BLE001 - addon surumune gore degisiyor
            self.report({"ERROR"}, f"evaluate hatasi: {error}")
            return {"CANCELLED"}
        return {"FINISHED"}


# Viewport'ta hafifletirken kullanilan degerler. Hepsi RENDER'i etkilemez --
# `display_*` ve `child_percent` sadece viewport icin; `render_step` ve
# `rendered_child_count` ayri alanlar, onlara dokunulmuyor.
HAIR_LIGHT = {"display_percentage": 10, "display_step": 1, "child_percent": 0}
HAIR_SAVE_KEY = "charface_hair_backup"


def hair_systems(context):
    """(obj, particle system, settings) -- sahnedeki sac sistemleri."""
    seen = set()
    for obj in context.scene.objects:
        for system in getattr(obj, "particle_systems", []):
            settings = system.settings
            if settings is None or settings.as_pointer() in seen:
                continue
            seen.add(settings.as_pointer())
            yield obj, system, settings


def viewport_strands(settings) -> int:
    """Viewport'ta gercekten cizilen tel sayisi.

    Panelde bu sayiyi gostermek sart: kullanici `count` alanini goruyor ama
    asil yuk cocuk particle'larla carpiliyor. 1000 tel x 10 cocuk = 10.000,
    ve her biri 2^display_step segmente bolunuyor.
    """
    parents = settings.count * settings.display_percentage / 100.0
    children = settings.child_percent if settings.child_type != "NONE" else 1
    return int(parents * max(children, 1))


class CHARFACE_OT_lighten_hair(bpy.types.Operator):
    """Sac particle'larini viewport'ta hafiflet (render etkilenmez)"""

    bl_idname = "charface.lighten_hair"
    bl_label = "Saci Hafiflet"

    def execute(self, context):
        found = list(hair_systems(context))
        if not found:
            self.report({"WARNING"}, "sahnede particle sistemi yok")
            return {"CANCELLED"}

        # Yedek particle settings datablock'unda durur, modul sozlugunde
        # degil: dosya kaydedilip acildiginda da geri alinabilsin.
        restoring = any(HAIR_SAVE_KEY in settings for _, _, settings in found)
        for _, _, settings in found:
            if restoring:
                backup = settings.get(HAIR_SAVE_KEY)
                if backup:
                    for key, value in backup.items():
                        setattr(settings, key, value)
                    del settings[HAIR_SAVE_KEY]
            else:
                settings[HAIR_SAVE_KEY] = {k: getattr(settings, k) for k in HAIR_LIGHT}
                for key, value in HAIR_LIGHT.items():
                    setattr(settings, key, value)

        self.report(
            {"INFO"},
            f"{len(found)} sac sistemi {'geri alindi' if restoring else 'hafifletildi'}",
        )
        return {"FINISHED"}


class CHARFACE_OT_toggle_particles(bpy.types.Operator):
    """Sahnedeki tum particle sistemlerini viewport'ta gizle/goster"""

    bl_idname = "charface.toggle_particles"
    bl_label = "Particle Sistemleri"

    def execute(self, context):
        found = list(particle_modifiers(context))
        if not found:
            self.report({"WARNING"}, "sahnede particle sistemi yok")
            return {"CANCELLED"}
        # hepsi acikken kapat, aksi halde hepsini ac
        target = not all(modifier.show_viewport for _, modifier in found)
        for _, modifier in found:
            modifier.show_viewport = target
        self.report({"INFO"}, f"{len(found)} particle sistemi {'acildi' if target else 'gizlendi'}")
        return {"FINISHED"}


class CHARFACE_PT_performance(bpy.types.Panel):
    bl_label = "Sahne Performansi"
    bl_idname = "CHARFACE_PT_performance"
    bl_parent_id = "CHARFACE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "charface"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.label(text="RigLogic", icon="ARMATURE_DATA")
        auto = riglogic_auto(context)
        if auto is None:
            box.label(text="addon bayragi bulunamadi", icon="ERROR")
        else:
            row = box.row()
            row.alert = auto
            row.operator(
                CHARFACE_OT_toggle_riglogic.bl_idname,
                text="Otomatik: ACIK (yavas)" if auto else "Otomatik: KAPALI (hizli)",
                icon="CHECKBOX_HLT" if auto else "CHECKBOX_DEHLT",
            )
            if auto:
                box.label(text="Her sahne degisikliginde rig yeniden hesaplaniyor")
            else:
                box.operator(CHARFACE_OT_evaluate_now.bl_idname, icon="FILE_REFRESH")
                box.label(text="Yuz kendiliginden guncellenmez")

        box = layout.box()
        box.label(text="Sac / kas / sakal", icon="PARTICLES")
        hair = list(hair_systems(context))
        modifiers = list(particle_modifiers(context))
        if not hair:
            box.label(text="sahnede particle sistemi yok")
        else:
            total = sum(viewport_strands(settings) for _, _, settings in hair)
            box.label(text=f"{len(hair)} sistem, viewport'ta {total:,} tel".replace(",", "."))
            lightened = any(HAIR_SAVE_KEY in settings for _, _, settings in hair)
            row = box.row()
            row.alert = not lightened
            row.operator(
                CHARFACE_OT_lighten_hair.bl_idname,
                text="Geri al (tam cozunurluk)" if lightened else "Saci hafiflet",
                icon="MOD_PARTICLES",
            )
            box.label(text="Render etkilenmez, sadece viewport")

            for _, system, settings in hair:
                line = box.box()
                line.label(text=system.name, icon="OUTLINER_OB_HAIR" if False else "PARTICLES")
                line.label(
                    text=f"{settings.count} tel x "
                    f"{settings.child_percent if settings.child_type != 'NONE' else 0} cocuk"
                    f"  ->  {viewport_strands(settings):,} ".replace(",", ".")
                )
                column = line.column(align=True)
                column.prop(settings, "display_percentage", text="Viewport %")
                column.prop(settings, "child_percent", text="Cocuk (viewport)")
                column.prop(settings, "display_step", text="Tel bolunmesi")

            if modifiers:
                visible = sum(1 for _, modifier in modifiers if modifier.show_viewport)
                box.operator(
                    CHARFACE_OT_toggle_particles.bl_idname,
                    text="Hepsini gizle" if visible else "Hepsini goster",
                    icon="HIDE_ON" if visible else "HIDE_OFF",
                )

        box = layout.box()
        box.label(text="Simplify", icon="MOD_DECIM")
        box.prop(scene.render, "use_simplify", text="Simplify acik")
        column = box.column()
        column.enabled = scene.render.use_simplify
        column.prop(scene.render, "simplify_child_particles", text="Cocuk particle", slider=True)
        column.prop(scene.render, "simplify_subdivision", text="Subdivision")

        box = layout.box()
        box.label(text="Canli yakalamada", icon="REC")
        settings = get_settings(context)
        if settings is not None:
            box.prop(settings, "preview_hz")
            box.label(text="Kayit tam hizda yazilir, kismadan etkilenmez")
            box.prop(settings, "light_capture")
        if _LIVE_STATUS["budgetHz"]:
            box.label(
                text=f"olculen: rig {_LIVE_STATUS['evalMs']:.1f} ms, "
                f"uygula {_LIVE_STATUS['applyMs']:.1f} ms -> {_LIVE_STATUS['budgetHz']:.0f} Hz"
            )
            if _LIVE_STATUS["skipped"]:
                box.label(text=f"{_LIVE_STATUS['skipped']} kare onizlemede atlandi")


CLASSES = (
    CharfaceSettings,
    CHARFACE_OT_bake_take,
    CHARFACE_OT_live,
    CHARFACE_OT_stop_live,
    CHARFACE_OT_reset_neutral,
    CHARFACE_OT_reset_calibration,
    CHARFACE_OT_clear_ranges,
    CHARFACE_OT_render_head,
    CHARFACE_OT_build_correspondence,
    CHARFACE_OT_save_ranges,
    CHARFACE_OT_load_ranges,
    CHARFACE_OT_test_head_axis,
    CHARFACE_OT_toggle_riglogic,
    CHARFACE_OT_evaluate_now,
    CHARFACE_OT_toggle_particles,
    CHARFACE_OT_lighten_hair,
    CHARFACE_PT_panel,
    CHARFACE_PT_face,
    CHARFACE_PT_calibration,
    CHARFACE_PT_performance,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.charface = bpy.props.PointerProperty(type=CharfaceSettings)


def unregister():
    # Addon kapatilirken/yeniden yuklenirken soketi birakma: aksi halde port
    # Blender kapanana kadar tutulu kalir.
    stop_live_session()
    if hasattr(bpy.types.Scene, "charface"):
        del bpy.types.Scene.charface
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
