"""
Imagenes reales -> patrones de entrada para las redes.

Las redes del paquete trabajan con vectores: la red de Hopfield de la
actividad 5 espera 42 valores bipolares (una retina de 7x6), y el perceptron
multicapa acepta cualquier vector real.  Una foto o un escaneo, en cambio, es
una matriz de cientos de pixeles en color, con fondo, iluminacion desigual y el
objeto en cualquier posicion.  Este modulo hace el puente.

Preprocesamiento (`preprocesar`)
--------------------------------
1. **Escala de grises** (luminancia 0..1).
2. **Polaridad e iluminacion** (`corregir_iluminacion`): el borde de la foto
   se toma como fondo para decidir si el trazo es oscuro o claro; se elimina
   el ruido con un filtro de mediana y se compensan sombras y gradientes de
   luz dividiendo por una superficie ajustada al fondo.
3. **Umbral de Otsu**: elige automaticamente el nivel de gris que mejor separa
   trazo y fondo (maximiza la varianza entre clases).  No hay que ajustar nada
   a mano aunque cambie la iluminacion.
4. **Recorte** (`caja_del_trazo`) al rectangulo que contiene el trazo, sin
   dejar que motas aisladas lo agranden, y normalizacion de tamano:
   el recorte se estira hasta llenar la retina (opcionalmente, se rellena para
   conservar la proporcion).  Asi la letra ocupa la retina completa este donde
   este en la foto y tenga el tamano que tenga.
5. **Reduccion** a la resolucion de la retina promediando bloques (cada pixel
   de la retina es la fraccion de trazo de su bloque).
6. **Binarizacion** de esa fraccion (por omision se enciende el pixel si al
   menos el 50 % del bloque es trazo) y codificacion bipolar: trazo = +1,
   fondo = -1.

Generacion de imagenes de prueba (`renderizar_letra`, `distorsionar`)
-------------------------------------------------------------------
Para experimentar sin depender de fotos concretas, se renderizan letras con
tipografias reales (las DejaVu que trae matplotlib) a 160x160 pixeles y se les
aplican distorsiones de "camara": rotacion, desplazamiento, escala, desenfoque,
ruido gaussiano, gradiente de iluminacion y bajo contraste.  Cualquier imagen
propia (PNG, JPG...) pasa exactamente por el mismo `preprocesar`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # Pillow viene instalado como dependencia de matplotlib
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except ImportError as error:  # pragma: no cover
    raise ImportError("El modulo `imagenes` necesita Pillow (se instala con matplotlib)") from error

import matplotlib

DIR_FUENTES = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

#: Tipografias reales usadas en los experimentos (todas vienen con matplotlib).
FUENTES = {
    "sans_negrita": "DejaVuSans-Bold.ttf",
    "sans": "DejaVuSans.ttf",
    "serif_negrita": "DejaVuSerif-Bold.ttf",
    "serif": "DejaVuSerif.ttf",
    "mono_negrita": "DejaVuSansMono-Bold.ttf",
    "mono": "DejaVuSansMono.ttf",
    "sans_oblicua": "DejaVuSans-BoldOblique.ttf",
}

EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Lectura y preprocesamiento
# ---------------------------------------------------------------------------

def a_grises(imagen) -> np.ndarray:
    """Imagen (ruta, PIL o array) -> matriz de luminancia en [0, 1]."""
    if isinstance(imagen, (str, Path)):
        imagen = Image.open(imagen)
    if isinstance(imagen, Image.Image):
        if imagen.mode in ("RGBA", "LA", "P"):
            # la transparencia se compone sobre fondo blanco
            fondo = Image.new("RGBA", imagen.size, (255, 255, 255, 255))
            imagen = Image.alpha_composite(fondo, imagen.convert("RGBA"))
        return np.asarray(imagen.convert("L"), dtype=float) / 255.0
    arreglo = np.asarray(imagen, dtype=float)
    if arreglo.ndim == 3:
        arreglo = arreglo[..., :3] @ np.array([0.299, 0.587, 0.114])
    return arreglo / 255.0 if arreglo.max() > 1.0 else arreglo


def umbral_otsu(grises: np.ndarray, niveles: int = 256) -> float:
    """Umbral de Otsu (1979): maximiza la varianza entre las dos clases."""
    hist, bordes = np.histogram(grises.ravel(), bins=niveles, range=(0.0, 1.0))
    p = hist / max(hist.sum(), 1)
    centros = (bordes[:-1] + bordes[1:]) / 2
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centros)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        varianza_entre = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0))
    varianza_entre = np.nan_to_num(varianza_entre)
    return float(centros[int(np.argmax(varianza_entre))])


def corregir_iluminacion(grises: np.ndarray, lado_max: int = 256) -> np.ndarray:
    """Deja el trazo oscuro sobre un fondo blanco uniforme y sin ruido de sensor.

    1. Las fotos grandes se reducen a `lado_max` pixeles de lado: la retina solo
       tiene 7x6 y el detalle fino no aporta nada (y acelera todo).
    2. Polaridad: el borde de la foto es fondo.  Si el borde es oscuro (tiza
       sobre pizarra, texto claro sobre fondo oscuro) se invierte la imagen.
    3. Un filtro de mediana elimina el ruido de sensor sin redondear los bordes
       (solo si la imagen tiene al menos 64 pixeles de lado).
    4. Iluminacion: se ajusta por minimos cuadrados una superficie cuadratica
       (1, x, y, x^2, xy, y^2) a los pixeles de fondo y se divide la imagen por
       ella; asi desaparecen gradientes de luz y vinetas.  Se repite dos veces,
       reclasificando el fondo con Otsu tras cada correccion.  Como solo se
       usan pixeles de fondo, el grosor del trazo no influye.
    """
    img = Image.fromarray((np.clip(grises, 0, 1) * 255).astype(np.uint8))
    if max(img.size) > lado_max:
        img.thumbnail((lado_max, lado_max), Image.Resampling.LANCZOS)
    a = np.asarray(img, dtype=float) / 255.0
    borde = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if np.median(borde) < 0.5:
        a = 1.0 - a
    if min(a.shape) >= 64:  # en imagenes diminutas la mediana borraria trazos finos
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),
                       dtype=float) / 255.0

    filas, cols = a.shape
    y, x = np.mgrid[0:filas, 0:cols]
    x = x.ravel() / cols - 0.5
    y = y.ravel() / filas - 0.5
    base = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    corregida = a
    for _ in range(2):
        fondo_px = (corregida >= umbral_otsu(corregida)).ravel()
        if fondo_px.sum() < base.shape[1]:
            break
        coef, *_ = np.linalg.lstsq(base[fondo_px], a.ravel()[fondo_px], rcond=None)
        fondo = np.maximum((base @ coef).reshape(a.shape), 0.05)
        corregida = np.clip(a / fondo, 0, 1)
    return corregida


def caja_del_trazo(trazo: np.ndarray, fraccion: float = 0.05):
    """Rectangulo (f0, f1, c0, c1) que contiene el trazo, ignorando motas.

    Se descartan de cada borde las filas y columnas cuya cantidad de trazo es
    menor que `fraccion` de la fila (o columna) mas cargada: una mota aislada
    de ruido no puede agrandar el recorte y desplazar la letra en la retina.
    """
    def extremos(perfil):
        indices = np.flatnonzero(perfil >= max(1.0, fraccion * perfil.max()))
        if indices.size == 0:
            indices = np.flatnonzero(perfil)
        return indices[0], indices[-1] + 1

    f0, f1 = extremos(trazo.sum(axis=1))
    c0, c1 = extremos(trazo.sum(axis=0))
    return f0, f1, c0, c1


def preprocesar(imagen, forma=(7, 6), margen: float = 0.0, umbral_bloque: float = 0.5,
                conservar_proporcion: bool = False, devolver_pasos: bool = False):
    """Convierte una imagen cualquiera en un patron bipolar de `forma` pixeles.

    Parameters
    ----------
    imagen:
        Ruta a un archivo, imagen PIL o array (grises o RGB).
    forma:
        (filas, columnas) de la retina; (7, 6) = 42 neuronas.
    margen:
        Margen alrededor del trazo tras el recorte, como fraccion del lado.
    conservar_proporcion:
        Con False (por omision) el recorte se estira hasta llenar la retina,
        como en la normalizacion habitual de OCR: una letra estrecha usa todas
        las columnas.  Con True se rellena para respetar la proporcion.
    umbral_bloque:
        Fraccion minima de trazo en un bloque para que el pixel de la retina se
        encienda.  Por debajo de 0.5 favorece conservar trazos finos.
    devolver_pasos:
        Si es True devuelve tambien un dict con las imagenes intermedias (para
        ilustrar el proceso).

    Returns
    -------
    Vector bipolar (filas*columnas,) y, opcionalmente, el dict de pasos.
    """
    grises = a_grises(imagen)
    limpia = corregir_iluminacion(grises)
    t = umbral_otsu(limpia)
    trazo = limpia < t
    if not trazo.any():
        raise ValueError("no se encontro ningun trazo en la imagen")

    f0, f1, c0, c1 = caja_del_trazo(trazo)
    recorte = trazo[f0:f1, c0:c1].astype(float)

    # relleno centrado hasta la proporcion de la retina, con margen
    alto, ancho = recorte.shape
    if conservar_proporcion:
        proporcion = forma[0] / forma[1]
        alto_obj = max(alto, int(np.ceil(ancho * proporcion)))
        ancho_obj = max(ancho, int(np.ceil(alto_obj / proporcion)))
        alto_obj = int(np.ceil(ancho_obj * proporcion))
    else:
        alto_obj, ancho_obj = alto, ancho
    extra = int(round(margen * max(alto_obj, ancho_obj)))
    lienzo = np.zeros((alto_obj + 2 * extra, ancho_obj + 2 * extra))
    df = (lienzo.shape[0] - alto) // 2
    dc = (lienzo.shape[1] - ancho) // 2
    lienzo[df:df + alto, dc:dc + ancho] = recorte

    # reduccion por promedio de bloques (area) con Pillow
    reducida = np.asarray(
        Image.fromarray((lienzo * 255).astype(np.uint8)).resize((forma[1], forma[0]), Image.Resampling.BOX),
        dtype=float) / 255.0
    patron = np.where(reducida >= umbral_bloque, 1.0, -1.0).ravel()

    if devolver_pasos:
        return patron, {"grises": grises, "limpia": limpia, "umbral": t, "trazo": trazo, "recorte": lienzo,
                        "reducida": reducida}
    return patron


def cargar_carpeta(carpeta, forma=(7, 6), **kwargs):
    """Lee una carpeta organizada por clases y devuelve (X, etiquetas, rutas).

    Estructura esperada (una subcarpeta por clase, cualquier numero de imagenes)::

        carpeta/
            A/ foto1.jpg  foto2.png ...
            B/ ...

    Si la carpeta contiene imagenes sueltas, la clase es la primera letra del
    nombre del archivo (``A_movil.jpg`` -> clase ``A``).
    """
    carpeta = Path(carpeta)
    X, etiquetas, rutas = [], [], []
    for ruta in sorted(carpeta.rglob("*")):
        if ruta.suffix.lower() not in EXTENSIONES:
            continue
        clase = ruta.parent.name if ruta.parent != carpeta else ruta.stem[0].upper()
        X.append(preprocesar(ruta, forma=forma, **kwargs))
        etiquetas.append(clase)
        rutas.append(ruta)
    if not X:
        raise FileNotFoundError(f"no hay imagenes en {carpeta}")
    return np.array(X), etiquetas, rutas


# ---------------------------------------------------------------------------
# Generacion de imagenes de letras con tipografias reales
# ---------------------------------------------------------------------------

def renderizar_letra(letra: str, fuente: str = "sans_negrita", lado: int = 160,
                     tamano: float = 0.78) -> Image.Image:
    """Letra negra sobre fondo blanco, centrada, en una imagen de `lado` pixeles."""
    ruta = DIR_FUENTES / FUENTES.get(fuente, fuente)
    tipo = ImageFont.truetype(str(ruta), int(lado * tamano))
    imagen = Image.new("L", (lado, lado), 255)
    dibujo = ImageDraw.Draw(imagen)
    caja = dibujo.textbbox((0, 0), letra, font=tipo)
    x = (lado - (caja[2] - caja[0])) / 2 - caja[0]
    y = (lado - (caja[3] - caja[1])) / 2 - caja[1]
    dibujo.text((x, y), letra, fill=0, font=tipo)
    return imagen


def distorsionar(imagen: Image.Image, rng: np.random.Generator, intensidad: float = 1.0) -> Image.Image:
    """Distorsiones de 'foto' aleatorias: geometria, desenfoque, luz y ruido."""
    lado = imagen.size[0]
    angulo = rng.uniform(-10, 10) * intensidad
    escala = max(0.4, 1.0 + rng.uniform(-0.25, 0.1) * intensidad)
    dx, dy = rng.uniform(-0.12, 0.12, size=2) * lado * intensidad
    img = imagen.rotate(angulo, resample=Image.Resampling.BILINEAR, fillcolor=255)
    nuevo = max(8, int(lado * escala))
    img = img.resize((nuevo, nuevo), Image.Resampling.BILINEAR)
    lienzo = Image.new("L", (lado, lado), 255)
    lienzo.paste(img, (int((lado - nuevo) / 2 + dx), int((lado - nuevo) / 2 + dy)))
    img = lienzo.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0, 2.5) * intensidad))

    a = np.asarray(img, dtype=float) / 255.0
    # bajo contraste + gradiente de iluminacion + ruido de sensor
    # el contraste nunca baja de 0.2: con intensidades altas la imagen no se invierte
    contraste = max(0.2, 1.0 - rng.uniform(0.0, 0.45) * intensidad)
    a = 0.5 + (a - 0.5) * contraste
    gx, gy = np.meshgrid(np.linspace(-1, 1, lado), np.linspace(-1, 1, lado))
    direccion = rng.uniform(0, 2 * np.pi)
    a = a + 0.15 * intensidad * (np.cos(direccion) * gx + np.sin(direccion) * gy)
    a = a + rng.normal(0, 0.06 * intensidad, size=a.shape)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
