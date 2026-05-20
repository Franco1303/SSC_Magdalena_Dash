\# CSS Magdalena — Análisis Exploratorio de Datos



Dashboard interactivo desarrollado como parte de la tesis de pregrado en Geología

(Universidad del Norte) para el análisis exploratorio de datos de concentración de

sedimentos en suspensión (CSS) en el tramo estuarino del río Magdalena, Barranquilla.



\## Requisitos



\- Python 3.10 o superior

\- Git



\## Instalación y ejecución local



1\. Clona el repositorio:

```

https://github.com/Franco1303/SSC_Magdalena_Dasht

&#x20;  cd css-magdalena-eda

```



2\. Crea un entorno virtual (recomendado):

```

&#x20;  python -m venv venv

&#x20;  venv\\Scripts\\activate        # Windows

&#x20;  source venv/bin/activate     # Mac/Linux

```



3\. Instala las dependencias:

En su ambiente virtual corra el siguiente comando para instalar todos los paquetes requeridos.

```

&#x20;  pip install -r requirements.txt

```



4\. Corre la app:

En este paso ya deberia poder correr la app a traves del script app.py

```

&#x20;  python app.py

```



5\. Abre tu navegador en: http://127.0.0.1:8050



\## Estructura del proyecto



\- `app.py` — aplicación principal Dash

\- `puntos_alternativos.csv` — dataset final de calibración (reflectancia + CSS)

\- `Q\_MEDIA\_D@29037020.data` — serie de caudal diario estación Calamar

\- `TR\_KT\_D\_QS\_D@29037020.data` — serie de TSS diario estación Calamar

\- `caudal\_ganara.xlsx` — caudal medido en Barranquilla (Jun 2025 – Mar 2026)

\- `DATOS\_FRANCISCO/` — perfiles LISST organizados por mes y fecha



\## Datos



Los datos de campo fueron recolectados entre junio 2025 y marzo 2026 en el tramo

estuarino del río Magdalena utilizando un perfilador LISST. Las imágenes Sentinel-2

fueron procesadas mediante Google Earth Engine.

```


