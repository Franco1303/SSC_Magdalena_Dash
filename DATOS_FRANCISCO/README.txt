========================================================================
README - DOCUMENTACIÓN DE DATOS DEL RÍO MAGDALENA JUNIO-DICIEMBRE 2025
========================================================================

DESCRIPCIÓN GENERAL
-------------------
Este archivo describe la organización de las carpetas y el contenido de los archivos CSV almacenados en este directorio. 

ORIGEN DE LOS DATOS Y ATRIBUCIÓN
--------------------------------
Estos datos fueron recolectados en el marco de las pasantías investigativas realizadas en la Oficina Hidrográfica de Barranquilla de la Dirección General Marítima (DIMAR), y utilizados 
como parte integral de la tesis de grado titulada:

"Variabilidad espacio-temporal de la concentración y distribución de tamaño de partícula de sedimentos en suspensión en el tramo estuarino 
del Río Magdalena: Influencia de forzantes hidro-oceanográficos"

Autor: María Gándara Barboza
Institución: Universidad del Norte

USO DE DATOS Y LICENCIA
-----------------------
Este dataset se comparte con fines académicos y de investigación. El uso de estos datos está permitido bajo la condición de otorgar el crédito apropiado al autor y a la institución mencionada.

Gándara Barboza, María Ángel (Universidad del Norte Geología Departamento de física y geociencias Barranquilla, Colombia, 2025)

ESTRUCTURA DE DIRECTORIO
-------------------------
El directorio principal contiene una subcarpeta mayor. Dentro de esta, la información se organiza cronológicamente por meses (de Junio a Diciembre). Es importante recalcar que las mediciones se realizaron en ascenso a pleamar, aproximadamente entre las 7am-1pm.

.
+-- SEDIMENTOS EN SUSPENSIÓN
|   +-- JUNIO
|   |   +-- 11062025.csv
|   |   +-- ...
|   +-- JULIO
|   +-- AGOSTO
|   +-- SEPTIEMBRE
|   +-- OCTUBRE
|   +-- NOVIEMBRE
|   +-- DICIEMBRE


CONVENCIÓN DE NOMBRES DE ARCHIVO
--------------------------------
Los archivos siguen el formato de fecha: DDMMAAAA.csv
Ejemplo: "11062025.csv" corresponde a los datos del 11 de Junio de 2025.

========================================================================
DESCRIPCIÓN DE VARIABLES (COLUMNAS)
========================================================================

- SEDIMENTOS EN SUSPENSIÓN
---------------------------
Ubicación: Carpeta "SEDIMENTOS EN SUSPENSIÓN"
Formato de columnas (en orden):

- K: Identificador del Km desde la desembocadura como punto de referencia el Km0 (11.10634,-74.8521)
- +m: (Variable auxiliar/distancia)
- Depth (m): Profundidad en metros
- Concentracion volumetrica total (ppm): Concentración en partes por millón

Ejemplo de encabezado en el archivo:
K,'+m,Depth (m),Concentracion volumetrica total (ppm)

========================================================================
NOTAS TÉCNICAS
========================================================================
- Formato de archivo: CSV (Valores Separados por Comas).
- Codificación recomendada: UTF-8.