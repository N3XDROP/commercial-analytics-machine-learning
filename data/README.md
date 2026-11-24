# Online Retail II

- Fuente original: Daqing Chen (2012), Online Retail II [conjunto de datos], UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Versión CSV consultada: Mashlyn, Online Retail II UCI, Kaggle. https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
- Nombre esperado: `online_retail_II.csv`
- Registros: 1.067.371 (sin contar el encabezado).
- Tamaño del archivo facilitado: 94.850.204 bytes.
- SHA-256: `32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9`

El conjunto original está disponible bajo CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). La copia en formato CSV procede de Kaggle; deben respetarse también las condiciones aplicables a esa versión. El dataset no forma parte de la licencia del código propio. No se modificaron los datos originales para cambiar el nombre del archivo.

Para verificar el archivo en Windows (PowerShell):

```powershell
Get-FileHash .\data\online_retail_II.csv -Algorithm SHA256
```

Si el CSV no se distribuye con el repositorio, descárgalo de la fuente indicada, ubícalo en esta carpeta con el nombre esperado y comprueba su SHA-256 antes de ejecutar `app.py`.