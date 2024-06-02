# Filter Group Filter Banks
etiquetas = ["Promedio","DesviacionEstandar","Media","Minimo", "Maximo"]
encabezado = []
for i in range(12):
    for j in range(5):
        encabezado.append(etiquetas[j] + str(i+1))

print(len(encabezado))