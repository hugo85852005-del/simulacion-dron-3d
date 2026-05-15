from SimulacionDron.simulacion import Vector3D, Obstaculo, Entorno, Dron, Simulacion
from SimulacionDron.visualizacion import Visualizacion


def crear_entorno():
    entorno = Entorno(limites=(120, 120, 60))
    entorno.definir_viento(0.4, 0.2, 0)

    # ── Zona central (cerca del origen) ──────────────────────────────────────────
    entorno.agregar_obstaculo(Obstaculo(Vector3D(10, 10, 5),  radio=4,  nombre="TORRE-1"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(18, 6,  7),  radio=3,  nombre="OBS-A"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(8,  20, 4),  radio=5,  nombre="EDIF-1"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(25, 15, 6),  radio=4,  nombre="TORRE-2"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(22, 25, 8),  radio=6,  nombre="EDIF-2"))

    # ── Zona media ────────────────────────────────────────────────────────────────
    entorno.agregar_obstaculo(Obstaculo(Vector3D(30, 30, 7),  radio=5,  nombre="TORRE-3"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(15, 35, 4),  radio=4,  nombre="OBS-C"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(40, 20, 6),  radio=3,  nombre="OBS-D"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(5,  30, 5),  radio=4,  nombre="EDIF-3"))

    # ── Alrededores (más lejos) ───────────────────────────────────────────────────
    entorno.agregar_obstaculo(Obstaculo(Vector3D(50, 5,  6),  radio=5,  nombre="TORRE-4"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(55, 25, 8),  radio=4,  nombre="EDIF-4"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(48, 40, 5),  radio=3,  nombre="OBS-E"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(60, 15, 7),  radio=6,  nombre="TORRE-5"))

    entorno.agregar_obstaculo(Obstaculo(Vector3D(-8, 15, 5),  radio=4,  nombre="OBS-G"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(-5, 30, 6),  radio=5,  nombre="EDIF-5"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(5,  -8, 4),  radio=3,  nombre="OBS-H"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(20, -10, 7), radio=4,  nombre="TORRE-6"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(38, -5, 5),  radio=5,  nombre="EDIF-6"))

    # ── Zona lejana (destino) ─────────────────────────────────────────────────────
    entorno.agregar_obstaculo(Obstaculo(Vector3D(70, 50, 6),  radio=4,  nombre="TORRE-7"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(65, 60, 5),  radio=3,  nombre="OBS-I"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(80, 40, 8),  radio=6,  nombre="EDIF-7"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(75, 65, 4),  radio=4,  nombre="OBS-J"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(85, 55, 5),  radio=3,  nombre="TORRE-8"))

    return entorno


def main():
    print("SIMULACIÓN DE DRON EN 3D")

    entorno = crear_entorno()
    dron = Dron(Vector3D(0, 0, 3), "Dron-Alpha")
    sim = Simulacion(dron, entorno)
    destino = Vector3D(45, 40, 10)
    viz = Visualizacion(sim)
    viz.animar(destino)


if __name__ == "__main__":
    main()
