"""
===================================================
  SIMULACIÓN DE MOVIMIENTO DE UN DRON EN 3D
  Proyecto Universitario - Cálculo Vectorial
===================================================
Autor:     [Tu nombre]
Semestre:  2do semestre
Materia:   Cálculo Vectorial / Programación
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math


# ─────────────────────────────────────────────
#  MÓDULO 1: Vector3D
#  Representa un vector en el espacio 3D y
#  sus operaciones matemáticas fundamentales
# ─────────────────────────────────────────────
class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # --- Operaciones vectoriales ---

    def __add__(self, otro):
        """Suma vectorial: v1 + v2"""
        return Vector3D(self.x + otro.x, self.y + otro.y, self.z + otro.z)

    def __sub__(self, otro):
        """Resta vectorial: v1 - v2"""
        return Vector3D(self.x - otro.x, self.y - otro.y, self.z - otro.z)

    def __mul__(self, escalar):
        """Multiplicación por escalar: v * k"""
        return Vector3D(self.x * escalar, self.y * escalar, self.z * escalar)

    def __rmul__(self, escalar):
        return self.__mul__(escalar)

    def magnitud(self):
        """||v|| = √(x² + y² + z²)"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalizar(self):
        """Vector unitario: v̂ = v / ||v||"""
        mag = self.magnitud()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)

    def punto(self, otro):
        """Producto punto: v1 · v2 = x1x2 + y1y2 + z1z2"""
        return self.x*otro.x + self.y*otro.y + self.z*otro.z

    def cruz(self, otro):
        """Producto cruzado: v1 × v2"""
        return Vector3D(
            self.y*otro.z - self.z*otro.y,
            self.z*otro.x - self.x*otro.z,
            self.x*otro.y - self.y*otro.x
        )

    def distancia(self, otro):
        """Distancia entre dos puntos en el espacio"""
        return (self - otro).magnitud()

    def a_array(self):
        """Convierte a array de NumPy para graficación"""
        return np.array([self.x, self.y, self.z])

    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

    def __repr__(self):
        return f"Vector3D{self}"


# ─────────────────────────────────────────────
#  MÓDULO 2: Obstáculo
#  Esfera en el espacio 3D que el dron evita
# ─────────────────────────────────────────────
class Obstaculo:
    def __init__(self, posicion: Vector3D, radio: float, nombre: str = ""):
        self.posicion = posicion
        self.radio = radio
        self.nombre = nombre

    def hay_colision(self, punto: Vector3D, margen: float = 1.0) -> bool:
        """Detecta si un punto está dentro del obstáculo (con margen de seguridad)"""
        return self.posicion.distancia(punto) < (self.radio + margen)

    def fuerza_repulsion(self, punto: Vector3D) -> Vector3D:
        """
        Campo vectorial de repulsión: empuja al dron LEJOS del obstáculo.
        La fuerza es inversamente proporcional al cuadrado de la distancia.
        """
        d = self.posicion.distancia(punto)
        if d < 0.001:
            return Vector3D(0, 0, 1)  # Empuje vertical si está justo encima
        
        zona_influencia = self.radio * 4
        if d > zona_influencia:
            return Vector3D(0, 0, 0)  # Fuera del rango, no hay repulsión
        
        # Dirección de repulsión (apunta DESDE el obstáculo HACIA el dron)
        direccion = punto - self.posicion
        fuerza_magnitud = 2.0 / (d**2)
        return direccion.normalizar() * fuerza_magnitud


# ─────────────────────────────────────────────
#  MÓDULO 3: Entorno
#  Gestiona obstáculos, viento y límites
# ─────────────────────────────────────────────
class Entorno:
    def __init__(self, limites=(50, 50, 30)):
        self.limites = limites      # (x_max, y_max, z_max) en metros
        self.obstaculos = []
        self.viento = Vector3D(0, 0, 0)  # Vector de viento (puede cambiar)

    def agregar_obstaculo(self, obstaculo: Obstaculo):
        self.obstaculos.append(obstaculo)

    def definir_viento(self, vx, vy, vz):
        """El viento es un campo vectorial que afecta la velocidad del dron"""
        self.viento = Vector3D(vx, vy, vz)

    def fuerza_total_obstaculos(self, posicion: Vector3D) -> Vector3D:
        """Suma vectorial de todas las fuerzas de repulsión de obstáculos"""
        fuerza_total = Vector3D(0, 0, 0)
        for obs in self.obstaculos:
            fuerza_total = fuerza_total + obs.fuerza_repulsion(posicion)
        return fuerza_total

    def hay_colision(self, posicion: Vector3D) -> bool:
        return any(obs.hay_colision(posicion) for obs in self.obstaculos)

    def fuera_de_limites(self, posicion: Vector3D) -> bool:
        return (abs(posicion.x) > self.limites[0] or
                abs(posicion.y) > self.limites[1] or
                posicion.z > self.limites[2] or
                posicion.z < 0)


# ─────────────────────────────────────────────
#  MÓDULO 4: Dron
#  Posición, velocidad y aceleración en 3D
# ─────────────────────────────────────────────
class Dron:
    def __init__(self, posicion_inicial: Vector3D, nombre: str = "Dron-1"):
        self.nombre = nombre
        self.posicion = posicion_inicial
        self.velocidad = Vector3D(0, 0, 0)
        self.aceleracion = Vector3D(0, 0, 0)
        
        # Parámetros físicos del dron
        self.velocidad_maxima = 10.0   # m/s
        self.aceleracion_maxima = 5.0  # m/s²
        self.amortiguacion = 0.95      # Factor de fricción del aire (0-1)

    def aplicar_aceleracion(self, a: Vector3D):
        """
        Aplica una aceleración al dron.
        Si supera el límite, se normaliza (dirección conservada, magnitud limitada).
        """
        if a.magnitud() > self.aceleracion_maxima:
            a = a.normalizar() * self.aceleracion_maxima
        self.aceleracion = a

    def actualizar(self, dt: float, entorno: Entorno):
        """
        Integración numérica del movimiento (Método de Euler):
        
        v(t+dt) = v(t) + a(t)·dt          ← derivada de posición
        r(t+dt) = r(t) + v(t)·dt          ← integración
        
        También aplica el efecto del viento y amortiguación del aire.
        """
        # Efecto del viento sobre la velocidad (campo vectorial externo)
        velocidad_efectiva = self.velocidad + entorno.viento * 0.3

        # Actualizar velocidad: v = v + a·Δt
        self.velocidad = velocidad_efectiva + self.aceleracion * dt

        # Limitar velocidad máxima
        if self.velocidad.magnitud() > self.velocidad_maxima:
            self.velocidad = self.velocidad.normalizar() * self.velocidad_maxima

        # Amortiguación del aire (simula fricción)
        self.velocidad = self.velocidad * self.amortiguacion

        # Actualizar posición: r = r + v·Δt
        nueva_posicion = self.posicion + self.velocidad * dt

        # Verificar límites y colisiones
        if not entorno.fuera_de_limites(nueva_posicion) and not entorno.hay_colision(nueva_posicion):
            self.posicion = nueva_posicion
        else:
            # Si hay colisión, invertir velocidad levemente y subir
            self.velocidad = self.velocidad * (-0.5)
            self.velocidad.z = abs(self.velocidad.z) + 0.5

    def info(self):
        print(f"\n{'─'*40}")
        print(f"  🚁 {self.nombre}")
        print(f"  Posición:     {self.posicion}")
        print(f"  Velocidad:    {self.velocidad} (magnitud: {self.velocidad.magnitud():.2f} m/s)")
        print(f"  Aceleración:  {self.aceleracion} (magnitud: {self.aceleracion.magnitud():.2f} m/s²)")
        print(f"{'─'*40}")


# ─────────────────────────────────────────────
#  MÓDULO 5: Simulación
#  Control del movimiento, navegación A→B
# ─────────────────────────────────────────────
class Simulacion:
    def __init__(self, dron: Dron, entorno: Entorno, dt: float = 0.1):
        self.dron = dron
        self.entorno = entorno
        self.dt = dt  # Paso de tiempo en segundos
        
        # Historial para graficar la trayectoria
        self.historial_posiciones = []
        self.historial_velocidades = []
        self.historial_tiempos = []
        self.tiempo_actual = 0.0

    def navegar_a(self, destino: Vector3D, ganancia: float = 2.0) -> bool:
        """
        Navegación automática A → B usando control proporcional.
        
        Concepto: El dron calcula el vector dirección hacia el destino
        y se acelera proporcionalmente a la distancia.
        
        a = k · (destino - posicion)   ← campo vectorial de atracción
        """
        error = destino - self.dron.posicion   # Vector de error (posición actual → destino)
        distancia = error.magnitud()

        if distancia < 1.0:
            return True  # Llegó al destino

        # Aceleración proporcional hacia el destino
        direccion = error.normalizar()
        factor = min(ganancia * distancia / 10, 1.0)  # Normalizar entre 0 y 1
        aceleracion_destino = direccion * (self.dron.aceleracion_maxima * factor)

        # Sumar repulsión de obstáculos
        repulsion = self.entorno.fuerza_total_obstaculos(self.dron.posicion)
        
        # Aceleración total = dirección al destino + evasión de obstáculos
        aceleracion_total = aceleracion_destino + repulsion * 3.0

        self.dron.aplicar_aceleracion(aceleracion_total)
        return False  # Aún no llegó

    def ejecutar_paso(self, destino: Vector3D = None):
        """Ejecuta un paso de simulación y registra el estado"""
        if destino:
            self.navegar_a(destino)

        self.dron.actualizar(self.dt, self.entorno)

        # Registrar estado actual
        self.historial_posiciones.append(self.dron.posicion.a_array().copy())
        self.historial_velocidades.append(self.dron.velocidad.magnitud())
        self.historial_tiempos.append(self.tiempo_actual)
        self.tiempo_actual += self.dt

    def ejecutar(self, destino: Vector3D, tiempo_max: float = 30.0, verbose: bool = True):
        """Ejecuta la simulación completa desde la posición actual hasta el destino"""
        pasos = int(tiempo_max / self.dt)
        llegó = False

        if verbose:
            print(f"\n{'═'*50}")
            print(f"  INICIANDO SIMULACIÓN")
            print(f"  Origen:  {self.dron.posicion}")
            print(f"  Destino: {destino}")
            print(f"  Viento:  {self.entorno.viento}")
            print(f"  Obstáculos: {len(self.entorno.obstaculos)}")
            print(f"{'═'*50}")

        for i in range(pasos):
            llegó = self.navegar_a(destino)
            self.dron.actualizar(self.dt, self.entorno)
            self.historial_posiciones.append(self.dron.posicion.a_array().copy())
            self.historial_velocidades.append(self.dron.velocidad.magnitud())
            self.historial_tiempos.append(self.tiempo_actual)
            self.tiempo_actual += self.dt

            if verbose and i % 20 == 0:
                dist = self.dron.posicion.distancia(destino)
                print(f"  t={self.tiempo_actual:.1f}s | pos={self.dron.posicion} | dist={dist:.1f}m")

            if llegó:
                if verbose:
                    print(f"\n  ✅ ¡DESTINO ALCANZADO en t={self.tiempo_actual:.2f}s!")
                break

        if not llegó and verbose:
            print(f"\n  ⏱ Tiempo máximo alcanzado ({tiempo_max}s)")

        return self.historial_posiciones


# ─────────────────────────────────────────────
#  MÓDULO 6: Visualización
#  Gráficas 3D y animaciones con Matplotlib
# ─────────────────────────────────────────────
class Visualizacion:
    def __init__(self, simulacion: Simulacion):
        self.sim = simulacion
        self.colores_trayectoria = plt.cm.plasma

    def _preparar_datos(self):
        """Convierte el historial en arrays de NumPy para graficar"""
        pos = np.array(self.sim.historial_posiciones)
        return pos[:, 0], pos[:, 1], pos[:, 2]

    def grafica_trayectoria_3d(self, destino: Vector3D = None, titulo: str = "Trayectoria del Dron"):
        """Gráfica 3D completa de la trayectoria con obstáculos y viento"""
        fig = plt.figure(figsize=(14, 9))
        fig.patch.set_facecolor('#0d1117')

        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#0d1117')

        xs, ys, zs = self._preparar_datos()
        n = len(xs)

        # Trayectoria con gradiente de color (tiempo → color)
        for i in range(n - 1):
            t = i / n
            color = self.colores_trayectoria(t)
            ax.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2], color=color, linewidth=2, alpha=0.8)

        # Punto de inicio
        ax.scatter(xs[0], ys[0], zs[0], color='#00ff88', s=120, zorder=5,
                   label='Inicio', depthshade=True)

        # Punto final
        ax.scatter(xs[-1], ys[-1], zs[-1], color='#ff4488', s=120, zorder=5,
                   label='Posición final', depthshade=True)

        # Destino (si se proporciona)
        if destino:
            ax.scatter(destino.x, destino.y, destino.z, color='#ffdd00', s=200,
                       marker='*', zorder=6, label='Destino', depthshade=True)
            ax.text(destino.x, destino.y, destino.z + 1.5, 'DESTINO',
                    color='#ffdd00', fontsize=8, ha='center')

        # Dibujar obstáculos como esferas
        for obs in self.sim.entorno.obstaculos:
            u = np.linspace(0, 2*np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_s = obs.posicion.x + obs.radio * np.outer(np.cos(u), np.sin(v))
            y_s = obs.posicion.y + obs.radio * np.outer(np.sin(u), np.sin(v))
            z_s = obs.posicion.z + obs.radio * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x_s, y_s, z_s, color='#ff4400', alpha=0.35)
            ax.text(obs.posicion.x, obs.posicion.y, obs.posicion.z + obs.radio + 1,
                    obs.nombre or '⚠', color='#ff8800', fontsize=9, ha='center')

        # Flecha de viento
        viento = self.sim.entorno.viento
        if viento.magnitud() > 0:
            cx, cy, cz = np.mean(xs), np.mean(ys), np.max(zs) + 3
            ax.quiver(cx, cy, cz, viento.x * 3, viento.y * 3, viento.z * 3,
                      color='#44aaff', arrow_length_ratio=0.3, linewidth=2)
            ax.text(cx + viento.x * 3, cy + viento.y * 3, cz + viento.z * 3 + 1,
                    f'Viento\n{viento}', color='#44aaff', fontsize=8, ha='center')

        # Proyección de sombra en el suelo (útil para percibir altura)
        ax.plot(xs, ys, np.zeros_like(zs), color='#444466', linewidth=1, alpha=0.4, linestyle='--')

        # Estilo de ejes
        ax.set_xlabel('X (m)', color='#aaaacc', fontsize=10)
        ax.set_ylabel('Y (m)', color='#aaaacc', fontsize=10)
        ax.set_zlabel('Z - Altitud (m)', color='#aaaacc', fontsize=10)
        ax.tick_params(colors='#888899')
        ax.set_title(titulo, color='white', fontsize=13, pad=15)

        # Barra de color para el tiempo
        sm = plt.cm.ScalarMappable(cmap=self.colores_trayectoria,
                                   norm=plt.Normalize(0, self.sim.tiempo_actual))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.1)
        cbar.set_label('Tiempo (s)', color='white', fontsize=9)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        ax.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='white',
                  fontsize=9, loc='upper left')

        plt.tight_layout()
        plt.savefig('trayectoria_3d.png', dpi=150, bbox_inches='tight',
                    facecolor='#0d1117')
        plt.show()
        print("  → Gráfica guardada como 'trayectoria_3d.png'")

    def grafica_analisis(self):
        """Panel de análisis: posición, velocidad y aceleración vs tiempo"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.patch.set_facecolor('#0d1117')
        fig.suptitle('Análisis del Movimiento - Cálculo Vectorial Aplicado',
                     color='white', fontsize=13, y=0.98)

        xs, ys, zs = self._preparar_datos()
        tiempos = self.sim.historial_tiempos
        velocidades = self.sim.historial_velocidades

        colores_xyz = ['#ff6688', '#66ff88', '#6688ff']
        labels_xyz = ['X (m)', 'Y (m)', 'Z - Altitud (m)']

        for ax in axes:
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#888899')
            ax.spines['bottom'].set_color('#333355')
            ax.spines['left'].set_color('#333355')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # --- Gráfica 1: Posición vs tiempo ---
        for comp, color, label in zip([xs, ys, zs], colores_xyz, labels_xyz):
            axes[0].plot(tiempos, comp, color=color, linewidth=1.8, label=label)
        axes[0].set_ylabel('Posición (m)', color='#aaaacc', fontsize=10)
        axes[0].set_title('Vectores de Posición r(t)', color='#ccccee', fontsize=10)
        axes[0].legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor='white', fontsize=9)
        axes[0].grid(True, color='#222244', alpha=0.6)

        # --- Gráfica 2: Velocidad escalar vs tiempo ---
        axes[1].plot(tiempos, velocidades, color='#ffcc44', linewidth=1.8)
        axes[1].fill_between(tiempos, 0, velocidades, color='#ffcc44', alpha=0.15)
        axes[1].axhline(y=self.sim.dron.velocidad_maxima, color='#ff4444',
                        linewidth=1, linestyle='--', label='Velocidad máxima')
        axes[1].set_ylabel('||v(t)|| (m/s)', color='#aaaacc', fontsize=10)
        axes[1].set_title('Magnitud del Vector Velocidad |v(t)|', color='#ccccee', fontsize=10)
        axes[1].legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor='white', fontsize=9)
        axes[1].grid(True, color='#222244', alpha=0.6)

        # --- Gráfica 3: Aceleración (derivada numérica de la velocidad) ---
        if len(velocidades) > 2:
            aceleracion_num = np.gradient(velocidades, self.sim.dt)
            axes[2].plot(tiempos, aceleracion_num, color='#aa66ff', linewidth=1.8)
            axes[2].fill_between(tiempos, 0, aceleracion_num, color='#aa66ff', alpha=0.12)
        axes[2].set_xlabel('Tiempo (s)', color='#aaaacc', fontsize=10)
        axes[2].set_ylabel('dv/dt (m/s²)', color='#aaaacc', fontsize=10)
        axes[2].set_title('Aceleración Escalar a(t) = d|v|/dt', color='#ccccee', fontsize=10)
        axes[2].grid(True, color='#222244', alpha=0.6)

        plt.tight_layout()
        plt.savefig('analisis_vectorial.png', dpi=150, bbox_inches='tight',
                    facecolor='#0d1117')
        plt.show()
        print("  → Gráfica guardada como 'analisis_vectorial.png'")

    def animacion_3d(self, destino: Vector3D = None, intervalo: int = 30):
        """Animación en tiempo real del dron moviéndose en 3D"""
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('#0d1117')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#0d1117')

        xs, ys, zs = self._preparar_datos()
        n = len(xs)

        # Configurar límites de la gráfica
        margen = 5
        ax.set_xlim(xs.min() - margen, xs.max() + margen)
        ax.set_ylim(ys.min() - margen, ys.max() + margen)
        ax.set_zlim(0, zs.max() + margen)

        # Obstáculos (estáticos)
        for obs in self.sim.entorno.obstaculos:
            u = np.linspace(0, 2*np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            x_s = obs.posicion.x + obs.radio * np.outer(np.cos(u), np.sin(v))
            y_s = obs.posicion.y + obs.radio * np.outer(np.sin(u), np.sin(v))
            z_s = obs.posicion.z + obs.radio * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x_s, y_s, z_s, color='#ff4400', alpha=0.3)

        # Destino
        if destino:
            ax.scatter([destino.x], [destino.y], [destino.z],
                       color='#ffdd00', s=200, marker='*', zorder=6)

        # Elementos que se actualizan en cada frame
        trayectoria_linea, = ax.plot([], [], [], color='#00aaff', linewidth=1.5, alpha=0.6)
        dron_punto, = ax.plot([], [], [], 'o', color='#00ff88', markersize=10, zorder=7)
        sombra_punto, = ax.plot([], [], [], 'o', color='#336644', markersize=6, alpha=0.4)
        titulo_ax = ax.set_title('', color='white', fontsize=11)

        ax.set_xlabel('X (m)', color='#aaaacc')
        ax.set_ylabel('Y (m)', color='#aaaacc')
        ax.set_zlabel('Z (m)', color='#aaaacc')
        ax.tick_params(colors='#888899')

        def actualizar(frame):
            idx = min(frame * 3, n - 1)  # Saltar frames para mayor velocidad
            start = max(0, idx - 40)     # Mostrar solo los últimos 40 puntos

            trayectoria_linea.set_data(xs[start:idx], ys[start:idx])
            trayectoria_linea.set_3d_properties(zs[start:idx])

            dron_punto.set_data([xs[idx]], [ys[idx]])
            dron_punto.set_3d_properties([zs[idx]])

            sombra_punto.set_data([xs[idx]], [ys[idx]])
            sombra_punto.set_3d_properties([0])

            t = self.sim.historial_tiempos[idx] if idx < len(self.sim.historial_tiempos) else 0
            vel = self.sim.historial_velocidades[idx] if idx < len(self.sim.historial_velocidades) else 0
            titulo_ax.set_text(f'🚁 Dron | t={t:.1f}s | vel={vel:.2f}m/s | '
                               f'pos=({xs[idx]:.1f}, {ys[idx]:.1f}, {zs[idx]:.1f})')

            return trayectoria_linea, dron_punto, sombra_punto, titulo_ax

        frames = n // 3 + 1
        anim = FuncAnimation(fig, actualizar, frames=frames, interval=intervalo, blit=False)
        plt.tight_layout()
        plt.show()
        return anim


# ─────────────────────────────────────────────
#  PROGRAMA PRINCIPAL
#  Menú interactivo y escenarios de prueba
# ─────────────────────────────────────────────
def escenario_basico():
    """Escenario 1: Vuelo simple con viento"""
    print("\n🔵 ESCENARIO 1: Vuelo con viento")
    
    entorno = Entorno(limites=(60, 60, 35))
    entorno.definir_viento(0.5, 0.3, 0.1)  # Viento suave

    dron = Dron(Vector3D(0, 0, 2), nombre="Dron-Alpha")
    dron.info()

    sim = Simulacion(dron, entorno, dt=0.1)
    destino = Vector3D(30, 25, 15)
    sim.ejecutar(destino, tiempo_max=40, verbose=True)

    viz = Visualizacion(sim)
    viz.grafica_trayectoria_3d(destino, "Escenario 1: Vuelo con Viento")
    viz.grafica_analisis()


def escenario_obstaculos():
    """Escenario 2: Navegación con obstáculos (evasión automática)"""
    print("\n🔴 ESCENARIO 2: Evasión de obstáculos")

    entorno = Entorno(limites=(70, 70, 40))
    entorno.definir_viento(0.2, -0.1, 0.0)

    # Colocar obstáculos en el camino del dron
    entorno.agregar_obstaculo(Obstaculo(Vector3D(10, 10, 8), radio=4, nombre="OBS-A"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(20, 18, 12), radio=5, nombre="OBS-B"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(30, 12, 6), radio=3, nombre="OBS-C"))

    dron = Dron(Vector3D(0, 0, 3), nombre="Dron-Beta")
    sim = Simulacion(dron, entorno, dt=0.1)
    destino = Vector3D(45, 30, 10)
    sim.ejecutar(destino, tiempo_max=60, verbose=True)

    viz = Visualizacion(sim)
    viz.grafica_trayectoria_3d(destino, "Escenario 2: Evasión Automática de Obstáculos")
    viz.grafica_analisis()


def escenario_demo():
    """Escenario 3: Demo completo con múltiples waypoints"""
    print("\n🟢 ESCENARIO 3: Misión con waypoints")

    entorno = Entorno(limites=(80, 80, 40))
    entorno.definir_viento(1.0, 0.5, 0.0)
    entorno.agregar_obstaculo(Obstaculo(Vector3D(15, 15, 10), radio=5, nombre="Edificio-1"))
    entorno.agregar_obstaculo(Obstaculo(Vector3D(35, 25, 8), radio=4, nombre="Edificio-2"))

    # Múltiples waypoints (puntos de paso)
    waypoints = [
        Vector3D(10, 5, 8),
        Vector3D(20, 20, 15),
        Vector3D(40, 15, 10),
        Vector3D(50, 40, 12),
    ]

    dron = Dron(Vector3D(0, 0, 2), nombre="Dron-Gamma")
    sim = Simulacion(dron, entorno, dt=0.1)

    print(f"\n  Recorriendo {len(waypoints)} waypoints...")
    for i, wp in enumerate(waypoints):
        print(f"\n  📍 Waypoint {i+1}/{len(waypoints)}: {wp}")
        sim.ejecutar(wp, tiempo_max=25, verbose=False)
        print(f"     Llegado en t={sim.tiempo_actual:.1f}s | pos={sim.dron.posicion}")

    viz = Visualizacion(sim)
    viz.grafica_trayectoria_3d(waypoints[-1], "Escenario 3: Misión con Múltiples Waypoints")
    viz.grafica_analisis()
    # viz.animacion_3d(waypoints[-1])  # Descomentar para ver animación


def menu_principal():
    """Menú interactivo del programa"""
    print("""
╔══════════════════════════════════════════════════════╗
║      SIMULACIÓN DE DRON EN 3D - CÁLCULO VECTORIAL   ║
║             Proyecto Universitario                   ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  [1] Escenario básico: vuelo con viento              ║
║  [2] Navegación con evasión de obstáculos            ║
║  [3] Misión completa con múltiples waypoints         ║
║  [4] Personalizado: ingresa tus propios parámetros   ║
║  [0] Salir                                           ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
    """)

    opcion = input("  Selecciona un escenario [0-4]: ").strip()

    if opcion == '1':
        escenario_basico()
    elif opcion == '2':
        escenario_obstaculos()
    elif opcion == '3':
        escenario_demo()
    elif opcion == '4':
        escenario_personalizado()
    elif opcion == '0':
        print("\n  ¡Hasta luego! 👋\n")
    else:
        print("\n  Opción inválida. Ejecutando demo completo...\n")
        escenario_demo()


def escenario_personalizado():
    """Permite al usuario ingresar parámetros manualmente"""
    print("\n⚙️  ESCENARIO PERSONALIZADO\n")

    print("  Posición inicial del dron:")
    px = float(input("    x: ") or "0")
    py = float(input("    y: ") or "0")
    pz = float(input("    z (altitud): ") or "2")

    print("\n  Posición del destino:")
    dx = float(input("    x: ") or "30")
    dy = float(input("    y: ") or "20")
    dz = float(input("    z (altitud): ") or "10")

    print("\n  Viento (vector en m/s, puede ser 0):")
    vx = float(input("    viento_x: ") or "0")
    vy = float(input("    viento_y: ") or "0")
    vz = float(input("    viento_z: ") or "0")

    n_obs = int(input("\n  ¿Cuántos obstáculos agregar? [0-5]: ") or "0")
    
    entorno = Entorno()
    entorno.definir_viento(vx, vy, vz)

    for i in range(n_obs):
        print(f"\n  Obstáculo {i+1}:")
        ox = float(input("    x: ") or "10")
        oy = float(input("    y: ") or "10")
        oz = float(input("    z: ") or "5")
        radio = float(input("    radio (metros): ") or "3")
        entorno.agregar_obstaculo(Obstaculo(Vector3D(ox, oy, oz), radio, f"OBS-{i+1}"))

    dron = Dron(Vector3D(px, py, pz))
    sim = Simulacion(dron, entorno, dt=0.1)
    destino = Vector3D(dx, dy, dz)
    sim.ejecutar(destino, tiempo_max=60, verbose=True)

    viz = Visualizacion(sim)
    viz.grafica_trayectoria_3d(destino, "Simulación Personalizada")
    viz.grafica_analisis()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    menu_principal()
