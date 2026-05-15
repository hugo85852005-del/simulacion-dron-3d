"""
╔══════════════════════════════════════════════════════════════╗
║   SIMULACIÓN DE DRON 3D — VERSIÓN PROFESIONAL               ║
║   Cálculo Vectorial Aplicado — Proyecto Universitario        ║
╠══════════════════════════════════════════════════════════════╣
║  CONCEPTOS MATEMÁTICOS IMPLEMENTADOS:                        ║
║  • Campo vectorial F(x,y,z) = (P,Q,R)                        ║
║  • Gradiente ∇f — dirección de máximo crecimiento            ║
║  • Divergencia ∇·F — expansión/compresión del flujo          ║
║  • Rotacional ∇×F — tendencia de giro del campo              ║
║  • Integral de línea ∫F·dr — trabajo sobre la trayectoria    ║
║  • Campos repulsivos y atractivos vectoriales                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import pygame
import pygame.mixer
import random
import math
import numpy as np

from SimulacionDron.simulacion import Vector3D, Dron, Obstaculo, Entorno, Simulacion


# ══════════════════════════════════════════════════════════════════════════════
#  CÁLCULO VECTORIAL — FUNCIONES MATEMÁTICAS
#  Estas funciones implementan los conceptos del proyecto universitario
# ══════════════════════════════════════════════════════════════════════════════

def campo_vectorial(x, y, z, t=0):
    """
    Campo vectorial F(x,y,z) = (P, Q, R)
    Combina remolino + viento dinámico + turbulencia.
    
    Matemáticamente: F genera fuerzas sobre el dron según su posición.
    """
    # Componente de remolino (rotación en XY)
    # P = -y·k,  Q = x·k  → crea rotación alrededor del eje Z
    k_remolino = 0.08
    P = -y * k_remolino + math.sin(z * 0.1 + t * 0.5) * 0.3
    Q =  x * k_remolino + math.cos(z * 0.1 + t * 0.5) * 0.3
    # Componente vertical (turbulencia)
    R = math.sin(x * 0.15 + t) * math.cos(y * 0.15) * 0.2
    return Vector3D(P, Q, R)


def gradiente(f, x, y, z, h=0.1):
    """
    Gradiente numérico: ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)
    
    Usando diferencias finitas centradas:
    ∂f/∂x ≈ [f(x+h) - f(x-h)] / (2h)
    
    El gradiente apunta en la dirección de MÁXIMO crecimiento de f.
    Físicamente: indica hacia dónde la función crece más rápido.
    """
    gx = (f(x+h, y, z) - f(x-h, y, z)) / (2*h)
    gy = (f(x, y+h, z) - f(x, y-h, z)) / (2*h)
    gz = (f(x, y, z+h) - f(x, y, z-h)) / (2*h)
    return Vector3D(gx, gy, gz)


def divergencia(x, y, z, h=0.5):
    """
    Divergencia numérica: ∇·F = ∂P/∂x + ∂Q/∂y + ∂R/∂z
    
    Usando diferencias finitas:
    ∂P/∂x ≈ [P(x+h) - P(x-h)] / (2h)
    
    > 0: zona de expansión (el fluido "sale")
    < 0: zona de compresión (el fluido "entra")
    = 0: flujo incompresible (conservativo)
    """
    t = 0
    dPdx = (campo_vectorial(x+h,y,z,t).x - campo_vectorial(x-h,y,z,t).x)/(2*h)
    dQdy = (campo_vectorial(x,y+h,z,t).y - campo_vectorial(x,y-h,z,t).y)/(2*h)
    dRdz = (campo_vectorial(x,y,z+h,t).z - campo_vectorial(x,y,z-h,t).z)/(2*h)
    return dPdx + dQdy + dRdz


def rotacional(x, y, z, h=0.5):
    """
    Rotacional numérico: ∇×F = (∂R/∂y-∂Q/∂z, ∂P/∂z-∂R/∂x, ∂Q/∂x-∂P/∂y)
    
    Mide la tendencia de ROTACIÓN del campo vectorial.
    Si |∇×F| > 0: hay remolino o giro en esa región.
    Si ∇×F = 0: campo irrotacional (conservativo, como la gravedad).
    """
    t = 0
    dRdy = (campo_vectorial(x,y+h,z,t).z - campo_vectorial(x,y-h,z,t).z)/(2*h)
    dQdz = (campo_vectorial(x,y,z+h,t).y - campo_vectorial(x,y,z-h,t).y)/(2*h)
    dPdz = (campo_vectorial(x,y,z+h,t).x - campo_vectorial(x,y,z-h,t).x)/(2*h)
    dRdx = (campo_vectorial(x+h,y,z,t).z - campo_vectorial(x-h,y,z,t).z)/(2*h)
    dQdx = (campo_vectorial(x+h,y,z,t).y - campo_vectorial(x-h,y,z,t).y)/(2*h)
    dPdy = (campo_vectorial(x,y+h,z,t).x - campo_vectorial(x,y-h,z,t).x)/(2*h)
    return Vector3D(dRdy-dQdz, dPdz-dRdx, dQdx-dPdy)


def integral_linea(trayectoria, dt=0.1):
    """
    Integral de línea: W = ∫ F·dr ≈ Σ F(rᵢ)·Δrᵢ
    
    Calcula el TRABAJO realizado por el campo vectorial sobre el dron.
    
    Físicamente: si W > 0, el campo ayudó al dron (le dio energía).
                 si W < 0, el campo frenó al dron (le quitó energía).
    
    Unidades: Joules (si las fuerzas están en Newtons y distancias en metros)
    """
    if len(trayectoria) < 2:
        return 0.0
    trabajo = 0.0
    for i in range(1, len(trayectoria)):
        r0 = trayectoria[i-1]
        r1 = trayectoria[i]
        # Vector desplazamiento Δr
        dr = Vector3D(r1[0]-r0[0], r1[1]-r0[1],
                      r1[2] if len(r1)>2 else 0)
        # Campo en el punto medio
        xm = (r0[0]+r1[0])/2; ym = (r0[1]+r1[1])/2
        zm = (r0[2]+r1[2])/2 if len(r0)>2 else 0
        F  = campo_vectorial(xm, ym, zm)
        # Producto punto F·dr
        trabajo += F.x*dr.x + F.y*dr.y + F.z*dr.z
    return trabajo


# ══════════════════════════════════════════════════════════════════════════════
#  SONIDO PROCEDURAL
# ══════════════════════════════════════════════════════════════════════════════
class SonidoFuturista:
    SAMPLE_RATE = 22050
    DURACION    = 0.18

    def __init__(self):
        self.activo = True
        try:
            pygame.mixer.init(frequency=self.SAMPLE_RATE, size=-16,
                              channels=1, buffer=512)
            self._motor   = self._gen("motor")
            self._alerta  = self._gen("alerta")
            self._beep    = self._gen("beep")
            self._llegada = self._gen("llegada")
            self._disparo = self._gen("disparo")
        except Exception:
            self.activo = False

    def _gen(self, tipo):
        n = int(self.SAMPLE_RATE * self.DURACION)
        t = np.linspace(0, self.DURACION, n, False)
        if tipo == "motor":
            v = 1 + 0.02*np.sin(2*np.pi*6*t)
            w = 0.5*np.sin(2*np.pi*120*t*v)+0.3*np.sin(2*np.pi*240*t*v)
            e = np.ones(n)
        elif tipo == "alerta":
            w = 0.6*np.sin(2*np.pi*880*t)+0.4*np.sin(2*np.pi*660*t)
            e = np.exp(-8*t/self.DURACION)
        elif tipo == "beep":
            w = np.sin(2*np.pi*1200*t)
            e = np.exp(-12*t/self.DURACION)
        elif tipo == "disparo":
            w = np.sin(2*np.pi*400*t*np.exp(-t/self.DURACION*3))
            e = np.exp(-15*t/self.DURACION)
        else:
            w = (np.sin(2*np.pi*523*t)+np.sin(2*np.pi*659*t)+np.sin(2*np.pi*784*t))/3
            e = np.sin(np.pi*t/self.DURACION)
        data = (w*e*26000).astype(np.int16)
        return pygame.sndarray.make_sound(data)

    def motor(self):
        if self.activo: self._motor.play()
    def alerta(self):
        if self.activo: self._alerta.play()
    def beep_radar(self):
        if self.activo: self._beep.set_volume(0.3); self._beep.play()
    def llegada(self):
        if self.activo: self._llegada.play()
    def disparo(self):
        if self.activo: self._disparo.play()


# ══════════════════════════════════════════════════════════════════════════════
#  PARALLAX DE ESTRELLAS
# ══════════════════════════════════════════════════════════════════════════════
class CapaEstrellas:
    def __init__(self, n, vp, radio, alpha, W, H):
        self.vp=vp; self.radio=radio; self.alpha=alpha; self.W=W; self.H=H
        self.pts = [[random.uniform(0,W), random.uniform(0,H)] for _ in range(n)]

    def dibujar(self, pantalla, cdx, cdy):
        surf = pygame.Surface((self.W,self.H), pygame.SRCALPHA)
        for p in self.pts:
            vx=(p[0]-cdx*self.vp)%self.W
            vy=(p[1]-cdy*self.vp)%self.H
            pygame.draw.circle(surf,(255,255,255,self.alpha),(int(vx),int(vy)),self.radio)
        pantalla.blit(surf,(0,0))


# ══════════════════════════════════════════════════════════════════════════════
#  PARTÍCULA
# ══════════════════════════════════════════════════════════════════════════════
class Particula:
    def __init__(self, x, y, vx, vy, vida, color, radio=None):
        self.x=x; self.y=y; self.vx=vx; self.vy=vy
        self.vida=vida; self.v_max=vida; self.color=color
        self.radio=radio if radio else random.randint(2,5)

    def actualizar(self):
        self.x+=self.vx; self.y+=self.vy
        self.vx*=0.92; self.vy*=0.92
        self.vida-=1; self.radio=max(1,self.radio-0.08)

    def vivo(self): return self.vida>0

    def dibujar(self, pantalla):
        a=int(255*self.vida/self.v_max)
        r,g,b=self.color
        s=pygame.Surface((self.radio*2+2,self.radio*2+2),pygame.SRCALPHA)
        pygame.draw.circle(s,(r,g,b,a),(self.radio+1,self.radio+1),max(1,int(self.radio)))
        pantalla.blit(s,(int(self.x)-self.radio-1,int(self.y)-self.radio-1))


# ══════════════════════════════════════════════════════════════════════════════
#  PROYECTIL
# ══════════════════════════════════════════════════════════════════════════════
class Proyectil:
    """
    Clase Proyectil — física balística simple.
    El dron puede disparar proyectiles que destruyen obstáculos.
    """
    VELOCIDAD = 25.0   # m/s
    VIDA_MAX  = 80     # frames

    def __init__(self, pos: Vector3D, direccion: Vector3D):
        self.pos      = Vector3D(pos.x, pos.y, pos.z)
        self.vel      = direccion.normalizar() * self.VELOCIDAD
        self.vida     = self.VIDA_MAX
        self.activo   = True
        self.trail    = []  # estela visual

    def actualizar(self, dt=0.05):
        if not self.activo: return
        self.trail.append((self.pos.x, self.pos.y))
        if len(self.trail) > 12: self.trail.pop(0)
        self.pos = self.pos + self.vel * dt
        self.vida -= 1
        if self.vida <= 0: self.activo = False

    def dibujar(self, pantalla, w2s_fn, zoom):
        if not self.activo: return
        sx,sy = w2s_fn(self.pos.x, self.pos.y)
        # Estela
        for i,p in enumerate(self.trail):
            if i < 1: continue
            p0 = w2s_fn(self.trail[i-1][0], self.trail[i-1][1])
            p1 = w2s_fn(p[0], p[1])
            alpha = int(200*i/len(self.trail))
            pygame.draw.line(pantalla,(255,alpha,0),p0,p1,2)
        # Núcleo
        pygame.draw.circle(pantalla,(255,255,100),(sx,sy),5)
        pygame.draw.circle(pantalla,(255,200,0),(sx,sy),3)

    def verificar_impacto(self, obstaculos):
        """Retorna el obstáculo golpeado o None."""
        for obs in obstaculos:
            if self.pos.distancia(obs.posicion) < obs.radio + 1.0:
                self.activo = False
                return obs
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  BOTÓN CLICKEABLE
# ══════════════════════════════════════════════════════════════════════════════
class Boton:
    def __init__(self, texto, x, y, w, h, color_base, color_activo=None, toggle=False):
        self.texto=texto
        self.rect=pygame.Rect(x,y,w,h)
        self.color_base=color_base
        self.color_act=color_activo or tuple(min(255,c+60) for c in color_base)
        self.toggle=toggle; self.activo=False; self.hover=False

    def actualizar(self, eventos, pos_mouse):
        self.hover=self.rect.collidepoint(pos_mouse)
        for ev in eventos:
            if ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1 and self.hover:
                if self.toggle: self.activo=not self.activo
                return True
        return False

    def dibujar(self, pantalla, fuente):
        c=self.color_act if (self.activo or self.hover) else self.color_base
        pygame.draw.rect(pantalla,(0,0,0),self.rect.move(3,3),border_radius=10)
        pygame.draw.rect(pantalla,c,self.rect,border_radius=10)
        bc=(180,220,255) if self.hover else (80,110,150)
        pygame.draw.rect(pantalla,bc,self.rect,2,border_radius=10)
        if fuente:
            txt=fuente.render(self.texto,True,(255,255,255))
            pantalla.blit(txt,(self.rect.x+(self.rect.w-txt.get_width())//2,
                               self.rect.y+(self.rect.h-txt.get_height())//2))


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
class Visualizacion:

    # ── Paleta de colores ─────────────────────────────────────────────────────
    C_FONDO    = (4,   6,   15)
    C_GRID_1   = (14,  18,  35)
    C_GRID_2   = (22,  28,  55)
    C_DRON     = (0,   255, 180)
    C_DRON_B   = (0,   180, 130)
    C_HELICE   = (160, 160, 190)
    C_DEST     = (255, 210, 40)
    C_OBS      = (210, 50,  50)
    C_OBS_GL   = (255, 90,  70)
    C_HUD      = (0,   220, 255)
    C_RADAR_ON = (0,   255, 110)
    C_RADAR_OF = (0,   80,  50)
    C_ALERTA   = (255, 70,  70)
    C_CAMPO    = (0,   180, 255)  # color del campo vectorial
    C_GRAD     = (255, 180, 0)    # color del gradiente
    C_ROT      = (180, 0,   255)  # color del rotacional

    ZOOM_MIN=5; ZOOM_MAX=30
    MM_X,MM_Y,MM_W,MM_H = 1090,20,280,280

    def __init__(self, simulacion):
        self.sim=simulacion

        self.zoom=14.0; self.cam_x=0.0; self.cam_y=0.0
        self.vel_input_x=0.0; self.vel_input_y=0.0; self.vel_input_z=0.0

        # Flags visualización
        self.mostrar_flujo        = False
        self.mostrar_campo_vec    = False   # campo vectorial F
        self.mostrar_gradiente    = False   # gradiente ∇f
        self.mostrar_rotacional   = False   # rotacional ∇×F
        self.mostrar_sensores     = True
        self.mostrar_minimapa     = True
        self.mostrar_hud          = True
        self.mostrar_radar        = True
        self.mostrar_trayectoria  = True
        self.modo_ia              = True
        self.pausa                = False

        # Sistemas de partículas y efectos
        self.particulas  = []
        self.explosiones = []
        self.trayectoria = []         # (x, y, z)
        self.trayectoria_3d = []      # para integral de línea
        self.proyectiles = []

        # Vida del dron (sistema de daño)
        self.vida_max    = 100
        self.vida        = 100
        self.escudo      = False      # SHIFT activa escudo temporal

        # Colisión
        self.colision_timer = 0
        self.colision_obs   = ""

        # Velocidad editable
        self.vel_max_actual = 8.0

        # Nubes
        self.nubes=[[random.randint(0,1400),random.randint(0,420),
                     random.randint(55,130)] for _ in range(20)]

        # Parallax estrellas
        self.capas=[
            CapaEstrellas(90,0.015,1,80,1400,900),
            CapaEstrellas(55,0.05,1,155,1400,900),
            CapaEstrellas(30,0.12,2,220,1400,900),
        ]

        self.inclinacion_x=0.0; self.inclinacion_y=0.0
        self.radar_ang=0.0; self.radar_beep=0

        # Datos matemáticos en tiempo real
        self.trabajo_acumulado = 0.0   # integral de línea
        self.div_actual        = 0.0
        self.rot_magnitud      = 0.0
        self.t_campo           = 0.0   # tiempo para animación del campo

        self.sonido=SonidoFuturista()
        self.frame=0

        self.fuente_hud=None; self.fuente_small=None; self.fuente_big=None
        self.botones=[]
        self.ANCHO=1400; self.ALTO=900

    # ── Coordenadas ───────────────────────────────────────────────────────────
    def w2s(self,wx,wy):
        cx=self.ANCHO//2; cy=self.ALTO//2
        return (int(cx+(wx-self.cam_x)*self.zoom),
                int(cy-(wy-self.cam_y)*self.zoom))

    def _cam_lerp(self, dron):
        dx = dron.posicion.x - self.cam_x
        dy = dron.posicion.y - self.cam_y

        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.45:
            self.cam_x = dron.posicion.x
            self.cam_y = dron.posicion.y
            return

        velocidad_cam = min(0.55, 0.08 + dist * 0.04)
        self.cam_x += dx * velocidad_cam
        self.cam_y += dy * velocidad_cam

    # ── Fondo y grid ──────────────────────────────────────────────────────────
    def _fondo(self,pantalla):
        pantalla.fill(self.C_FONDO)

    def _grid(self,pantalla):
        paso=max(18,min(110,int(48*self.zoom/14)))
        ox=int(self.cam_x*self.zoom)%paso
        oy=int(self.cam_y*self.zoom)%paso
        for x in range(-ox,self.ANCHO+paso,paso):
            pygame.draw.line(pantalla,self.C_GRID_1,(x,0),(x,self.ALTO))
        for y in range(-oy,self.ALTO+paso,paso):
            pygame.draw.line(pantalla,self.C_GRID_1,(0,y),(self.ANCHO,y))
        paso5=paso*5
        ox5=int(self.cam_x*self.zoom)%paso5
        oy5=int(self.cam_y*self.zoom)%paso5
        for x in range(-ox5,self.ANCHO+paso5,paso5):
            pygame.draw.line(pantalla,self.C_GRID_2,(x,0),(x,self.ALTO),2)
        for y in range(-oy5,self.ALTO+paso5,paso5):
            pygame.draw.line(pantalla,self.C_GRID_2,(0,y),(self.ANCHO,y),2)

    # ── Nubes mejoradas ───────────────────────────────────────────────────────
    def _nubes(self,pantalla):
        t_g=pygame.time.get_ticks()*0.0003
        for idx,n in enumerate(self.nubes):
            x,y,tam=n
            x=(x-self.cam_x*0.025+t_g*8*(0.5+idx*0.08))%(self.ANCHO+250)-120
            sw=tam*4+60; sh=tam*2+30
            cs=pygame.Surface((sw,sh),pygame.SRCALPHA)
            cxs=sw//2; cys=sh//2
            pulso=int(4*math.sin(t_g*1.5+idx))
            for (ox,oy_,r,al,col) in [
                (0,0,tam+pulso,55,(45,50,75)),
                (-tam//2,tam//5,tam-10,65,(38,44,65)),
                (tam//2,tam//5,tam-15,65,(38,44,65)),
                (tam//4,-tam//4,tam-20,70,(42,48,70)),
                (-tam//4,-tam//5,tam-25,70,(42,48,70)),
                (0,0,tam-5,80,(50,56,82)),
                (-tam//3,-tam//3,max(4,tam//5),40,(80,90,130)),
                (tam//4,-tam//4,max(3,tam//7),30,(90,100,145)),
            ]:
                pygame.draw.circle(cs,(*col,al),(cxs+ox,cys+oy_),max(4,r))
            pantalla.blit(cs,(int(x)-sw//2,int(y)-sh//2))

    # ── Campo vectorial visual ────────────────────────────────────────────────
    def _dibujar_campo_vectorial(self, pantalla, dron):
        """
        Dibuja flechas del campo vectorial F(x,y,z) en la pantalla.
        Cada flecha muestra dirección e intensidad en esa posición del mundo.
        """
        paso_px = 80  # separación en píxeles
        for px in range(0, self.ANCHO, paso_px):
            for py in range(0, self.ALTO, paso_px):
                # Convertir pixel → coordenadas mundo
                cx=self.ANCHO//2; cy=self.ALTO//2
                wx=(px-cx)/self.zoom+self.cam_x
                wy=-(py-cy)/self.zoom+self.cam_y
                wz=dron.posicion.z  # misma altitud que el dron

                F = campo_vectorial(wx, wy, wz, self.t_campo)
                mag = F.magnitud()
                if mag < 0.001: continue

                # Dirección en píxeles (escalar para visualización)
                escala = min(30, mag * 200)
                Fn = F.normalizar()
                ex = px + Fn.x * escala
                ey = py - Fn.y * escala  # Y invertida en pantalla

                # Color según magnitud (azul→cian→verde)
                t_col = min(1.0, mag*3)
                r = int(0 + t_col*0)
                g = int(120 + t_col*135)
                b = int(255 - t_col*75)
                alpha = int(80 + t_col*100)

                surf = pygame.Surface((4,4), pygame.SRCALPHA)
                pygame.draw.line(pantalla, (r,g,b,alpha),
                                 (px,py), (int(ex),int(ey)), 1)
                pygame.draw.circle(pantalla, (r,g,b), (px,py), 2)

    # ── Gradiente visual ──────────────────────────────────────────────────────
    def _dibujar_gradiente(self, pantalla, dron):
        """
        Visualiza ∇f en la posición del dron.
        Muestra la dirección de máximo crecimiento del potencial escalar.
        """
        # Función escalar: potencial = magnitud del campo
        def potencial(x, y, z):
            return campo_vectorial(x, y, z).magnitud()

        pos = dron.posicion
        grad = gradiente(potencial, pos.x, pos.y, pos.z)
        mag_grad = grad.magnitud()
        if mag_grad < 0.001: return

        sx, sy = self.w2s(pos.x, pos.y)
        Gn = grad.normalizar()
        escala = 60
        ex = sx + Gn.x * escala
        ey = sy - Gn.y * escala

        # Flecha dorada del gradiente
        pygame.draw.line(pantalla, self.C_GRAD, (sx,sy), (int(ex),int(ey)), 3)
        # Punta de flecha
        angulo = math.atan2(-(int(ey)-sy), int(ex)-sx)
        for da in [0.4, -0.4]:
            px_ = int(ex) + int(math.cos(angulo+math.pi+da)*10)
            py_ = int(ey) + int(math.sin(angulo+math.pi+da)*10)
            pygame.draw.line(pantalla, self.C_GRAD, (int(ex),int(ey)), (px_,py_), 2)

        if self.fuente_small:
            lbl = self.fuente_small.render(f"∇f={mag_grad:.2f}", True, self.C_GRAD)
            pantalla.blit(lbl, (sx+8, sy-30))

    # ── Rotacional visual ─────────────────────────────────────────────────────
    def _dibujar_rotacional(self, pantalla, dron):
        """
        Visualiza ∇×F en la posición del dron.
        Muestra círculos si hay remolino, línea si el campo es irrotacional.
        """
        pos = dron.posicion
        rot = rotacional(pos.x, pos.y, pos.z)
        mag_rot = rot.magnitud()

        sx, sy = self.w2s(pos.x, pos.y)

        if mag_rot > 0.05:
            # Círculos concéntricos morados = hay remolino
            for r in range(15, int(15+mag_rot*80), 15):
                t_anim = pygame.time.get_ticks()*0.002
                alpha = int(100*(1-r/(15+mag_rot*80))*abs(math.sin(t_anim+r*0.1)))
                surf = pygame.Surface((r*2+4,r*2+4), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*self.C_ROT, alpha), (r+2,r+2), r, 2)
                pantalla.blit(surf, (sx-r-2, sy-r-2))
            if self.fuente_small:
                lbl = self.fuente_small.render(f"∇×F={mag_rot:.2f}", True, self.C_ROT)
                pantalla.blit(lbl, (sx+8, sy-50))

    # ── Trayectoria ───────────────────────────────────────────────────────────
    def _trayectoria_draw(self,pantalla):
        if not self.mostrar_trayectoria or len(self.trayectoria)<2: return
        pts=[self.w2s(p[0],p[1]) for p in self.trayectoria]
        n=len(pts)
        for i in range(1,n):
            t=i/n
            pygame.draw.line(pantalla,(int(0),int(180*t+60*(1-t)),int(255*t+120*(1-t))),
                             pts[i-1],pts[i],2)

    # ── Obstáculos ────────────────────────────────────────────────────────────
    def _obstaculo(self,pantalla,sx,sy,radio_px,nombre,tipo=0):
        if tipo==0:
            sombra=max(4,radio_px//4)
            pygame.draw.circle(pantalla,(60,0,0),(sx+sombra,sy+sombra),radio_px)
            pygame.draw.circle(pantalla,self.C_OBS,(sx,sy),radio_px)
            pygame.draw.circle(pantalla,self.C_OBS_GL,(sx,sy),radio_px,3)
            for r_off in range(radio_px//3,radio_px,max(1,radio_px//3)):
                pygame.draw.circle(pantalla,(180,40,40),(sx,sy),r_off,1)
            br=max(3,radio_px//4)
            pygame.draw.circle(pantalla,(255,160,140),(sx-radio_px//3,sy-radio_px//3),br)
        elif tipo==1:
            w=int(radio_px*1.6); h=int(radio_px*2.2); d=max(5,radio_px//3)
            pygame.draw.polygon(pantalla,(120,40,20),
                [(sx+w//2,sy-h//2),(sx+w//2+d,sy-h//2-d),
                 (sx+w//2+d,sy+h//2-d),(sx+w//2,sy+h//2)])
            pygame.draw.polygon(pantalla,(200,70,40),
                [(sx-w//2,sy-h//2),(sx+w//2,sy-h//2),
                 (sx+w//2+d,sy-h//2-d),(sx-w//2+d,sy-h//2-d)])
            pygame.draw.rect(pantalla,self.C_OBS,(sx-w//2,sy-h//2,w,h))
            pygame.draw.rect(pantalla,self.C_OBS_GL,(sx-w//2,sy-h//2,w,h),2)
            t_win=pygame.time.get_ticks()*0.001
            for wy in range(sy-h//2+8,sy+h//2-8,14):
                for wx in range(sx-w//2+8,sx+w//2-8,14):
                    blink=math.sin(t_win+wx*0.3+wy*0.2)>0.3
                    pygame.draw.rect(pantalla,(255,220,80) if blink else (40,40,80),
                                     (wx,wy,6,8))
        elif tipo==2:
            w=max(8,radio_px//2); h=radio_px*2
            pygame.draw.rect(pantalla,(60,0,60),(sx-w//2+4,sy-h//2+4,w,h))
            pygame.draw.rect(pantalla,(180,50,200),(sx-w//2,sy-h//2,w,h))
            pygame.draw.rect(pantalla,(220,100,255),(sx-w//2,sy-h//2,w,h),2)
            pygame.draw.line(pantalla,(255,180,255),(sx,sy-h//2),(sx,sy-h//2-20),2)
            pygame.draw.circle(pantalla,(255,100,255),(sx,sy-h//2-22),4)
        if self.fuente_small and nombre:
            txt=self.fuente_small.render(nombre,True,(255,160,120))
            pantalla.blit(txt,(sx-txt.get_width()//2,sy-radio_px-18))

    # ── Destino ───────────────────────────────────────────────────────────────
    def _destino(self,pantalla,sx,sy):
        t=pygame.time.get_ticks()*0.003
        for r in range(8,60,10):
            a=int(200*(1-r/65)*abs(math.sin(t+r*0.25)))
            s=pygame.Surface((r*2+4,r*2+4),pygame.SRCALPHA)
            pygame.draw.circle(s,(255,210,40,a),(r+2,r+2),r,2)
            pantalla.blit(s,(sx-r-2,sy-r-2))
        pygame.draw.line(pantalla,self.C_DEST,(sx-10,sy),(sx+10,sy),2)
        pygame.draw.line(pantalla,self.C_DEST,(sx,sy-10),(sx,sy+10),2)
        pygame.draw.circle(pantalla,self.C_DEST,(sx,sy),6)
        pygame.draw.circle(pantalla,(255,255,200),(sx,sy),3)

    # ── Dron ──────────────────────────────────────────────────────────────────
    def _dron(self,pantalla,sx,sy,dron):
        t=pygame.time.get_ticks()*0.022
        vel=dron.velocidad

        self.inclinacion_x+=(vel.x*3.0-self.inclinacion_x)*0.10
        self.inclinacion_y+=(vel.y*3.0-self.inclinacion_y)*0.10
        ix=max(-26,min(26,int(self.inclinacion_x)))
        iy=max(-26,min(26,int(self.inclinacion_y)))
        cx=sx+ix; cy=sy-iy

        # Escudo visual
        if self.escudo:
            t_sh=pygame.time.get_ticks()*0.004
            alpha_sh=int(80+50*math.sin(t_sh))
            ss=pygame.Surface((80,80),pygame.SRCALPHA)
            pygame.draw.circle(ss,(0,180,255,alpha_sh),(40,40),38,3)
            pantalla.blit(ss,(cx-40,cy-40))

        # Sombra
        ss=pygame.Surface((60,20),pygame.SRCALPHA)
        pygame.draw.ellipse(ss,(0,255,180,55),(0,0,60,20))
        pantalla.blit(ss,(sx-30,sy+20))

        arm_len=32
        for ax,ay in [(-1,-1),(1,-1),(-1,1),(1,1)]:
            ex=cx+ax*arm_len; ey=cy+ay*arm_len
            pygame.draw.line(pantalla,(80,80,100),(cx,cy),(ex,ey),6)
            pygame.draw.line(pantalla,self.C_HELICE,(cx,cy),(ex,ey),3)

        for ax,ay in [(-1,-1),(1,-1),(-1,1),(1,1)]:
            hx=cx+ax*arm_len; hy=cy+ay*arm_len
            spd=1+vel.magnitud()*0.15
            off=t*spd*(1+abs(ax+ay)*0.1)
            pygame.draw.circle(pantalla,(60,60,80),(hx,hy),16,3)
            pygame.draw.circle(pantalla,(120,120,150),(hx,hy),14,1)
            for pala in range(2):
                ang=off+pala*math.pi
                bx1=hx+math.cos(ang)*13; by1=hy+math.sin(ang)*4
                bx2=hx+math.cos(ang+math.pi)*13; by2=hy+math.sin(ang+math.pi)*4
                pygame.draw.line(pantalla,self.C_HUD,(int(bx1),int(by1)),(int(bx2),int(by2)),3)
            glow_a=int(80+60*abs(math.sin(off)))
            gs=pygame.Surface((10,10),pygame.SRCALPHA)
            pygame.draw.circle(gs,(0,255,200,glow_a),(5,5),5)
            pantalla.blit(gs,(hx-5,hy-5))

        pygame.draw.circle(pantalla,(10,12,22),(cx+5,cy+5),24)
        # Color del cuerpo según vida
        vida_t = self.vida/self.vida_max
        c_body = (int((1-vida_t)*255), int(vida_t*255), int(vida_t*180))
        pygame.draw.circle(pantalla,c_body,(cx,cy),24)
        pygame.draw.circle(pantalla,self.C_DRON,(cx,cy),20)
        pygame.draw.circle(pantalla,(0,200,160),(cx,cy),14,2)
        pygame.draw.circle(pantalla,(220,255,255),(cx,cy),7)
        pygame.draw.circle(pantalla,self.C_HUD,(cx,cy),5)

        vm=vel.magnitud()
        if vm>0.3 and abs(math.sin(t*(1+vm*0.4)))>0.45:
            pygame.draw.circle(pantalla,(255,60,60),(cx,cy-24),5)

        if self.fuente_small:
            nm=self.fuente_small.render(dron.nombre,True,(0,200,160))
            pantalla.blit(nm,(cx-nm.get_width()//2,cy-38))

    # ── Propulsión ────────────────────────────────────────────────────────────
    def _propulsion(self,sx,sy,dron):
        vel=dron.velocidad
        ix=max(-26,min(26,int(self.inclinacion_x)))
        iy=max(-26,min(26,int(self.inclinacion_y)))
        cx=sx+ix; cy=sy-iy
        for ax,ay in [(-32,-32),(32,-32),(-32,32),(32,32)]:
            hx=cx+ax; hy=cy+ay
            vx=-vel.x*1.5+random.uniform(-2,2)
            vy=vel.y*1.5+random.uniform(-2,2)+1.0
            vida=random.randint(16,32)
            c=random.choice([(0,220,255),(40,160,255),(100,80,255),(0,255,170)])
            self.particulas.append(Particula(hx,hy,vx,vy,vida,c))

    # ── Barra de vida ─────────────────────────────────────────────────────────
    def _barra_vida(self, pantalla, sx, sy):
        bw=50; bh=6
        pygame.draw.rect(pantalla,(60,0,0),(sx-bw//2,sy+32,bw,bh))
        vida_w=int(bw*(self.vida/self.vida_max))
        t=self.vida/self.vida_max
        c=(int((1-t)*255),int(t*200),0)
        pygame.draw.rect(pantalla,c,(sx-bw//2,sy+32,vida_w,bh))
        pygame.draw.rect(pantalla,(100,100,150),(sx-bw//2,sy+32,bw,bh),1)

    # ── Proyectiles ───────────────────────────────────────────────────────────
    def _disparar(self, dron, destino):
        """Crea un proyectil desde el dron hacia el destino."""
        dir_vec = destino - dron.posicion
        if dir_vec.magnitud() > 0.1:
            p = Proyectil(dron.posicion, dir_vec)
            self.proyectiles.append(p)
            self.sonido.disparo()
            # Partículas de disparo
            sx,sy = self.w2s(dron.posicion.x, dron.posicion.y)
            for _ in range(12):
                self.particulas.append(Particula(sx,sy,
                    random.uniform(-3,3),random.uniform(-3,3),
                    random.randint(8,18),(255,200,50)))

    def _actualizar_proyectiles(self, pantalla, destino):
        activos=[]
        for p in self.proyectiles:
            p.actualizar()
            if p.activo:
                obs_golpeado = p.verificar_impacto(self.sim.entorno.obstaculos)
                if obs_golpeado:
                    # ¡Explosión y eliminar obstáculo!
                    sx,sy=self.w2s(p.pos.x,p.pos.y)
                    for _ in range(35):
                        self.explosiones.append({
                            "x":sx,"y":sy,
                            "vx":random.uniform(-6,6),
                            "vy":random.uniform(-6,6),
                            "r":random.randint(4,9),
                            "vida":random.randint(25,50),
                            "color":(255,random.randint(80,200),0)
                        })
                    self.sim.entorno.obstaculos.remove(obs_golpeado)
                    self.sonido.alerta()
                else:
                    p.dibujar(pantalla, self.w2s, self.zoom)
                    activos.append(p)
        self.proyectiles=activos

    # ── Radar ─────────────────────────────────────────────────────────────────
    def _radar(self,pantalla,dron):
        rx,ry=self.ANCHO-180,self.ALTO-200
        r_radio=90; alcance=35.0

        pygame.draw.circle(pantalla,(4,16,8),(rx,ry),r_radio)
        for ani in range(18,r_radio+1,18):
            pygame.draw.circle(pantalla,self.C_RADAR_OF,(rx,ry),ani,1)
        for ang in [0,90,180,270]:
            a=math.radians(ang)
            pygame.draw.line(pantalla,self.C_RADAR_OF,(rx,ry),
                (int(rx+math.cos(a)*r_radio),int(ry+math.sin(a)*r_radio)),1)

        self.radar_ang=(self.radar_ang+2.5)%360
        ar=math.radians(self.radar_ang)
        bx=rx+math.cos(ar)*r_radio; by=ry+math.sin(ar)*r_radio
        pygame.draw.line(pantalla,self.C_RADAR_ON,(rx,ry),(int(bx),int(by)),2)
        for fade in range(1,25):
            af=math.radians(self.radar_ang-fade*3)
            fa=int(140*(1-fade/25))
            fx=rx+math.cos(af)*r_radio; fy=ry+math.sin(af)*r_radio
            pygame.draw.line(pantalla,(0,fa,35),(rx,ry),(int(fx),int(fy)),1)

        for obs in self.sim.entorno.obstaculos:
            rel=obs.posicion-dron.posicion
            d=rel.magnitud()
            if d<alcance:
                ang_obs=math.degrees(math.atan2(-rel.y,rel.x))%360
                nx=int(rx+(rel.x/alcance)*r_radio)
                ny=int(ry-(rel.y/alcance)*r_radio)
                # Color según distancia: verde→amarillo→rojo
                peligro=1-d/alcance
                rc=int(peligro*255); gc=int((1-peligro)*200)
                pygame.draw.circle(pantalla,(rc,gc,0),(nx,ny),6)
                pygame.draw.circle(pantalla,(255,200,0),(nx,ny),3)
                # Distancia en radar
                if self.fuente_small:
                    ds=self.fuente_small.render(f"{d:.0f}m",True,(rc,gc,0))
                    pantalla.blit(ds,(nx+6,ny-6))
                diff=abs((self.radar_ang-ang_obs+360)%360)
                if diff<4 and self.radar_beep<=0:
                    self.sonido.beep_radar(); self.radar_beep=18

        pygame.draw.circle(pantalla,self.C_DRON,(rx,ry),5)
        pygame.draw.circle(pantalla,self.C_RADAR_ON,(rx,ry),r_radio,2)
        if self.fuente_small:
            l=self.fuente_small.render("RADAR",True,self.C_RADAR_ON)
            pantalla.blit(l,(rx-l.get_width()//2,ry+r_radio+5))
        if self.radar_beep>0: self.radar_beep-=1

    # ── HUD mejorado ──────────────────────────────────────────────────────────
    def _hud(self,pantalla,dron,destino,fps):
        if not self.fuente_hud: return
        vel=dron.velocidad.magnitud()
        dist=dron.posicion.distancia(destino)

        alertas=[]
        for obs in self.sim.entorno.obstaculos:
            d=dron.posicion.distancia(obs.posicion)
            if d<obs.radio*3.5:
                alertas.append(f" ALERTA: {obs.nombre} a {d:.0f}m")

        # Calcular datos vectoriales en tiempo real
        F_pos=campo_vectorial(dron.posicion.x,dron.posicion.y,dron.posicion.z,self.t_campo)
        div_val=divergencia(dron.posicion.x,dron.posicion.y,dron.posicion.z)
        rot_val=rotacional(dron.posicion.x,dron.posicion.y,dron.posicion.z)

        lineas=[
            (f"  POS  ({dron.posicion.x:>5.1f}, {dron.posicion.y:>5.1f}, {dron.posicion.z:>5.1f})", self.C_HUD),
            (f"  VEL  {vel:>5.2f} m/s  ALT:{dron.posicion.z:>4.1f}m", self.C_HUD),
            (f"  DIST {dist:>5.1f} m  FPS:{int(fps):>3}", (100,220,255)),
            (f"  VIDA {'█'*int(self.vida//10)}{'░'*(10-int(self.vida//10))} {self.vida:.0f}%",
             (0,255,100) if self.vida>60 else (255,200,0) if self.vida>30 else self.C_ALERTA),
            (f"  IA   {'ACTIVA' if self.modo_ia else 'MANUAL'}",
             self.C_RADAR_ON if self.modo_ia else self.C_ALERTA),
            ("─"*38, (40,60,100)),
            (f"  CAMPO F=({F_pos.x:>4.2f},{F_pos.y:>4.2f},{F_pos.z:>4.2f})", self.C_CAMPO),
            (f"  ∇·F  {div_val:>+6.3f}  {'EXPANDE' if div_val>0 else 'COMPRIME'}", self.C_GRAD),
            (f"  |∇×F| {rot_val.magnitud():>5.3f}  {'REMOLINO' if rot_val.magnitud()>0.1 else 'IRROT.'}", self.C_ROT),
            (f"  ∫F·dr {self.trabajo_acumulado:>+7.2f} J", (200,200,100)),
            ("─"*38, (40,60,100)),
            (f"  OBS  {len(self.sim.entorno.obstaculos)}  VEL MAX:{self.vel_max_actual:.0f}", (200,140,90)),
            (f"  VIENTO {self.sim.entorno.viento}", (80,150,255)),
        ]

        w_hud=390; h_hud=len(lineas)*22+16+len(alertas)*22
        s=pygame.Surface((w_hud,h_hud),pygame.SRCALPHA)
        s.fill((0,8,22,180))
        pygame.draw.rect(s,(0,180,255,70),(0,0,w_hud,h_hud),1,border_radius=10)
        pantalla.blit(s,(10,10))

        y=14
        for txt,col in lineas:
            t=self.fuente_hud.render(txt,True,col)
            pantalla.blit(t,(14,y)); y+=22
        for al in alertas:
            t=self.fuente_hud.render(al,True,self.C_ALERTA)
            pantalla.blit(t,(14,y)); y+=22

        ley=["WASD:mover | QE:altitud | SPACE:disparar | SHIFT:escudo | I:IA | F:campo",
             "C:campo vec | G:gradiente | V:rotacional | T:tray | R:mapa | ESC:salir"]
        for i,l in enumerate(ley):
            t=self.fuente_small.render(l,True,(60,90,130))
            pantalla.blit(t,(14,self.ALTO-30-(len(ley)-i-1)*20))

    # ── Minimapa ──────────────────────────────────────────────────────────────
    def _minimapa(self,pantalla,dron,destino):
        mx,my,mw,mh=self.MM_X,self.MM_Y,self.MM_W,self.MM_H
        escala=0.45
        cx_mm=mx+mw//2
        cy_mm=my+mh//2

        s=pygame.Surface((mw,mh),pygame.SRCALPHA)
        s.fill((8,12,28,210))
        pantalla.blit(s,(mx,my))
        pygame.draw.rect(pantalla,(50,70,110),(mx,my,mw,mh),1,border_radius=12)

        def w2mm(wx, wy):
            dx = wx - dron.posicion.x
            dy = wy - dron.posicion.y
            return (int(cx_mm + dx * escala), int(cy_mm - dy * escala))

        for gx in range(-mw//2,mw//2,22):
            pygame.draw.line(pantalla,(18,28,50),(cx_mm+gx,my),(cx_mm+gx,my+mh))
        for gy in range(-mh//2,mh//2,22):
            pygame.draw.line(pantalla,(18,28,50),(mx,cy_mm+gy),(mx+mw,cy_mm+gy))

        if len(self.trayectoria)>2:
            pts=[w2mm(p[0],p[1]) for p in self.trayectoria[::3]]
            if len(pts)>1:
                pygame.draw.lines(pantalla,(0,70,150),False,pts,1)

        for obs in self.sim.entorno.obstaculos:
            ox,oy=w2mm(obs.posicion.x,obs.posicion.y)
            r_mm=max(3,int(obs.radio*escala))
            pygame.draw.circle(pantalla,self.C_OBS,(ox,oy),r_mm)

        tx,ty=w2mm(destino.x,destino.y)
        pygame.draw.circle(pantalla,self.C_DEST,(tx,ty),7)

        dx,dy=w2mm(dron.posicion.x,dron.posicion.y)
        pygame.draw.circle(pantalla,self.C_DRON,(dx,dy),6)

        # Proyectiles en minimapa
        for p in self.proyectiles:
            px,py=w2mm(p.pos.x,p.pos.y)
            pygame.draw.circle(pantalla,(255,200,0),(px,py),3)

        pygame.draw.rect(pantalla,(70,110,160),(mx,my,mw,mh),2,border_radius=12)
        if self.fuente_small:
            l=self.fuente_small.render("MAPA — click=mover destino",True,(50,80,120))
            pantalla.blit(l,(mx+4,my+mh+5))

    def _mm_clic(self,sx,sy):
        mx,my,mw,mh=self.MM_X,self.MM_Y,self.MM_W,self.MM_H
        cx_mm=mx+mw//2; cy_mm=my+mh//2; escala=0.45
        if mx<=sx<=mx+mw and my<=sy<=my+mh:
            wx = self.sim.dron.posicion.x + (sx-cx_mm)/escala
            wy = self.sim.dron.posicion.y - (sy-cy_mm)/escala
            return wx, wy
        return None
        return None

    # ── Flujo vectorial ───────────────────────────────────────────────────────
    def _flujo(self,pantalla):
        viento=self.sim.entorno.viento
        for i in range(0,self.ANCHO,65):
            for j in range(0,self.ALTO,65):
                ang=math.sin(i*0.012+self.frame*0.01)+math.cos(j*0.012)
                fx=i+math.cos(ang)*16+viento.x*6
                fy=j+math.sin(ang)*16-viento.y*6
                pygame.draw.line(pantalla,(0,180,210),(i,j),(int(fx),int(fy)),1)
                pygame.draw.circle(pantalla,(0,220,255),(i,j),2)

    # ── Colisión ──────────────────────────────────────────────────────────────
    def _verificar_colision(self,dron,sx_d,sy_d):
        for obs in self.sim.entorno.obstaculos:
            d=dron.posicion.distancia(obs.posicion)
            if d<obs.radio+1.2:
                if self.colision_timer<=0:
                    dir_e=dron.posicion-obs.posicion
                    if dir_e.magnitud()>0.01: dir_e=dir_e.normalizar()
                    dron.velocidad=dir_e*min(4.0,dron.velocidad.magnitud()*0.8+1.5)
                    dron.posicion=dron.posicion+dir_e*1.5
                    # Daño al dron
                    if not self.escudo:
                        self.vida=max(0, self.vida-10)
                    for _ in range(25):
                        self.explosiones.append({
                            "x":sx_d,"y":sy_d,
                            "vx":random.uniform(-5,5),"vy":random.uniform(-5,5),
                            "r":random.randint(3,7),"vida":random.randint(18,35),
                            "color":(255,random.randint(60,180),0)
                        })
                    self.sonido.alerta()
                    self.colision_timer=45; self.colision_obs=obs.nombre
                return True
        return False

    def _flash_colision(self,pantalla):
        if self.colision_timer>0:
            self.colision_timer-=1
            alpha=int(160*(self.colision_timer/45))
            sf=pygame.Surface((self.ANCHO,self.ALTO),pygame.SRCALPHA)
            sf.fill((255,0,0,alpha)); pantalla.blit(sf,(0,0))
            if self.fuente_big:
                msg=self.fuente_big.render(
                    f"COLISIÓN: {self.colision_obs}  VIDA:{self.vida:.0f}%",
                    True,(255,80,80))
                pantalla.blit(msg,(self.ANCHO//2-msg.get_width()//2,self.ALTO//2+60))

    # ── IA evasiva ────────────────────────────────────────────────────────────
    def _ia_step(self,dron,destino):
        error=destino-dron.posicion
        dist=error.magnitud()
        if dist<0.8: return
        k_att=2.2; factor=min(1.0,dist/14.0)
        a_dest=error.normalizar()*(dron.aceleracion_maxima*factor*k_att)
        # Incluir campo vectorial como fuerza adicional
        F=campo_vectorial(dron.posicion.x,dron.posicion.y,dron.posicion.z,self.t_campo)
        repul=self.sim.entorno.fuerza_total_obstaculos(dron.posicion)
        a_alt=Vector3D(0,0,3) if dron.posicion.z<2 else Vector3D(0,0,0)
        dron.aplicar_aceleracion(a_dest+repul*3.5+a_alt+F*0.3)

    # ══════════════════════════════════════════════════════════════════════════
    #  BUCLE PRINCIPAL
    # ══════════════════════════════════════════════════════════════════════════
    def animar(self,destino_inicial):
        pygame.init()

        pantalla=pygame.display.set_mode((0,0),pygame.FULLSCREEN)
        self.ANCHO,self.ALTO=pantalla.get_size()
        pygame.display.set_caption("DRON VECTORIAL AVANZADO — Cálculo Vectorial")

        self.fuente_hud   = pygame.font.SysFont("Consolas",16)
        self.fuente_small = pygame.font.SysFont("Consolas",13)
        self.fuente_big   = pygame.font.SysFont("Consolas",28,bold=True)

        # ── Botones ───────────────────────────────────────────────────────────
        btn_y=self.ALTO-55; btn_h=38; btn_w=95; bx=10; gap=6

        def mkbtn(texto, color, color2, toggle=True):
            nonlocal bx
            b=Boton(texto,bx,btn_y,btn_w,btn_h,color,color2,toggle=toggle)
            bx+=btn_w+gap; return b

        btn_ia    = mkbtn("IA ON/OFF",(20,70,25),(30,130,40))
        btn_flujo = mkbtn("FLUJO",    (20,40,80),(30,70,150))
        btn_campo = mkbtn("CAMPO F",  (10,40,90),(20,80,180))
        btn_grad  = mkbtn("∇f GRAD",  (60,40,10),(120,80,20))
        btn_rot   = mkbtn("∇×F ROT",  (50,10,60),(100,20,130))
        btn_tray  = mkbtn("TRAYECT",  (20,40,80),(30,70,150))
        btn_mapa  = mkbtn("MAPA",     (20,40,80),(30,70,150))
        btn_radar = mkbtn("RADAR",    (20,40,80),(30,70,150))
        btn_pausa = mkbtn("PAUSA",    (70,40,10),(140,80,20))
        btn_vel_p = mkbtn("VEL +",    (50,20,80),(100,40,160),toggle=False)
        btn_vel_m = mkbtn("VEL -",    (60,10,40),(130,20,80), toggle=False)

        btn_ia.activo=self.modo_ia; btn_tray.activo=self.mostrar_trayectoria
        btn_mapa.activo=self.mostrar_minimapa; btn_radar.activo=self.mostrar_radar

        self.botones=[btn_ia,btn_flujo,btn_campo,btn_grad,btn_rot,
                      btn_tray,btn_mapa,btn_radar,btn_pausa,btn_vel_p,btn_vel_m]

        reloj=pygame.time.Clock()
        destino=Vector3D(destino_inicial.x,destino_inicial.y,destino_inicial.z)
        dron=self.sim.dron; dt=0.05

        self.cam_x=dron.posicion.x; self.cam_y=dron.posicion.y
        llegado=False; llegado_timer=0
        running=True

        while running:
            fps=reloj.get_fps(); reloj.tick(60); self.frame+=1
            self.t_campo+=0.016  # tiempo para animación del campo

            pos_mouse=pygame.mouse.get_pos()
            eventos=pygame.event.get()

            # ── Actualizar botones ────────────────────────────────────────────
            if btn_ia.actualizar(eventos,pos_mouse):    self.modo_ia=btn_ia.activo
            if btn_flujo.actualizar(eventos,pos_mouse): self.mostrar_flujo=btn_flujo.activo
            if btn_campo.actualizar(eventos,pos_mouse): self.mostrar_campo_vec=btn_campo.activo
            if btn_grad.actualizar(eventos,pos_mouse):  self.mostrar_gradiente=btn_grad.activo
            if btn_rot.actualizar(eventos,pos_mouse):   self.mostrar_rotacional=btn_rot.activo
            if btn_tray.actualizar(eventos,pos_mouse):  self.mostrar_trayectoria=btn_tray.activo
            if btn_mapa.actualizar(eventos,pos_mouse):  self.mostrar_minimapa=btn_mapa.activo
            if btn_radar.actualizar(eventos,pos_mouse): self.mostrar_radar=btn_radar.activo
            if btn_pausa.actualizar(eventos,pos_mouse): self.pausa=btn_pausa.activo
            if btn_vel_p.actualizar(eventos,pos_mouse):
                self.vel_max_actual=min(20.0,self.vel_max_actual+2.0)
                dron.velocidad_maxima=self.vel_max_actual
                dron.aceleracion_maxima=self.vel_max_actual*0.6
            if btn_vel_m.actualizar(eventos,pos_mouse):
                self.vel_max_actual=max(2.0,self.vel_max_actual-2.0)
                dron.velocidad_maxima=self.vel_max_actual
                dron.aceleracion_maxima=self.vel_max_actual*0.6

            # ── Eventos ───────────────────────────────────────────────────────
            teclas_mod=pygame.key.get_mods()
            for ev in eventos:
                if ev.type==pygame.QUIT: running=False
                elif ev.type==pygame.KEYDOWN:
                    if ev.key==pygame.K_ESCAPE: running=False
                    elif ev.key==pygame.K_SPACE:
                        if self.modo_ia:
                            self._disparar(dron,destino)
                        else:
                            self.pausa=not self.pausa; btn_pausa.activo=self.pausa
                    elif ev.key==pygame.K_i:
                        self.modo_ia=not self.modo_ia; btn_ia.activo=self.modo_ia
                    elif ev.key==pygame.K_f:
                        self.mostrar_flujo=not self.mostrar_flujo; btn_flujo.activo=self.mostrar_flujo
                    elif ev.key==pygame.K_c:
                        self.mostrar_campo_vec=not self.mostrar_campo_vec; btn_campo.activo=self.mostrar_campo_vec
                    elif ev.key==pygame.K_g:
                        self.mostrar_gradiente=not self.mostrar_gradiente; btn_grad.activo=self.mostrar_gradiente
                    elif ev.key==pygame.K_v:
                        self.mostrar_rotacional=not self.mostrar_rotacional; btn_rot.activo=self.mostrar_rotacional
                    elif ev.key==pygame.K_t:
                        self.mostrar_trayectoria=not self.mostrar_trayectoria; btn_tray.activo=self.mostrar_trayectoria
                    elif ev.key==pygame.K_r:
                        self.mostrar_minimapa=not self.mostrar_minimapa; btn_mapa.activo=self.mostrar_minimapa
                    elif ev.key==pygame.K_UP:
                        self.zoom=min(self.ZOOM_MAX,self.zoom+1)
                    elif ev.key==pygame.K_DOWN:
                        self.zoom=max(self.ZOOM_MIN,self.zoom-1)
                elif ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1:
                    res=self._mm_clic(*ev.pos)
                    if res: destino.x=res[0]; destino.y=res[1]; llegado=False
                elif ev.type==pygame.MOUSEWHEEL:
                    self.zoom=max(self.ZOOM_MIN,min(self.ZOOM_MAX,self.zoom+ev.y))

            # ── Teclado ───────────────────────────────────────────────────────
            teclas=pygame.key.get_pressed()
            self.escudo=(teclas_mod & pygame.KMOD_SHIFT) != 0

            acel=0.6; damp=0.80
            if not self.modo_ia:
                if teclas[pygame.K_w]:   self.vel_input_y+=acel
                elif teclas[pygame.K_s]: self.vel_input_y-=acel
                else:                    self.vel_input_y*=damp
                if teclas[pygame.K_d]:   self.vel_input_x+=acel
                elif teclas[pygame.K_a]: self.vel_input_x-=acel
                else:                    self.vel_input_x*=damp
                if teclas[pygame.K_q]:   self.vel_input_z+=acel*0.5
                elif teclas[pygame.K_e]: self.vel_input_z-=acel*0.5
                else:                    self.vel_input_z*=damp
                # SHIFT = turbo
                mult=2.0 if self.escudo else 1.0
                lim=6.0*mult
                self.vel_input_x=max(-lim,min(lim,self.vel_input_x))
                self.vel_input_y=max(-lim,min(lim,self.vel_input_y))
                self.vel_input_z=max(-3,min(3,self.vel_input_z))
                dron.posicion.x+=self.vel_input_x*dt
                dron.posicion.y+=self.vel_input_y*dt
                dron.posicion.z+=self.vel_input_z*dt
                dron.posicion.z=max(0.5,dron.posicion.z)
                dron.velocidad=Vector3D(self.vel_input_x,self.vel_input_y,self.vel_input_z)
                # SPACE en manual = disparar
                if teclas[pygame.K_SPACE] and self.frame%15==0:
                    self._disparar(dron,destino)
            else:
                spd=0.35*(2 if self.escudo else 1)
                if teclas[pygame.K_w]: destino.y+=spd
                if teclas[pygame.K_s]: destino.y-=spd
                if teclas[pygame.K_a]: destino.x-=spd
                if teclas[pygame.K_d]: destino.x+=spd
                if teclas[pygame.K_q]: destino.z+=spd*0.5
                if teclas[pygame.K_e]: destino.z-=spd*0.5

            # ── Física ────────────────────────────────────────────────────────
            if not self.pausa:
                if self.modo_ia:
                    self._ia_step(dron,destino)
                    dron.actualizar(dt,self.sim.entorno)
                    if dron.posicion.distancia(destino)<1.2 and not llegado:
                        llegado=True; llegado_timer=130; self.sonido.llegada()
                else:
                    dron.velocidad=dron.velocidad+self.sim.entorno.viento*0.01

                sx_col,sy_col=self.w2s(dron.posicion.x,dron.posicion.y)
                self._verificar_colision(dron,sx_col,sy_col)

                if self.frame%2==0:
                    self.trayectoria.append((dron.posicion.x,dron.posicion.y))
                    self.trayectoria_3d.append((dron.posicion.x,dron.posicion.y,dron.posicion.z))
                    if len(self.trayectoria)>700: self.trayectoria.pop(0)
                    if len(self.trayectoria_3d)>700: self.trayectoria_3d.pop(0)

                # Actualizar integral de línea cada 30 frames
                if self.frame%30==0 and len(self.trayectoria_3d)>5:
                    self.trabajo_acumulado=integral_linea(self.trayectoria_3d[-50:])

                # Divergencia y rotacional en tiempo real
                if self.frame%20==0:
                    self.div_actual=divergencia(dron.posicion.x,dron.posicion.y,dron.posicion.z)
                    self.rot_magnitud=rotacional(dron.posicion.x,dron.posicion.y,dron.posicion.z).magnitud()

            if self.frame%8==0 and dron.velocidad.magnitud()>0.2:
                self.sonido.motor()

            self._cam_lerp(dron)

            # ── RENDER ────────────────────────────────────────────────────────
            self._fondo(pantalla)

            cdx=self.cam_x*self.zoom; cdy=self.cam_y*self.zoom
            for capa in self.capas:
                capa.dibujar(pantalla,cdx,cdy)

            self._nubes(pantalla)
            self._grid(pantalla)

            # Capas matemáticas opcionales
            if self.mostrar_flujo:
                self._flujo(pantalla)
            if self.mostrar_campo_vec:
                self._dibujar_campo_vectorial(pantalla,dron)

            self._trayectoria_draw(pantalla)

            # Obstáculos
            for i,obs in enumerate(self.sim.entorno.obstaculos):
                sx,sy=self.w2s(obs.posicion.x,obs.posicion.y)
                radio_px=max(8,int(obs.radio*self.zoom*0.7))
                self._obstaculo(pantalla,sx,sy,radio_px,obs.nombre,i%3)

            # Destino
            dx,dy=self.w2s(destino.x,destino.y)
            self._destino(pantalla,dx,dy)

            # Proyectiles
            self._actualizar_proyectiles(pantalla,destino)

            # Propulsión
            sx_d,sy_d=self.w2s(dron.posicion.x,dron.posicion.y)
            if not self.pausa and self.frame%2==0:
                self._propulsion(sx_d,sy_d,dron)

            # Partículas
            vivas=[]
            for p in self.particulas:
                p.actualizar()
                if p.vivo(): p.dibujar(pantalla); vivas.append(p)
            self.particulas=vivas

            # Explosiones
            nex=[]
            for ex in self.explosiones:
                ex["x"]+=ex["vx"]; ex["y"]+=ex["vy"]; ex["vida"]-=1
                if ex["vida"]>0:
                    c=ex.get("color",(255,120,0))
                    pygame.draw.circle(pantalla,c,(int(ex["x"]),int(ex["y"])),ex["r"])
                    nex.append(ex)
            self.explosiones=nex

            # Sensores
            if self.mostrar_sensores:
                ss=pygame.Surface((self.ANCHO,self.ALTO),pygame.SRCALPHA)
                for ang in range(0,360,30):
                    a=math.radians(ang)
                    ex_=sx_d+math.cos(a)*85; ey_=sy_d+math.sin(a)*85
                    pygame.draw.line(ss,(0,180,255,35),(sx_d,sy_d),(int(ex_),int(ey_)),1)
                pantalla.blit(ss,(0,0))

            # Visualizaciones matemáticas (sobre el dron)
            if self.mostrar_gradiente:
                self._dibujar_gradiente(pantalla,dron)
            if self.mostrar_rotacional:
                self._dibujar_rotacional(pantalla,dron)

            # Dron
            self._dron(pantalla,sx_d,sy_d,dron)
            self._barra_vida(pantalla,sx_d,sy_d)

            # UI
            if self.mostrar_hud:
                self._hud(pantalla,dron,destino,fps)
            if self.mostrar_minimapa:
                self._minimapa(pantalla,dron,destino)
            if self.mostrar_radar:
                self._radar(pantalla,dron)

            # Botones
            for btn in self.botones:
                btn.dibujar(pantalla,self.fuente_small)

            # Indicador VEL MAX
            if self.fuente_small:
                vt=self.fuente_small.render(
                    f"VEL MAX: {self.vel_max_actual:.0f} m/s",True,(180,120,255))
                pantalla.blit(vt,(10,self.ALTO-78))

            # Flash colisión
            self._flash_colision(pantalla)

            # Game over
            if self.vida<=0:
                self.fuente_big and pantalla.blit(
                    self.fuente_big.render("💀 DRON DESTRUIDO — ESC para salir",True,(255,50,50)),
                    (self.ANCHO//2-280,self.ALTO//2-20))

            # Banner llegada
            if llegado and llegado_timer>0:
                llegado_timer-=1
                txt=self.fuente_big.render("✓  DESTINO ALCANZADO  ✓",True,(0,255,150))
                sx_l=self.ANCHO//2-txt.get_width()//2; sy_l=self.ALTO//2-35
                sf=pygame.Surface((txt.get_width()+30,txt.get_height()+18),pygame.SRCALPHA)
                sf.fill((0,25,18,190))
                pygame.draw.rect(sf,(0,200,120,80),(0,0,sf.get_width(),sf.get_height()),2,border_radius=10)
                pantalla.blit(sf,(sx_l-15,sy_l-9)); pantalla.blit(txt,(sx_l,sy_l))

            pygame.display.update()

        pygame.quit()
