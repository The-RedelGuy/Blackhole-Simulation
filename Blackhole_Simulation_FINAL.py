#!/usr/bin/env python3
"""
=============================================================================
  BLACK HOLE SIMULATION — Schwarzschild Geodesic Ray Tracer
=============================================================================
  Real-time, interactive, physics-accurate black hole visualization.
  GPU-accelerated via GLSL fragment shaders with RK4 geodesic integration.

  Physics:
    - Schwarzschild metric null geodesics (light ray tracing)
    - Gravitational lensing & Einstein rings
    - Accretion disk with Novikov-Thorne temperature profile
    - Doppler beaming (approaching side brighter/blue-shifted)
    - Gravitational redshift
    - Photon sphere at r = 1.5 rs
    - ISCO (innermost stable circular orbit) at r = 3 rs

  Controls:
    Mouse Drag .... Orbit camera around the black hole
    Scroll Wheel .. Zoom in / out
    W / S ......... Tilt camera up / down
    A / D ......... Rotate camera left / right
    Q / E ......... Roll camera
    Space ......... Toggle accretion disk
    1 / 2 / 3 ..... Color scheme (Interstellar / X-Ray / Cosmic)
    + / - ......... Increase / decrease black hole mass
    P ............. Pause / resume disk rotation
    H ............. Toggle physics HUD
    R ............. Reset camera & mass
    Esc ........... Quit
=============================================================================
"""

import sys
import math
import time
import struct

# ---------------------------------------------------------------------------
# Auto-install dependencies
# ---------------------------------------------------------------------------
def _ensure(pkg, pip_name=None):
    """Import a package, auto-installing via pip if missing."""
    try:
        return __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name or pkg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return __import__(pkg)

pygame   = _ensure("pygame")
moderngl = _ensure("moderngl")

# ═══════════════════════════════════════════════════════════════════════════
# GLSL SHADERS
# ═══════════════════════════════════════════════════════════════════════════

VERTEX_SHADER = """
#version 330 core
in vec2 in_position;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_position * 0.5 + 0.5;
}
"""

# ---------------------------------------------------------------------------
# Scene fragment shader — Schwarzschild geodesic ray tracer
# ---------------------------------------------------------------------------
SCENE_FRAGMENT_SHADER = """
#version 330 core

uniform vec2  u_resolution;
uniform float u_time;
uniform vec3  u_camera_pos;
uniform mat3  u_camera_rot;
uniform float u_fov;
uniform float u_rs;            // Schwarzschild radius
uniform int   u_show_disk;
uniform int   u_color_scheme;  // 0=Interstellar, 1=X-Ray, 2=Cosmic
uniform float u_disk_rotation;

out vec4 fragColor;

// ── Constants ──────────────────────────────────────────────────────────────
#define PI        3.14159265359
#define MAX_STEPS 200
#define MAX_DIST  50.0

// ── Hash helpers (procedural content) ──────────────────────────────────────
float hash21(vec2 p) {
    p  = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

// ── Procedural star field ──────────────────────────────────────────────────
vec3 starField(vec3 dir) {
    dir = normalize(dir);
    float theta = acos(clamp(dir.y, -1.0, 1.0));
    float phi   = atan(dir.z, dir.x);

    vec3 col = vec3(0.0);

    // Milky-way band
    float band  = exp(-6.0 * pow(abs(dir.y - 0.1 * sin(phi * 2.0)), 2.0));
    float n1    = hash21(floor(vec2(phi, theta) * 200.0)) * 0.5 + 0.5;
    float n2    = hash21(floor(vec2(phi, theta) * 80.0))  * 0.5 + 0.5;
    col += vec3(0.14, 0.10, 0.22) * band * n1 * n2 * 0.18;

    // Multi-layer stars
    for (int layer = 0; layer < 4; layer++) {
        float scale = 40.0 + float(layer) * 55.0;
        float thresh = 0.985 - float(layer) * 0.004;

        vec2 gUV  = vec2(phi / (2.0*PI) + 0.5, theta / PI) * scale;
        vec2 cell = floor(gUV);
        vec2 f    = fract(gUV);

        float h = hash21(cell + float(layer) * 137.0);
        if (h > thresh) {
            vec2 sp = vec2(hash21(cell*1.73 + float(layer)*51.0 + 0.5),
                           hash21(cell*2.44 + float(layer)*73.0 + 1.5));
            float d = length(f - sp);

            float sz  = 0.018 + 0.028 * hash21(cell*3.7 + 2.5);
            float bri = smoothstep(sz, 0.0, d)
                      * (0.3 + 0.7 * hash21(cell*5.0 + 3.5));

            // Spectral class colour
            float tc = hash21(cell * 7.0 + 4.5);
            vec3 sc;
            if      (tc < 0.12) sc = vec3(0.62, 0.72, 1.00); // O / B
            else if (tc < 0.28) sc = vec3(0.82, 0.87, 1.00); // A
            else if (tc < 0.48) sc = vec3(1.00, 1.00, 0.95); // F
            else if (tc < 0.68) sc = vec3(1.00, 0.96, 0.82); // G
            else if (tc < 0.84) sc = vec3(1.00, 0.80, 0.52); // K
            else                sc = vec3(1.00, 0.52, 0.32);  // M

            col += sc * bri * (0.8 + float(layer) * 0.3);
        }
    }
    return col;
}

// ── Accretion disk colour ──────────────────────────────────────────────────
vec3 accretionDisk(vec3 hitPos, float r, float rs, vec3 rayDir) {

    float isco = 3.0 * rs;
    float rOut = 14.0 * rs;
    if (r < isco * 0.9 || r > rOut) return vec3(0.0);

    float rNorm = clamp((r - isco) / (rOut - isco), 0.0, 1.0);

    // Novikov–Thorne temperature profile (simplified)
    float x    = r / isco;
    float temp = pow(x, -0.75) * pow(max(1.0 - sqrt(1.0/x), 0.001), 0.25);
    temp = clamp(temp, 0.0, 2.5);

    // Keplerian orbital velocity → Doppler factor
    float v_orb   = sqrt(0.5 * rs / r);
    vec3  radDir  = normalize(vec3(hitPos.x, 0.0, hitPos.z));
    vec3  orbDir  = vec3(-radDir.z, 0.0, radDir.x);          // ⊥ radial
    float doppler = 1.0 / (1.0 + dot(normalize(rayDir), orbDir) * v_orb * 4.0);
    doppler = clamp(doppler, 0.2, 5.0);

    // Gravitational redshift
    float gz = sqrt(max(1.0 - rs / r, 0.01));

    float T = temp * doppler * gz;

    // Disk-angle noise (spiral turbulence)
    float ang   = atan(hitPos.z, hitPos.x) + u_disk_rotation;
    float turb  = 0.7 + 0.3 * (
        sin(ang*8.0  + r*3.0)                     * 0.45
      + sin(ang*23.0 - r*7.0  + u_time*0.5)       * 0.30
      + sin(ang*47.0 + r*11.0 - u_time*0.3)       * 0.25
    );

    // Edge falloffs
    float inner = smoothstep(isco*0.9, isco*1.25, r);
    float outer = smoothstep(rOut,     rOut*0.65,  r);

    // ── Colour schemes ─────────────────────────────────────────────────
    vec3 color;
    if (u_color_scheme == 0) {                       // Interstellar
        vec3 cIn  = vec3(1.0,  0.95, 0.85);
        vec3 cMid = vec3(1.0,  0.55, 0.12);
        vec3 cOut = vec3(0.55, 0.12, 0.02);
        color = rNorm < 0.3
              ? mix(cIn, cMid, rNorm / 0.3)
              : mix(cMid, cOut, (rNorm-0.3) / 0.7);
    } else if (u_color_scheme == 1) {                // X-Ray
        vec3 cIn  = vec3(0.92, 0.96, 1.0);
        vec3 cMid = vec3(0.20, 0.50, 1.0);
        vec3 cOut = vec3(0.04, 0.08, 0.35);
        color = rNorm < 0.3
              ? mix(cIn, cMid, rNorm / 0.3)
              : mix(cMid, cOut, (rNorm-0.3) / 0.7);
    } else {                                         // Cosmic
        vec3 cIn  = vec3(0.92, 0.82, 1.0);
        vec3 cMid = vec3(0.70, 0.10, 0.90);
        vec3 cOut = vec3(0.02, 0.25, 0.50);
        color = rNorm < 0.3
              ? mix(cIn, cMid, rNorm / 0.3)
              : mix(cMid, cOut, (rNorm-0.3) / 0.7);
    }

    float bri = T * T * 2.0;
    color *= clamp(bri, 0.0, 8.0) * turb * inner * outer;

    // Doppler colour shift
    if (doppler > 1.0)
        color = mix(color, color * vec3(0.80, 0.90, 1.25),
                    min((doppler-1.0)*0.5, 0.5));
    else
        color = mix(color, color * vec3(1.30, 0.70, 0.40),
                    min((1.0-doppler)*0.5, 0.5));

    return color;
}

// ── ACES filmic tone-map ───────────────────────────────────────────────────
vec3 ACES(vec3 x) {
    return clamp((x*(2.51*x + 0.03)) / (x*(2.43*x + 0.59) + 0.14), 0.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
void main() {

    vec2 uv = (gl_FragCoord.xy - 0.5*u_resolution)
            / min(u_resolution.x, u_resolution.y);

    // ── Camera ray ─────────────────────────────────────────────────────────
    float fovF   = tan(u_fov * 0.5);
    vec3  rayDir = normalize(u_camera_rot * vec3(uv * fovF, -1.0));
    vec3  rayPos = u_camera_pos;

    float rs = u_rs;

    // ── Ray-march through Schwarzschild spacetime ──────────────────────────
    vec3  diskAccum = vec3(0.0);
    float prevY     = rayPos.y;
    float minR      = MAX_DIST;
    bool  absorbed  = false;

    for (int i = 0; i < MAX_STEPS; i++) {
        float r = length(rayPos);
        minR = min(minR, r);

        // Fallen past event horizon
        if (r < rs * 0.5) { absorbed = true; break; }

        // Escaped to infinity
        if (r > MAX_DIST && dot(rayPos, rayDir) > 0.0) break;

        // Adaptive step-size (small near BH, large far away)
        float dt = clamp((r - rs) * 0.12, 0.008, 0.45);

        // ── Geodesic acceleration ──────────────────────────────────────────
        // a  = −1.5 · rs · h² / r⁵ · r⃗
        // where h = |r⃗ × v⃗|  (conserved angular momentum magnitude)
        // Derived from Schwarzschild effective potential for null geodesics.

        vec3  h  = cross(rayPos, rayDir);
        float h2 = dot(h, h);
        float r5 = r*r*r*r*r;
        vec3  a0 = -1.5 * rs * h2 / r5 * rayPos;

        // ── RK4 integration ────────────────────────────────────────────────
        vec3 k1x = rayDir * dt;
        vec3 k1v = a0      * dt;

        vec3 p2 = rayPos + k1x*0.5;  vec3 v2 = rayDir + k1v*0.5;
        float r2 = length(p2);
        vec3 h2v = cross(p2, v2);
        vec3 a2  = -1.5*rs*dot(h2v,h2v) / (r2*r2*r2*r2*r2) * p2;
        vec3 k2x = v2 * dt;          vec3 k2v = a2 * dt;

        vec3 p3 = rayPos + k2x*0.5;  vec3 v3 = rayDir + k2v*0.5;
        float r3 = length(p3);
        vec3 h3v = cross(p3, v3);
        vec3 a3  = -1.5*rs*dot(h3v,h3v) / (r3*r3*r3*r3*r3) * p3;
        vec3 k3x = v3 * dt;          vec3 k3v = a3 * dt;

        vec3 p4 = rayPos + k3x;      vec3 v4 = rayDir + k3v;
        float r4 = length(p4);
        vec3 h4v = cross(p4, v4);
        vec3 a4  = -1.5*rs*dot(h4v,h4v) / (r4*r4*r4*r4*r4) * p4;
        vec3 k4x = v4 * dt;          vec3 k4v = a4 * dt;

        vec3 newPos = rayPos + (k1x + 2.0*k2x + 2.0*k3x + k4x) / 6.0;
        rayDir      = normalize(rayDir + (k1v + 2.0*k2v + 2.0*k3v + k4v) / 6.0);

        // ── Accretion-disk crossing (y = 0 plane) ─────────────────────────
        if (u_show_disk == 1) {
            float newY = newPos.y;
            if (prevY * newY < 0.0) {                     // crossed plane
                float t   = prevY / (prevY - newY);
                vec3  cp  = mix(rayPos, newPos, t);
                float cr  = length(cp);
                vec3  dc  = accretionDisk(cp, cr, rs, rayDir);
                float opc = clamp(length(dc) * 0.35, 0.0, 0.92);
                diskAccum += dc * (1.0 - clamp(length(diskAccum)/3.0, 0.0, 1.0));
            }
            prevY = newY;
        }

        // ── Faint volumetric jet glow ─────────────────────────────────────
        if (u_show_disk == 1) {
            float jy = abs(normalize(rayPos).y);
            if (jy > 0.96 && r > 1.2*rs && r < 10.0*rs) {
                float jFade  = exp(-r / (4.5*rs));
                float jShape = smoothstep(0.96, 1.0, jy);
                vec3  jCol;
                if (u_color_scheme == 0) jCol = vec3(1.0, 0.6, 0.2);
                else if (u_color_scheme == 1) jCol = vec3(0.3, 0.55, 1.0);
                else                          jCol = vec3(0.6, 0.2, 1.0);
                diskAccum += jCol * jShape * jFade * dt * 0.18;
            }
        }

        rayPos = newPos;
    }

    // ── Final colour composition ───────────────────────────────────────────
    vec3 color = vec3(0.0);

    if (absorbed) {
        color = vec3(0.0);
    } else if (length(rayPos) >= MAX_DIST) {
        color = starField(rayDir);       // Gravitationally-lensed starfield

        // Photon-ring brightening (rays that grazed the photon sphere)
        float ringGlow = exp(-2.5 * max(minR/rs - 1.5, 0.0));
        color *= 1.0 + ringGlow * 2.0;
    }

    // Accumulate disk / jet emission
    color += diskAccum;

    // Vignette
    vec2 vUV = gl_FragCoord.xy / u_resolution;
    color *= 1.0 - 0.25 * dot(vUV - 0.5, vUV - 0.5) * 2.0;

    // Tone-map & gamma
    color = ACES(color);
    color = pow(color, vec3(1.0/2.2));

    fragColor = vec4(color, 1.0);
}
"""

# ---------------------------------------------------------------------------
# HUD overlay fragment shader
# ---------------------------------------------------------------------------
HUD_FRAGMENT_SHADER = """
#version 330 core
uniform sampler2D u_hud_tex;
in  vec2 v_uv;
out vec4 fragColor;

void main() {
    fragColor = texture(u_hud_tex, v_uv);
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# CAMERA
# ═══════════════════════════════════════════════════════════════════════════

class Camera:
    """Orbital camera around the origin (spherical coordinates)."""

    def __init__(self):
        self.theta    = 0.35          # elevation  (rad)
        self.phi      = 0.0           # azimuth    (rad)
        self.distance = 15.0
        self.roll     = 0.0
        self.fov      = 1.2           # radians (~69°)
        self.min_dist = 2.8
        self.max_dist = 50.0
        self.sens     = 0.005

    # -- helpers ----------------------------------------------------------
    def position(self):
        ct, st = math.cos(self.theta), math.sin(self.theta)
        cp, sp = math.cos(self.phi),   math.sin(self.phi)
        return (self.distance * ct * sp,
                self.distance * st,
                self.distance * ct * cp)

    def rotation_matrix_column_major(self):
        """Return 9-float tuple (column-major) for the camera→world mat3."""
        px, py, pz = self.position()
        # forward = normalize(origin – pos) = normalize(-pos)
        d  = math.sqrt(px*px + py*py + pz*pz)
        fx, fy, fz = -px/d, -py/d, -pz/d

        # world up
        wux, wuy, wuz = 0.0, 1.0, 0.0
        # right = forward × world_up
        rx = fy*wuz - fz*wuy
        ry = fz*wux - fx*wuz
        rz = fx*wuy - fy*wux
        rl = math.sqrt(rx*rx + ry*ry + rz*rz)
        if rl < 1e-6:                         # looking straight up/down
            wux, wuy, wuz = 0.0, 0.0, -1.0
            rx = fy*wuz - fz*wuy
            ry = fz*wux - fx*wuz
            rz = fx*wuy - fy*wux
            rl = math.sqrt(rx*rx + ry*ry + rz*rz)
        rx /= rl;  ry /= rl;  rz /= rl

        # up = right × forward
        ux = ry*fz - rz*fy
        uy = rz*fx - rx*fz
        uz = rx*fy - ry*fx

        # apply roll (rotate right & up around forward)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        r2x = cr*rx + sr*ux;  r2y = cr*ry + sr*uy;  r2z = cr*rz + sr*uz
        u2x = -sr*rx + cr*ux; u2y = -sr*ry + cr*uy; u2z = -sr*rz + cr*uz

        #  col0 = right,  col1 = up,  col2 = −forward
        return (r2x, r2y, r2z,
                u2x, u2y, u2z,
                -fx, -fy, -fz)

    # -- controls ---------------------------------------------------------
    def orbit(self, dx, dy):
        self.phi   -= dx * self.sens
        self.theta  = max(-1.5, min(1.5, self.theta + dy * self.sens))

    def zoom(self, amount, rs=2.0):
        self.min_dist = max(2.5, rs * 1.5)
        self.distance *= 1.0 - amount * 0.1
        self.distance  = max(self.min_dist, min(self.max_dist, self.distance))

    def reset(self):
        self.__init__()


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

class BlackHoleSimulation:
    """Main simulation class — owns window, shaders, state, loop."""

    WIDTH, HEIGHT = 1280, 720
    TITLE = "Black Hole Simulation  |  Schwarzschild Geodesic Ray Tracer"

    # ── construction ────────────────────────────────────────────────────────
    def __init__(self):
        pygame.init()
        pygame.mixer.quit()                    # avoid audio-device issues

        self.width  = self.WIDTH
        self.height = self.HEIGHT

        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
        )
        pygame.display.set_caption(self.TITLE)

        # ModernGL context from the Pygame GL surface
        try:
            self.ctx = moderngl.create_context()
        except Exception as exc:
            print(f"[ERROR] Cannot create OpenGL 3.3 context: {exc}")
            print("Please update your graphics drivers and retry.")
            pygame.quit()
            sys.exit(1)

        # Simulation state
        self.camera        = Camera()
        self.mass           = 1.0
        self.rs             = 2.0 * self.mass    # Schwarzschild radius
        self.sim_time       = 0.0
        self.disk_rotation  = 0.0
        self.show_disk      = True
        self.show_hud       = True
        self.color_scheme   = 0                   # 0/1/2
        self.paused         = False
        self.running        = True
        self.dragging       = False
        self.last_mouse     = (0, 0)

        # Performance
        self.clock       = pygame.time.Clock()
        self.fps         = 0.0
        self._ftimes     = []

        # Build GPU objects
        self._build_quad()
        self._build_scene_shader()
        self._build_hud()

    # ── GPU setup ───────────────────────────────────────────────────────────
    def _build_quad(self):
        """Full-screen triangle-strip quad shared by scene & HUD."""
        data = struct.pack("8f", -1, -1,  1, -1,  -1, 1,  1, 1)
        self.quad_vbo = self.ctx.buffer(data)

    def _build_scene_shader(self):
        try:
            self.scene_prog = self.ctx.program(
                vertex_shader   = VERTEX_SHADER,
                fragment_shader = SCENE_FRAGMENT_SHADER,
            )
        except Exception as exc:
            print(f"[SHADER ERROR] {exc}")
            pygame.quit()
            sys.exit(1)

        self.scene_vao = self.ctx.simple_vertex_array(
            self.scene_prog, self.quad_vbo, "in_position"
        )

    def _build_hud(self):
        self.hud_prog = self.ctx.program(
            vertex_shader   = VERTEX_SHADER,
            fragment_shader = HUD_FRAGMENT_SHADER,
        )
        self.hud_vao = self.ctx.simple_vertex_array(
            self.hud_prog, self.quad_vbo, "in_position"
        )

        # Texture (RGBA, same size as window)
        self.hud_tex = self.ctx.texture(
            (self.width, self.height), 4,
            b"\x00" * (self.width * self.height * 4),
        )
        self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Fonts (Consolas → Courier New → default)
        for name in ("Consolas", "Courier New", "monospace", None):
            try:
                self.font_lg = pygame.font.SysFont(name, 22, bold=True)
                self.font_sm = pygame.font.SysFont(name, 16)
                self.font_ti = pygame.font.SysFont(name, 26, bold=True)
                break
            except Exception:
                continue

        self.hud_dirty   = True
        self.hud_counter = 0

    # ── HUD rendering (Pygame surface → GPU texture) ────────────────────────
    def _refresh_hud_texture(self):
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        if self.show_hud:
            y   = 18
            gap = 26

            # Title
            surf.blit(
                self.font_ti.render("BLACK HOLE SIMULATION", True,
                                    (255, 255, 255, 230)),
                (18, y),
            )
            y += 38
            pygame.draw.line(surf, (100, 100, 100, 140), (18, y), (330, y), 1)
            y += 12

            # Redshift at camera
            d = self.camera.distance
            if d > self.rs:
                zval = 1.0 / math.sqrt(abs(1.0 - self.rs / d)) - 1.0
                zstr = f"{zval:.4f}"
            else:
                zstr = "infinite"

            scheme_names = ("Interstellar", "X-Ray", "Cosmic")
            lines = [
                ("Mass",              f" {self.mass:.2f} Solar Masses"),
                ("Schwarzschild R",   f" {self.rs:.2f}"),
                ("Photon Sphere",     f" {1.5*self.rs:.2f}"),
                ("ISCO",              f" {3.0*self.rs:.2f}"),
                ("Camera Distance",   f" {d:.1f}"),
                ("Redshift (camera)", f" {zstr}"),
                ("Colour Scheme",     f" {scheme_names[self.color_scheme]}"),
                ("Accretion Disk",    f" {'ON' if self.show_disk else 'OFF'}"),
                ("Disk Rotation",     f" {'PAUSED' if self.paused else 'ACTIVE'}"),
                ("FPS",               f" {self.fps:.0f}"),
            ]
            label_col = (140, 175, 255, 210)
            value_col = (255, 255, 255, 235)
            for lbl, val in lines:
                ls = self.font_sm.render(lbl + ":", True, label_col)
                vs = self.font_sm.render(val, True, value_col)
                surf.blit(ls, (18, y))
                surf.blit(vs, (18 + ls.get_width(), y))
                y += gap

            # Controls hint (bottom-left)
            hints = [
                "Drag: Orbit  |  Scroll: Zoom  |  WASD/QE: Move",
                "1/2/3: Colour  |  +/-: Mass  |  Space: Disk  |  P: Pause",
                "H: HUD  |  R: Reset  |  Esc: Quit",
            ]
            hy = self.height - 18 - len(hints) * 22
            for h in hints:
                surf.blit(
                    self.font_sm.render(h, True, (170, 170, 170, 140)),
                    (18, hy),
                )
                hy += 22

        # Upload
        try:
            data = pygame.image.tobytes(surf, "RGBA", True)
        except AttributeError:                 # older Pygame
            data = pygame.image.tostring(surf, "RGBA", True)

        self.hud_tex.write(data)

    # ── resize handling ─────────────────────────────────────────────────────
    def _on_resize(self, w, h):
        if w < 64 or h < 64:
            return
        self.width, self.height = w, h
        self.ctx.viewport = (0, 0, w, h)

        # Recreate HUD texture at new resolution
        self.hud_tex.release()
        self.hud_tex = self.ctx.texture(
            (w, h), 4, b"\x00" * (w * h * 4),
        )
        self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hud_dirty = True

    # ── input ───────────────────────────────────────────────────────────────
    def _process_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False

            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    self.running = False
                elif k == pygame.K_SPACE:
                    self.show_disk = not self.show_disk;  self.hud_dirty = True
                elif k == pygame.K_h:
                    self.show_hud = not self.show_hud;    self.hud_dirty = True
                elif k == pygame.K_p:
                    self.paused = not self.paused;         self.hud_dirty = True
                elif k == pygame.K_r:
                    self.camera.reset()
                    self.mass = 1.0; self.rs = 2.0;       self.hud_dirty = True
                elif k == pygame.K_1:
                    self.color_scheme = 0;                 self.hud_dirty = True
                elif k == pygame.K_2:
                    self.color_scheme = 1;                 self.hud_dirty = True
                elif k == pygame.K_3:
                    self.color_scheme = 2;                 self.hud_dirty = True
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.mass = min(10.0, self.mass + 0.1)
                    self.rs = 2.0 * self.mass;             self.hud_dirty = True
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.mass = max(0.2, self.mass - 0.1)
                    self.rs = 2.0 * self.mass;             self.hud_dirty = True

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    self.dragging  = True
                    self.last_mouse = ev.pos

            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    self.dragging = False

            elif ev.type == pygame.MOUSEMOTION:
                if self.dragging:
                    dx = ev.pos[0] - self.last_mouse[0]
                    dy = ev.pos[1] - self.last_mouse[1]
                    self.camera.orbit(dx, dy)
                    self.last_mouse = ev.pos

            elif ev.type == pygame.MOUSEWHEEL:
                self.camera.zoom(ev.y, self.rs)
                self.hud_dirty = True

            elif ev.type == pygame.VIDEORESIZE:
                self._on_resize(ev.w, ev.h)

    def _process_keys(self, dt):
        keys = pygame.key.get_pressed()
        spd  = 2.0 * dt
        if keys[pygame.K_w]:
            self.camera.theta = min( 1.5, self.camera.theta + spd)
        if keys[pygame.K_s]:
            self.camera.theta = max(-1.5, self.camera.theta - spd)
        if keys[pygame.K_a]:
            self.camera.phi += spd
        if keys[pygame.K_d]:
            self.camera.phi -= spd
        if keys[pygame.K_q]:
            self.camera.roll -= dt
        if keys[pygame.K_e]:
            self.camera.roll += dt

    # ── update ──────────────────────────────────────────────────────────────
    def _update(self, dt):
        self.sim_time += dt
        if not self.paused:
            self.disk_rotation += dt * 0.35

        # FPS
        now = time.perf_counter()
        self._ftimes.append(now)
        self._ftimes = [t for t in self._ftimes if now - t < 1.0]
        self.fps = len(self._ftimes)

    # ── render ──────────────────────────────────────────────────────────────
    def _render(self):
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # ── Scene pass ──────────────────────────────────────────────────────
        prog = self.scene_prog
        prog["u_resolution"].value    = (float(self.width), float(self.height))
        prog["u_time"].value          = self.sim_time
        prog["u_camera_pos"].value    = self.camera.position()
        prog["u_camera_rot"].write(
            struct.pack("9f", *self.camera.rotation_matrix_column_major())
        )
        prog["u_fov"].value           = self.camera.fov
        prog["u_rs"].value            = self.rs
        prog["u_show_disk"].value     = 1 if self.show_disk else 0
        prog["u_color_scheme"].value  = self.color_scheme
        prog["u_disk_rotation"].value = self.disk_rotation

        self.scene_vao.render(moderngl.TRIANGLE_STRIP)

        # ── HUD overlay ─────────────────────────────────────────────────────
        self.hud_counter += 1
        if self.hud_dirty or self.hud_counter >= 8:
            self._refresh_hud_texture()
            self.hud_dirty   = False
            self.hud_counter = 0

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA,
                               moderngl.ONE_MINUS_SRC_ALPHA)
        self.hud_tex.use(0)
        self.hud_prog["u_hud_tex"].value = 0
        self.hud_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

        pygame.display.flip()

    # ── main loop ───────────────────────────────────────────────────────────
    def run(self):
        prev = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            dt  = min(now - prev, 0.1)
            prev = now

            self._process_events()
            self._process_keys(dt)
            self._update(dt)
            self._render()
            self.clock.tick(60)

        pygame.quit()


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    BlackHoleSimulation().run()
