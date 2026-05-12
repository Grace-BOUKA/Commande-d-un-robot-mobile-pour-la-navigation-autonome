import pygame
import numpy as np
import cv2
import matplotlib.pyplot as plt
from robot import Robot_Differentiel
from camera import Camera_Virtuel
from ibvs import Controlleur_IBVS

# X0,Y0,theta,Xu,Yu=3,5,0*np.pi,3,2 # Valeurs par defaut

##================================================= INTERFACE UTILISATEUR ===========================================================
print("===========================================================================================================\n")
print("                        SIMULATEUR DE LA NAVIGATION AUTONOME D'UN ROBOT DIFFERENTIEL\n")
print("===========================================================================================================\n\n")

# Position de depart du robot
print("Choisir la position (x,y,thetha) de depart du robot: x=[0; 10](m), y=[0;10](m), theta=[-pi;pi](en radian)")
X0=float(input("Valeur de x: "))
Y0=float(input("Valeur de y: "))
theta=np.pi*float(input("Valeur de theta: "))

print("\n\n")

# Ajout des obstacles (non obstruant) (x, y, rayon = 0.3)
nbr_obstacles= int(input("Combien d'obstacles vous desirer ?: "))
OBSTACLES = [] # Liste des obstacles

i=0
while(nbr_obstacles > i):
    # Position des obstacles
    print(f"\nChoisir la position de l'obstacle (x,y)  N°{i+1} : x=[0; 10](m), y=[0;10](m)")
    Xu=float(input("Valeur de x: "))
    Yu=float(input("Valeur de y: "))
    OBSTACLES.append((Xu,-Yu,0.3))
    
    i+=1
    
print("\n\n")





##================================================ INITIALISATION DES VARIABLES ====================================================


# Initailisation de la fenetre, macros sur les couleurs et ses parametres
pygame.init()
pygame.display.set_caption("SIMULATEUR IBVS ROBOT DIFFERENTIEL")
ecran = pygame.display.set_mode((1300, 820))
clock = pygame.time.Clock()
WHITE, BLUE, RED, GREEN, YELLOW, BLACK = (255,255,255), (0,100,200), (200,0,0), (0,200,0), (255,255,0), (0,0,0)
GRAY, DARK_GRAY = (60,60,60), (40,40,40)
ECHELLE, OFFSET_X, OFFSET_Y = 80, 10, 10

# Initialisation des objets (Robot + Camera + Controlleur IBVS + Cibles + Bruits + Obstacles)
START_POSE = (X0, -Y0, theta) # Pose initiale du robot
robot = Robot_Differentiel(*START_POSE)
camera = Camera_Virtuel(focal=140, resolution=(320, 240))
controller = Controlleur_IBVS(camera, gain=0.3)

# Cible par défaut en coordonées pixels
controller.s_star= np.array([58, 21, 255, 21, 250, 213, 63, 213])
print(f"\nCible s* (rectangle) par defaut en pixels {controller.s_star}\n")

# Cibles (4 points du marqueur) en coordonnées primitives
START_TARGETS = [
    # [np.array([4.0, -5.0, 0.5]), np.array([5, -5.0, 0.5]), np.array([5, -5.0, -0.5]), np.array([4.0, -5.0, -0.5])],
    [np.array([2.0, -0.22, 0.5]), np.array([3, -0.22, 0.5]), np.array([3, -0.2, -0.5]), np.array([2.0, -0.2, -0.5])],
    
]
CIBLES = list(START_TARGETS) # Liste des cibles

# Intensité du bruit (un écart-type en pixels)
BRUIT = 0.0

# # Ajout des obstacles (x, y, rayon)
# OBSTACLES = [
#     # (6.0, -2.0, 0.3), 
#     # (6.0, -6, 0.3),
#     # (8, -3.0, 0.3),
#     (Xu,-Yu,0.3)
# ]

# Variables pour le tracé de l'erreur visuelle e, des entrées de commandes et trajectoire dans le plan image de s
erreurs_par_point = [[], [], [], []]
v_g_history, v_d_history = [], []
w_robot_history, w_cam_history = [], []
time_history, current_time = [], 0.0

history_u = [[], [], [], []] # 4 listes pour les coordonnées u des 4 points
history_v = [[], [], [], []] # 4 listes pour les coordonnées v des 4 points

# Historique des positions pour la trajectoire du robot
trajectoire_points = [] 
orientation_history = []

##=================================================================================================================================





##============================================== FONCTIONS DE LA BOUCLE DE SIMULATION =============================================


# --------------------------------------------- Conversion coordonnées géometriques --> coordonnées écran -------------------------
def monde_vers_ecran(x, y): 
    return (int(x * ECHELLE + OFFSET_X), int(-y * ECHELLE + OFFSET_Y))


# ---------------------------------------------- Cible dans le champ de vision (Recherche cible) ----------------------------------
def champ_vision(target_pts, robot_pose, cam):
    for pt in target_pts:
        tx, ty, _ = pt
        x_r, y_r, theta_r = robot_pose
        
        # Position et angle réels de la caméra
        cam_x = x_r + cam.a * np.cos(theta_r)
        cam_y = y_r + cam.a * np.sin(theta_r)
        cam_theta = theta_r + cam.pan_angle
        
        dx, dy = tx - cam_x, ty - cam_y
        angle = (np.arctan2(dy, dx) - cam_theta + np.pi) % (2 * np.pi) - np.pi  #[modulo 2pi]
        if abs(np.degrees(angle)) <= 60: return True # Aire du champ de vison de la camera à 60°
    return False


# --------------------------------------------------- Affichage de la scene -------------------------------------------------------
def dessiner_interface(surface, robot_pose, current_targets):
    surface.fill(DARK_GRAY)
    # --------- Grille grise sur la fenetre
    grille_x, grille_y=10,10
    for i in range(-grille_x, grille_y):
        pygame.draw.line(surface, GRAY, monde_vers_ecran(i+1, -grille_y), monde_vers_ecran(i+1, grille_y), 1)
        pygame.draw.line(surface, GRAY, monde_vers_ecran(-grille_x, i), monde_vers_ecran(grille_x, i), 1)
        
    
    # --------- Dessin de la trajectoire du robot
    if len(trajectoire_points) > 1:
        # On convertit toutes les coordonnées du monde en coordonnées écran
        trace_robot = [monde_vers_ecran(p[0], p[1]) for p in trajectoire_points]
        # Ligne pour la visibilité de la trajectoire
        pygame.draw.lines(surface, (0, 255, 255), False, trace_robot, 2)
        
        
    # --------- Dessin des obstacles  sur la scene + Gestion des collisions
    x_r, y_r, theta_r = robot_pose # coordonnées primitives du robot
    largeur_robot = 0.2
    d_influence = 0.5
    for ox, oy, r_obs in OBSTACLES:
        dx = ox - x_r
        dy = oy - y_r
        dist = np.sqrt(dx**2 + dy**2) # Distance centre obstacle - centre du robot
        dist_bord = dist - r_obs - largeur_robot # Distance bord obstacle - bord du robot
               
        if 0 < dist_bord < d_influence: # Gestion Alerte collisions
            pygame.draw.circle(surface, RED, monde_vers_ecran(ox, oy), int(r_obs * ECHELLE)) # Alerte danger (obstacle en rouge)
 
        else: # Affichage obstacle (sans alerte)
            pygame.draw.circle(surface, WHITE, monde_vers_ecran(ox, oy), int(r_obs * ECHELLE))
            pygame.draw.circle(surface, RED, monde_vers_ecran(ox, oy), int(r_obs * ECHELLE),2)
        
        
    # --------- Gestion des cibles (Passer l'objet camera à champ_vision)
    for t_pts in current_targets:
        color = GREEN if champ_vision(t_pts, robot_pose, camera) else RED
        screen_pts = [monde_vers_ecran(p[0], p[1]) for p in t_pts]
        pygame.draw.polygon(surface, color, screen_pts, 4)
        

    # --------- Repere monde (X,Y,Z)
    origin = (10, 10)
    pygame.draw.line(surface, RED, origin, (origin[0] + 50, origin[1]), 3)
    pygame.draw.line(surface, GREEN, origin, (origin[0], origin[1]+50), 3)
    font_axes = pygame.font.SysFont(None, 20)
    surface.blit(font_axes.render("X", True, RED), (origin[0] + 50, origin[1]))
    surface.blit(font_axes.render("Y", True, GREEN), (origin[0], origin[1] + 50))
    pygame.draw.circle(surface, BLACK, origin, 5)
    

    # --------- Variables primitives du robot en variables ecran
    cx_rob, cy_rob = monde_vers_ecran(x_r, y_r)
    
    
    # --------- Position physique de la caméra (coordonnées primitives en coordonnées écran)
    cam_x = x_r + camera.a * np.cos(theta_r)
    cam_y = y_r + camera.a * np.sin(theta_r)
    cam_theta = theta_r + camera.pan_angle
    cx_cam, cy_cam = monde_vers_ecran(cam_x, cam_y)
    
    
    # --------- Cône de vision aligné sur l'aire de balayage de la camera
    length = 2 * ECHELLE
    l_end = (cx_cam + length * np.cos(cam_theta - np.radians(50)), cy_cam - length * np.sin(cam_theta - np.radians(50)))
    r_end = (cx_cam + length * np.cos(cam_theta + np.radians(50)), cy_cam - length * np.sin(cam_theta + np.radians(50)))
    cone_surf = pygame.Surface((1000, 820), pygame.SRCALPHA)
    pygame.draw.polygon(cone_surf, (255, 255, 0, 45), [(cx_cam, cy_cam), l_end, r_end])
    surface.blit(cone_surf, (0, 0))
    
    
    # ---------- Dessin du Robot
    size = 20 
    pts = [(cx_rob + size*np.cos(theta_r), cy_rob - size*np.sin(theta_r)), 
           (cx_rob + size*np.cos(theta_r+2.4), cy_rob - size*np.sin(theta_r+2.4)), 
           (cx_rob + size*np.cos(theta_r-2.4), cy_rob - size*np.sin(theta_r-2.4))]
    pygame.draw.polygon(surface, BLUE, pts)
    pygame.draw.circle(surface, BLACK, (cx_rob, cy_rob), 5)
    
    
    # ----------- Repère Robot (Xr, Yr, Zr)
    pygame.draw.line(surface, RED, (cx_rob, cy_rob),(cx_rob+30*np.cos(theta_r),cy_rob-30*np.sin(theta_r)), 2)
    pygame.draw.line(surface, GREEN, (cx_rob, cy_rob),(cx_rob-30*np.cos(theta_r+np.pi/2),cy_rob+30*np.sin(theta_r+np.pi/2)), 2)
    font_axes_r = pygame.font.SysFont(None, 20)
    surface.blit(font_axes_r.render("X_r", True, RED), (cx_rob+30*np.cos(theta_r),cy_rob-30*np.sin(theta_r)))
    surface.blit(font_axes_r.render("Y_r", True, GREEN), (cx_rob-30*np.cos(theta_r+np.pi/2),cy_rob+30*np.sin(theta_r+np.pi/2))) 
    
    
    # ----------- Dessin caméra et son axe de visée propre X_c
    pygame.draw.circle(surface, WHITE, (cx_cam, cy_cam), 5)
    pygame.draw.line(surface, YELLOW, (cx_cam, cy_cam), (cx_cam - 40*np.sin(cam_theta), cy_cam - 40*np.cos(cam_theta)), 3)
    surface.blit(font_axes_r.render("X_c", True, WHITE), (cx_cam - 40*np.sin(cam_theta), cy_cam - 40*np.cos(cam_theta))) 
    pygame.draw.line(surface, YELLOW, (cx_cam, cy_cam), (cx_cam + 40*np.cos(cam_theta), cy_cam - 40*np.sin(cam_theta)), 3)
    surface.blit(font_axes_r.render("Z_c", True, WHITE), (cx_cam + 40*np.cos(cam_theta), cy_cam - 40*np.sin(cam_theta))) 
     
##=============================================== FIN FONCTIONS BOUCLE DE SIMULATION ==============================================





## ================================================== BOUCLE PRINCIPALE ===========================================================
auto_mode = False # Variable pour le mode automatique / manuel
running = True # Variable de la boucle de simulation de la fenetre

while running:
    T_ech = clock.tick(60) / 1000.0 # Conversion temps en ms
    current_time += T_ech # vecteur temps
    v_g, v_d, w_cam_affichage,w_robot_affichage, x_err_affichage = 0, 0, 0.0,0.0, None # Initialisation des entrées de commande
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a: auto_mode = not auto_mode # Touche A
            if event.key == pygame.K_s: # Appuie sur S por fixer la cible s*
                if cible_visible:
                    controller.s_star = controller.sort_points(cible_visible[0][1])
                    print("Cible s* enregistrée en pixels!")
                    print(f"{controller.s_star}")
            if event.key == pygame.K_r: # Touche R
                #------------------RESET COMPLET--------------------
                robot = Robot_Differentiel(*START_POSE)
                camera.pan_angle=0.00
                CIBLES = list(START_TARGETS)
                auto_mode = False
                erreurs_par_point = [[], [], [], []]
                v_g_history, v_d_history = [], []
                w_robot_history, w_cam_history = [], []
                time_history, current_time = [], 0
                history_u = [[], [], [], []] # 4 listes pour les coordonnées u des 4 points
                history_v = [[], [], [], []] # 4 listes pour les coordonnées v des 4 points
                
                trajectoire_points = [] # Trajectoire dans le monde
                orientation_history = [] # Stockage de l'orientation du robot
                
                print("\nReset: Configuration de départ restaurée\n")

    # Obtenir les coordonnées normalisées de la vue courante cible s + images sur la camera (interface de vue de la camera)
    cam_img, cible_visible = camera.get_view(robot.get_pose(), CIBLES, OBSTACLES,s_star=controller.s_star)
    
    # ------------------- Injection du bruit sur les coordonnées des points détectés
    cible = [] # Cible + bruit éventuel
    if cible_visible:
        for cible_id, sommets in cible_visible:
            bruit_pts = []
            for (u, v) in sommets:
                # Ajout valeur aléatoire Gaussienne (moyenne 0, écart-type BRUIT)
                u_n = u + np.random.normal(0, BRUIT)
                v_n = v + np.random.normal(0, BRUIT)
                bruit_pts.append((u_n, v_n))
            cible.append((cible_id, bruit_pts))
            
            
    # ----------------------------------------------- Mode automatique (IBVS) --------------------------------------------------
               
    if auto_mode and CIBLES: # mode automatique
        # Récupération des commandes (On passe la cible, la position du robot et la liste des obstacles pour l'évitement)       
        v_c, w_c, w_cam, x_err, error_vec, s = controller.commande_IBVS(cible, robot_pose=robot.get_pose(),obstacles_list=OBSTACLES)
        
        x_err_affichage = x_err
        w_cam_affichage = w_cam
        w_robot_affichage= w_c
        
        # On tourne la platine (caméra)   
        camera.pan_angle += w_cam * T_ech
        
        # Modèle cinématique inverse (Application des commandes aux roues)
        v_d = (2*v_c + w_c * robot.entraxe) / (2 * robot.R_roue)
        v_g = (2*v_c - w_c * robot.entraxe) / (2 * robot.R_roue)

        # Echantillons des erreurs visuelles et des entrées de commandes
        if x_err is not None and error_vec is not None and len(error_vec) == 8:
            time_history.append(current_time)
            v_g_history.append(v_g)
            v_d_history.append(v_d)
            w_robot_history.append(w_c)
            w_cam_history.append(w_cam)
            
            # Orientation du robot dans le monde
            orientation_history.append(np.degrees(robot.theta)) 

            
            for i in range(4):
                # Erreurs visuelles e = s - s*
                e_i = np.sqrt(error_vec[2*i]**2 + error_vec[2*i+1]**2)
                erreurs_par_point[i].append(e_i)
                
                # Trajectoire dans le plan image de s  
                history_u[i].append(s[2*i])     # u du point i
                history_v[i].append(s[2*i+1])   # v du point i     
                                  
        
        
    # ------------------------------------------------- Mode manuelle (Avec les touches du clavier) --------------------------------
    
    elif not auto_mode: # mode manuel
        # Stockage de l'erreur visuelle
        if cible_visible: 
            all_pix = [pt for _, group in cible for pt in group]
            x_err_affichage = (sum(p[0] for p in all_pix)/len(all_pix)) - camera.u0
        
        # Commande manuelle des roues
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: v_g = v_d = 5 
        if keys[pygame.K_DOWN]: v_g = v_d = -5
        if keys[pygame.K_LEFT]: v_g, v_d = -0.5, 0.5
        if keys[pygame.K_RIGHT]: v_g, v_d = 0.5, -0.5
        
        # Pivoter la caméra avec Q et D
        if keys[pygame.K_q]: camera.pan_angle += 1.5 * T_ech
        if keys[pygame.K_d]: camera.pan_angle -= 1.5 * T_ech
    # ----------------------------------------------------------------------------------------------------------------------------------
    
    # # Test de validations du modèle cinematique et modele geometrique du robot
    # time_history.append(current_time)
    # v_g,v_d = 1,1
    # v_g_history.append(v_g)
    # v_d_history.append(v_d)
    # Orientation du robot dans le monde
    # orientation_history.append(np.degrees(robot.theta)) 
    
    
    
    robot.move(v_g, v_d, T_ech) # Déplacement du robot
    
    
    
    # -------------- Trajectoire du robot sur la scene 
    curr_pose = robot.get_pose() # Copie de la pose courante du robot
    
    if not trajectoire_points or np.linalg.norm(curr_pose[:2] - trajectoire_points[-1]) > 0.05:
        trajectoire_points.append((curr_pose[0], curr_pose[1]))
        if len(trajectoire_points) > 100:
            trajectoire_points.pop(0)
            
            

    # -------------- Affichage de la scene (environnement de simulation)
    dessiner_interface(ecran, robot.get_pose(), CIBLES)
    
    
    # -------------- Affichage interface Caméra sur la fenetre principale
    cam_surf = pygame.surfarray.make_surface(cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB).swapaxes(1, 0))
    ecran.blit(cam_surf, (850, 10))
    pygame.draw.rect(ecran, WHITE, (850, 10, 320, 240), 5)
    
      
    # -------------- Informations de l'interface de simulation
    status = "RECHERCHE..." if auto_mode and not cible_visible else ("SUIVI CIBLE" if auto_mode else "MANUEL")
    pose = robot.get_pose()
    err_txt = f"{x_err_affichage:.2f} px" if x_err_affichage is not None else "---"
    font = pygame.font.SysFont(None, 22) # Police des caracteres

    info = [("                      VUE DE LA CAMERA", GREEN, 850,260),
            
        (f"ETAT: {status}", YELLOW if auto_mode else GREEN, 850, 300),
        (f"Pose: x={pose[0]:.2f}, y={-pose[1]:.2f}, th={np.degrees(pose[2]):.0f}°", WHITE, 850, 330),
        (f"(Norme) Erreur e = s - s*: {err_txt}", YELLOW, 850, 360),
        (f"Cible(s) restante(s): {len(CIBLES)}", WHITE, 850, 390), 
        (f"V_roueG: {v_g:.2f} m/s, V_roueD: {v_d:.2f} m/s", WHITE, 850, 420),
        (f"W_cam: {w_cam_affichage:.2f} rad/s, W_robot: {w_robot_affichage:.2f} rad/s", WHITE, 850, 450),
        (f"Pan Caméra: {np.degrees(camera.pan_angle):.1f}°", YELLOW, 850, 480),
        ("- Touche 'A' : Switch mode auto/manuel", YELLOW, 850, 550),
        ("- Touche 'R' : Revenir configuration depart", RED, 850, 580),
        ("- Fleches : Direction pour le mode manuel", WHITE, 850, 610),
        ("- Touches 'Q'/'D' : Rotation caméra seule (manuel)", WHITE, 850, 640),
        ("- Touche 'S' : Fixer la cible s*",GREEN,850,670), 
    ]
    for text, color, x, y in info:
        ecran.blit(font.render(text, True, color), (x, y))

    pygame.display.flip()

pygame.quit()
## ===================================================== FIN BOUCLE PRINCIPALE ===================================================




## -------------------------------------------- Affichage graphique des courbes ---------------------------------------------------
if time_history:
    
    # FENETRE 1 : ERREURS VISUELLES
    plt.figure("Erreurs Visuelles", figsize=(10, 5))
    colors = ['r', 'g', 'b', 'm'] 
    linestyles = ['-', '-', '--', '--'] 
    for i in range(4):
        plt.plot(time_history, erreurs_par_point[i], label=f"Point {i}", color=colors[i], linestyle=linestyles[i])
    plt.title("(Norme) Erreur visuelle par point")
    plt.xlabel("Temps (s)"); plt.ylabel("Erreur (pixels)"); plt.legend(); plt.grid(True)
    
    
    # FENETRE 2 : COMMANDES DES VITESSES LINEAIRES DES ROUES
    plt.figure("Commandes linéaires", figsize=(10, 5))
    plt.plot(time_history, v_g_history, label="v_g (Roue Gauche)", color='blue')
    plt.plot(time_history, v_d_history, label="v_d (Roue Droite)", color='orange')
    plt.title("Commandes des vitesses linéaires")
    plt.xlabel("Temps (s)"); plt.ylabel("Vitesses"); plt.legend(); plt.grid(True)
    
    
    # FENETRE 3: COMMANDES DES VITESSES DE ROTATIONS (ROBOT + CAMERA)
    plt.figure("Commandes rotatives", figsize=(10, 5))
    plt.plot(time_history, w_robot_history, label="W_robot", color='green',linestyle='--')
    plt.plot(time_history, w_cam_history, label="W_cam", color='red',linestyle='--')
    plt.title("Commandes des vitesses de rotation")
    plt.xlabel("Temps (s)"); plt.ylabel("Vitesses"); plt.legend(); plt.grid(True)
    
    
    ## FENETRE 4: TRAJECTOIRES DANS LE PLAN IMAGE DES POINTS CIBLES S RELATIVE à S*
    plt.figure("Trajectoires Image", figsize=(8, 8))
    for i in range(4):
        plt.plot(history_u[i], history_v[i], color=colors[i], label=f"Point {i}")       
        # Marqueur point de départ (cercle)
        plt.scatter(history_u[i][0], history_v[i][0], color=colors[i], marker='o')
        # Marqueur  cible s_star (croix rouge)
        u_star = controller.s_star[2*i]
        v_star = controller.s_star[2*i+1]
        plt.scatter(u_star, v_star, color='red', marker='x', s=100)

    plt.gca().invert_yaxis()  # Configuration de l'axe (inversé pour correspondre à la configuration fenetre pygame)
    plt.title("Trajectoires des points dans le plan image (u, v)")
    plt.xlabel("u (pixels)"); plt.ylabel("v (pixels)"); plt.legend(); plt.grid(True)
    # plt.axis('equal') # Important pour ne pas déformer la géométrie
    

    ## FENETRE 5: ORIENTATION DU ROBOT DANS LE MONDE (DISTANCE PARCOURUE)
    plt.figure("Evolution de l'orientation du robot", figsize=(10, 5))
    plt.ylim(-np.degrees(np.pi), np.degrees(np.pi)) # Echelle visuelle fiée à [-180°; 180°]
    plt.plot(time_history,orientation_history,label= "Angle d'orientation")
    # On récupère la valeur actuelle
    current_theta = orientation_history[-1]

    # plt.ylim(-np.degrees(np.pi), np.degrees(np.pi)) # Echelle visuelle fiée à [-180°; 180°]
    plt.title("Orientation du robot (sous IBVS)")
    plt.xlabel("Temps (s)"); plt.ylabel("theta_robot (°)"); plt.legend(); plt.grid(True)
  
    
    
plt.show() # Affiche de l'ensemble des figures