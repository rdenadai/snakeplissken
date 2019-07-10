FPS = 10

W_WIDTH, W_HEIGHT = 150, 150

BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
CRIMSON = (220, 20, 60)
WHITE = (255, 255, 255)
GREEN = (34, 139, 34)

SNAKE_SIZE = 9
SNAKE_SPEED = 1
SNAKE_SEPARATION = 1
WALL_SIZE = SNAKE_SIZE
APPLE_SIZE = SNAKE_SIZE
APPLE_QTD = 2
APPLE_PRIZE = 10
APPLE_RELOAD_TIME = 3  # IN SECONDS

KEY = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}

# Deep Learning Params
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.02
EPS_DECAY = 500
TARGET_UPDATE = 15
MEM_LENGTH = 100_000
LEARNING_RATE = 1e-4
MOMENTUM = 1e-2
