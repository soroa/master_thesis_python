# CONSTANTS
PUSH_UPS = 1
KETTLE_BELL_SWINGS = 2
PULL_UPS = 3
BOX_JUMPS = 4
BURPEES = 5
SQUATS = 6
DEAD_LIFT = 7
KETTLEBELL_SQUAT_PRESS = 8
KETTLEBELL_PRESS = 9
CRUNCHES = 11
WALL_BALLS = 12

WORKOUT = [PUSH_UPS, PULL_UPS, BURPEES, DEAD_LIFT, BOX_JUMPS, SQUATS, CRUNCHES, WALL_BALLS,
           KETTLEBELL_PRESS, KETTLEBELL_SQUAT_PRESS]

DELAYS = {"Desiree Heller": 1000, "Daniel": 1000, "Matteo": 1000, "Tobi bro": 1000, "Renata farkas": 1000,
          "Karol Wojtas": 1000, "Riccardo Rigamonti": 1000, "Damian ga": 4000,
          "Donato pari": 4500}

EXERCISE_CODES_TO_NAME = {
    PUSH_UPS: "Push ups",
    PULL_UPS: "Pull ups",
    BURPEES: "Burpees",
    SQUATS: "Squats",
    BOX_JUMPS: "Box jumps",
    KETTLE_BELL_SWINGS: "Kettle B swings",
    DEAD_LIFT: "Dead lifts",
    KETTLEBELL_SQUAT_PRESS: "KB Squat press",
    KETTLEBELL_PRESS: "KB Press",
    CRUNCHES: "Crunches",
    WALL_BALLS: "Wall balls"
}

EXERCISE_NAME_TO_CLASS_LABEL = {
    "Push ups": 1,
    "Pull ups": 2,
    "Burpees": 3,
    "Dead lifts": 4,
    "Box jumps": 5,
    "Squats": 6,
    "Crunches": 7,
    "Wall balls": 8,
    "KB Squat press": 9,
    "KB Press": 10
}

CLASS_LABEL_TO_AVERAGE_REP_DURATION = [250, 300, 300, 200, 300, 200, 250, 250, 250, 300]

ACCELEROMETER_CODE = 1
GYROSCOPE_CODE = 4
ROTATION_MOTION = 11

SENSOR_POSITION_ANKLE = 0
SENSOR_POSITION_WRIST = 1

SENSOR_TO_VAR_COUNT = {
    ACCELEROMETER_CODE: 3,
    GYROSCOPE_CODE: 3,
    ROTATION_MOTION: 5}

SENSOR_TO_NAME = {
    ACCELEROMETER_CODE: "Accel",
    GYROSCOPE_CODE: "Gyro",
    ROTATION_MOTION: "Rot Motion"}

# TABLE COLUMNS
# READINGS
READING_ID = 0
READING_SENSOR_TYPE = 1
READING_VALUES = 2
READING_EXERCISE_ID = 3
READING_TIMESTAMP = 4
READING_PLACEMENT = 5
READING_REP = 6

READINGS_TABLE_NAME = "sensor_readings"
EXERCISES_TABLE_NAME = "exercises"
WORKOUTS_TABLE_NAME = "workout_sessions"

POSITION_WRIST = "wrist"
POSITION_ANKLE = "ankle"
SMARTWATCH_POSITIONS = [POSITION_WRIST, POSITION_ANKLE]

# Numpy data
WRIST_ACCEL_X = 1
WRIST_ACCEL_Y = 2
WRIST_ACCEL_Z = 3
WRIST_GYRO_X = 4
WRIST_GYRO_Y = 5
WRIST_GYRO_Z = 6
WRIST_ROT_X = 7
WRIST_ROT_Y = 8
WRIST_ROT_Z = 9

ANKLE_ACCEL_X = 10
ANKLE_ACCEL_Y = 11
ANKLE_ACCEL_Z = 12
ANKLE_GYRO_X = 12
ANKLE_GYRO_Y = 14
ANKLE_GYRO_Z = 15
ANKLE_ROT_X = 16
ANKLE_ROT_Y = 17
ANKLE_ROT_Z = 18

EXPERIENCE_LEVEL_MAP = {
    "Viviane des": 2,
    "Ada def": 3,
    "Adrian stetter": 3,
    "Agustin diaz": 2,
    "Alberto sanchez": 3,
    "Ale forino": 1,
    "Alex mil": 1,
    "Alex turicum": 3,
    "Andrea Soro": 2,
    "Anja vont": 2,
    "Anna fertig": 1,
    "Beni fueg": 1,
    "Camilla cav": 2,
    "Conan obri": 2,
    "Corneel van": 2,
    "Damian ga": 2,
    "Daniel luetolf": 1,
    "David geiter": 1,
    "Denis kara": 1,
    "Denis karatwo": 1,
    "Desiree Heller": 3,
    "Donato pari": 2,
    "Fra lam": 2,
    "Georg poll": 1,
    "Karl dei": 3,
    "Karol Wojtas": 1,
    "Lara riparip": 1,
    "Lisa bra": 3,
    "Llorenc mon": 2,
    "Lukas hofm": 2,
    "Marcel feh": 2,
    "Martin butt": 1,
    "Matt senn": 3,
    "Matteo": 1,
    "Max abe": 2,
    "Mike jiang": 1,
    "Muriel haug": 1,
    "Nick lad": 3,
    "Ramon fan": 2,
    "Raphael riedo": 2,
    "Renata farkas": 3,
    "Riccardo rigamonti": 3,
    "Seba curi": 2,
    "Simon bod": 2,
    "Simone pira": 3,
    "Starkaor Hrobjartsson": 2,
    "Tobi bro": 1,
    "Virginia Storni": 2
}

reps_to_keep = {
    "Virginia Storni": {PUSH_UPS: range(2, 16), PULL_UPS: range(2, 11), BURPEES: range(2, 11), DEAD_LIFT: range(2, 22),
                        BOX_JUMPS: None, SQUATS: None,
                        CRUNCHES: range(1, 2), WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Simon bod": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                  CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Alex mil": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                 CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Max abe": {PUSH_UPS: range(1, 15), PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Fra lam": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Marcel feh": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                   CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Starkaor Hrobjartsson": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None,
                              SQUATS: None, CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None,
                              KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Mike jiang": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                   CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Raphael riedo": {PUSH_UPS: range(2, 16), PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None,
                      SQUATS: None,
                      CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Daniel luetolf": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                       CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Matteo": {PUSH_UPS: range(1, 15), PULL_UPS: range(1, 7), BURPEES: range(1, 10), DEAD_LIFT: range(1, 14),
               BOX_JUMPS: range(1, 11), SQUATS: range(2, 15),
               CRUNCHES: range(1, 13), WALL_BALLS: None, KETTLEBELL_PRESS: range(1, 11),
               KETTLEBELL_SQUAT_PRESS: range(1, 13)}
    ,
    "Tobi bro": {PUSH_UPS: range(1, 14), PULL_UPS: range(1, 6), BURPEES: range(1, 9), DEAD_LIFT: range(1, 9),
                 BOX_JUMPS: range(1, 8), SQUATS: range(1, 11),
                 CRUNCHES: range(1, 7), WALL_BALLS: range(1, 10), KETTLEBELL_PRESS: range(1, 9),
                 KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Denis kara": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                   CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Damian ga": {PUSH_UPS: range(1, 12), PULL_UPS: range(1, 5), BURPEES: range(1, 11), DEAD_LIFT: range(1, 12),
                  BOX_JUMPS: range(1, 10), SQUATS: range(1, 14),
                  CRUNCHES: range(1, 13), WALL_BALLS: range(1, 14), KETTLEBELL_PRESS: range(1, 12),
                  KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Muriel haug": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                    CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Matt senn": {PUSH_UPS: range(3, 21), PULL_UPS: range(1, 6), BURPEES: None, DEAD_LIFT: range(2, 21),
                  BOX_JUMPS: range(2, 9), SQUATS: range(1, 2),
                  CRUNCHES: None, WALL_BALLS: range(3, 12), KETTLEBELL_PRESS: range(2, 11),
                  KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Donato pari": {PUSH_UPS: range(1, 9), PULL_UPS: None, BURPEES: range(1, 9), DEAD_LIFT: range(1, 10),
                    BOX_JUMPS: range(1, 12), SQUATS: range(1, 10),
                    CRUNCHES: range(1, 10), WALL_BALLS: range(1, 11), KETTLEBELL_PRESS: range(1, 9),
                    KETTLEBELL_SQUAT_PRESS: range(1, 8)}
    ,
    "Corneel van": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                    CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Alex turicum": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                     CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    " Viviane des": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                     CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Anna fertig": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                    CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Lara riparip": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                     CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Riccardo rigamonti": {PUSH_UPS: range(2, 20), PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None,
                           SQUATS: None, CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None,
                           KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Desiree Heller": {PUSH_UPS: range(1, 13), PULL_UPS: range(1, 6), BURPEES: range(1, 12), DEAD_LIFT: range(1, 15),
                       BOX_JUMPS: range(1, 12),
                       SQUATS: range(1, 15),
                       CRUNCHES: range(1, 15), WALL_BALLS: None, KETTLEBELL_PRESS: range(1, 12),
                       KETTLEBELL_SQUAT_PRESS: range(1, 9)}
    ,
    "Karol Wojtas": {PUSH_UPS: range(1, 11), PULL_UPS: None, BURPEES: range(1, 9), DEAD_LIFT: range(1, 11),
                     BOX_JUMPS: None, SQUATS: range(1, 13),
                     CRUNCHES: range(1, 14), WALL_BALLS: None, KETTLEBELL_PRESS: range(1, 10),
                     KETTLEBELL_SQUAT_PRESS: range(1, 9)}
    ,
    "Camilla cav": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                    CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Martin butt": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                    CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Beni fueg": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                  CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Alberto sanchez": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                        CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Karl dei": {PUSH_UPS: None, PULL_UPS: None, BURPEES: range(1, 9), DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                 CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Llorenc mon": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: None, BOX_JUMPS: None, SQUATS: None,
                    CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Renata farkas": {PUSH_UPS: range(1, 9), PULL_UPS: None, BURPEES: range(1, 10), DEAD_LIFT: range(1, 15),
                      BOX_JUMPS: range(1, 12),
                      SQUATS: range(1, 15),
                      CRUNCHES: None, WALL_BALLS: range(1, 5), KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
    ,
    "Ramon fan": {PUSH_UPS: None, PULL_UPS: None, BURPEES: None, DEAD_LIFT: range(1, 12), BOX_JUMPS: None, SQUATS: None,
                  CRUNCHES: None, WALL_BALLS: None, KETTLEBELL_PRESS: None, KETTLEBELL_SQUAT_PRESS: None}
}
copy_from_path = "./dbs/"
path = "./dbs2/"
numpy_reps_data_path = "./data/np_reps_data/"
numpy_exercises_data_path = "./data/np_exercise_data/"
