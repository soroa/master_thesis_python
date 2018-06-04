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