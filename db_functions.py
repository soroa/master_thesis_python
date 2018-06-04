import numpy as np

from constants import WORKOUTS_TABLE_NAME, EXERCISES_TABLE_NAME, READINGS_TABLE_NAME, EXERCISE_CODES_TO_NAME


def get_exercises_ids_for_participants(db, participant):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM {tn} WHERE participant = '{participant}'".format(tn=WORKOUTS_TABLE_NAME,
                                                                                    participant=participant))
    workout_id = np.array(cursor.fetchall())
    cursor.execute(
        'SELECT id FROM {tn} WHERE workout_session_id = {id}'.format(tn=EXERCISES_TABLE_NAME,
                                                                     id=workout_id[0, 0]))
    exercises_ids = cursor.fetchall()
    return np.array(exercises_ids)


def get_exercises_id_for_participant_and_code(db, participant, code):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM {tn} WHERE participant = '{participant}'".format(tn=WORKOUTS_TABLE_NAME,
                                                                                   participant=participant))
    workout_id = np.array(cursor.fetchall())
    cursor.execute(
        'SELECT id FROM {tn} WHERE workout_session_id = {id} AND exercise_code={ex_code}'.format(
            tn=EXERCISES_TABLE_NAME,
            id=workout_id[0, 0], ex_code=code))
    exercises_ids = cursor.fetchall()
    return np.array(exercises_ids)[0]


def get_exercises_ids_for_workout_id(workout_id, db):
    c = db.cursor()
    c.execute('SELECT id FROM {tn} WHERE workout_session_id={wid}'.format(tn=EXERCISES_TABLE_NAME, wid=workout_id))
    ids = np.array(c.fetchall())
    return ids


def get_exercise_codes_for_participants(db, participant):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM {tn} WHERE participant = '{participant}'".format(tn=WORKOUTS_TABLE_NAME,
                                                                                    participant=participant))
    workout_id = np.array(cursor.fetchall())
    cursor.execute(
        'SELECT exercise_code FROM {tn} WHERE workout_session_id = {id}'.format(tn=EXERCISES_TABLE_NAME,
                                                                                id=workout_id[0, 0]))
    exercise_codes = cursor.fetchall()
    return np.array(exercise_codes)


def get_readings_for_exercise(db, id):
    cursor = db.cursor()
    cursor.execute(
        'SELECT * FROM {tn} WHERE exercise_id = {exid}'.format(tn=READINGS_TABLE_NAME,
                                                               exid=id[0]))
    return np.array(cursor.fetchall())


def get_exercise_name_for_id(db, id):
    cursor = db.cursor()
    cursor.execute(
        'SELECT exercise_code FROM {tn} WHERE id = {id}'.format(tn=EXERCISES_TABLE_NAME,
                                                                id=id[0]))
    return EXERCISE_CODES_TO_NAME[np.array(cursor.fetchall())[0, 0]]
