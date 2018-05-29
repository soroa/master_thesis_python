import os
import sqlite3

import numpy as np

from constants import READINGS_TABLE_NAME, WORKOUTS_TABLE_NAME, EXERCISES_TABLE_NAME

tables = [WORKOUTS_TABLE_NAME, EXERCISES_TABLE_NAME, READINGS_TABLE_NAME]

dbs_names = os.listdir("./dbs/")
workout_id_count = 0
exercise_id = 0
reading_id = 0


def split_wrist_ankle_and_merge():
    ankle_readings = []
    wrist_readings = []
    for name in dbs_names:
        if "ankle" in name:
            ankle_readings.append(name)
        elif "wrist" in name:
            wrist_readings.append(name)
    print(wrist_readings)
    print(ankle_readings)
    merge_databases(wrist_readings, "wrist")
    merge_databases(ankle_readings, "ankle")


def merge_databases(names, position):
    first_name = names[0]
    first_db = sqlite3.connect("./dbs/" + first_name)
    first_cursor = first_db.cursor()
    names.remove(first_name)
    for name in names:
        print(name)
        db = sqlite3.connect("./dbs/" + name)
        b_cursor = db.cursor()
        for table in tables:
            print(table)
            b_cursor.execute('SELECT * FROM {tn}'.format(tn=table))
            entries = b_cursor.fetchall()  # Returns the results as a list.
            for e in entries:
                if table is READINGS_TABLE_NAME:
                    values_placeholder = "?,?,?,?,?,?,?"
                elif table is EXERCISES_TABLE_NAME:
                    values_placeholder = "?,?,?,?,?"
                elif table is WORKOUTS_TABLE_NAME:
                    values_placeholder = "?,?,?,?,?"
                first_cursor.execute('INSERT INTO {tn} VALUES({values_placeholder})'.format(tn=table,
                                                                                            values_placeholder=values_placeholder),
                                     e)
    first_db.commit()
    first_cursor.close()
    b_cursor.close()
    os.rename("./dbs/" + first_name, "./dbs/merged_" + position)

def print_participant_name():
    for name in dbs_names:
        db = sqlite3.connect("./dbs/" + name)
        b_cursor = db.cursor()
        b_cursor.execute('SELECT participant FROM {tn}'.format(tn=WORKOUTS_TABLE_NAME))
        participants = np.array(b_cursor.fetchall())
        for p in participants:
            print(name + " " + p[0])


def remove_null_reps(position):
    db = sqlite3.connect("./dbs/merged_" + position)
    cursor = db.cursor()
    cursor.execute('DELETE FROM {tn} WHERE rep_count=0'.format(tn=READINGS_TABLE_NAME))
    db.commit()
    cursor.close()

remove_null_reps("wrist")
remove_null_reps("ankle")

