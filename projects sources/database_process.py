import sqlite3
import datetime
import logging

conn = sqlite3.connect(r"C:\Users\Даниил\Desktop\КТ1\pythonProject1\BotDatabase.db")
logging.basicConfig(level=logging.INFO)

dictionary_of_disease_db = {
    0 : 'Акне', 1 : 'Актинический кератоз', 2 : 'Дерматит', 3: 'Пигментное пятно',
    4: 'Кандидоз кожи', 5: 'Сосудистые опухоли', 6: 'Вирусная инфекция'
}

def check_if_registrated(user_id):
    cur = conn.cursor()
    cur.execute("""SELECT EXISTS(SELECT 1 FROM UserInfo WHERE user_id = ?)""", (user_id, ))
    res = cur.fetchone()
    return res[0]

def registrate_user(user_id):
    cur = conn.cursor()
    date = datetime.datetime.now()
    cur.execute("""INSERT INTO UserInfo(user_id, register_date) VALUES(?, ?)""", (user_id, date))
    conn.commit()

def allow_to_write(user_id):
    cur = conn.cursor()
    cur.execute("""SELECT allow_write_story FROM UserInfo WHERE user_id = ?""", (user_id, ))
    res = cur.fetchone()
    return res[0]

def find_all_users_to_remind():
    cur = conn.cursor()
    cur.execute("""SELECT user_id FROM UserInfo WHERE allow_to_remind = 0""")
    return cur.fetchall()

def write_disease(user_id, disease_code):
    if(allow_to_write(user_id) == 1):
        date = datetime.datetime.now()
        cur = conn.cursor()
        cur.execute("""INSERT INTO DiseasesRequests(user_id, disease, request_date) VALUES(?, ?, ?)""", (user_id, dictionary_of_disease_db[disease_code], date))
        conn.commit()

def write_state(user_id, state_class):
    if(allow_to_write(user_id) == 1):
        date = datetime.datetime.now()
        cur = conn.cursor()
        cur.execute("""INSERT INTO UserFaceStateTable(user_id, state_class, request_date) VALUES(?, ?, ?)""", (user_id, state_class, date))
        conn.commit()

def write_detected(user_id, count_detected_request):
    if(allow_to_write(user_id) == 1):
        date = datetime.datetime.now()
        cur = conn.cursor()
        cur.execute("""INSERT INTO DetectedTable(user_id, count_detected_disease, request_date) VALUES(?, ?, ?)""", (user_id, count_detected_request, date))
        conn.commit()

def get_disease_story(user_id):
    cur = conn.cursor()
    cur.execute("""SELECT disease, request_date FROM DiseasesRequests WHERE user_id = ? ORDER BY request_date DESC LIMIT 5""", (user_id,))
    return cur.fetchall()

def get_state_story(user_id):
    cur = conn.cursor()
    cur.execute("""SELECT state_class, request_date FROM UserFaceStateTable WHERE user_id = ? ORDER BY request_date DESC LIMIT 5""", (user_id,))
    return cur.fetchall()

def get_detected_story(user_id):
    cur = conn.cursor()
    cur.execute("""SELECT count_detected_disease, request_date FROM DetectedTable WHERE user_id = ? ORDER BY request_date DESC LIMIT 5""", (user_id,))
    return cur.fetchall()